from collections.abc import Iterable
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
import math


class CosineDecayRestarts(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, first_decay_steps, alpha=0.0, t_mul=2.0, m_mul=1.0):
        super(CosineDecayRestarts, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.alpha = alpha
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.batch_step = 0

    def on_train_batch_begin(self, step, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(self.batch_step, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        self.batch_step += 1

    def schedule(self, step, lr):
        def compute_step(completed_fraction, geometric=False):
            """Helper for `cond` operation."""
            if geometric:
                i_restart = math_ops.floor(
                    math_ops.log(1.0 - completed_fraction * (1.0 - self.t_mul)) /
                    math_ops.log(self.t_mul))

                sum_r = (1.0 - self.t_mul ** i_restart) / (1.0 - self.t_mul)
                completed_fraction = (completed_fraction - sum_r) / self.t_mul ** i_restart

            else:
                i_restart = math_ops.floor(completed_fraction)
                completed_fraction -= i_restart

            return i_restart, completed_fraction

        completed_fraction = step / self.first_decay_steps

        i_restart, completed_fraction = control_flow_ops.cond(
            math_ops.equal(self.t_mul, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True))

        m_fac = self.m_mul ** i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
            constant_op.constant(math.pi) * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha

        return math_ops.multiply(self.initial_learning_rate, decayed)


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
