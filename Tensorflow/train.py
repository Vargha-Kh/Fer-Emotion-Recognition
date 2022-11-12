from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import load_model
from hp import load_hps
from tensorflow.keras import metrics, optimizers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from datasets.dataset import Dataset
from plotting import plot
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from cosine_annealing import CosineAnnealingScheduler
# from numba import cuda 

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
tf.keras.backend.clear_session()
# # device = cuda.get_current_device()
# device.reset()

def model_evaluation(model, test_gen):
    pred = (model.predict(test_gen) > 0.5).astype("int32")
    y_test = test_gen.labels
    print("Model Prediction:")
    print('Classification report:\n', classification_report(y_test, pred))
    print('Accuracy score:\n', accuracy_score(y_test, pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
    print("\nModel Evaluation on Test Set:")
    model.evaluate(test_gen, batch_size=200)


def train():
    hps = load_hps(dataset_dir="./fer2013/", model_name='custom_model', n_epochs=300, batch_size=512,
                   learning_rate=0.001,
                   lr_reducer_factor=0.1,
                   lr_reducer_patience=12, img_size=48, split_size=0.25, framework='keras')
    model = load_model(model_name=hps['model_name'])
    #model = models.load_model('./best_model.h5')
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.CategoricalAccuracy(name='categorical_accuracy'),
        metrics.Accuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]
    wd = 1e-4 * hps['learning_rate']
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=hps['learning_rate'], decay=1e-6),
                  metrics=["accuracy"])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=hps['lr_reducer_factor'],
                                  patience=hps['lr_reducer_patience'],
                                  verbose=1,
                                  min_delta=0.00001)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    
    # cosine_decay_restarts = optimizers.schedules.CosineDecayRestarts(hps['learning_rate'], 200, t_mul=1.0)
    cosine_decay_restarts = CosineAnnealingScheduler(T_max=200, eta_max=1e-2, eta_min=1e-4)
    tensorboard_callback = TensorBoard(log_dir="./logs")

    model_checkpoint_acc = ModelCheckpoint("./best_model.h5", monitor='val_accuracy', save_best_only=True,
                                           verbose=1)
    # model_checkpoint_loss = ModelCheckpoint("./best_model_loss_{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True,
    #                                         verbose=1)

    callbacks = [reduce_lr, model_checkpoint_acc, early_stopping, tensorboard_callback]

    if hps['framework'] == 'tensorflow':
        train_ds, val_ds = Dataset.tensorflow_preprocess(dataset_dir=hps['dataset_dir'],
                                                                  img_size=hps['img_size'],
                                                                  batch_size=hps['batch_size'],
                                                                  train_augment=True, val_augment=True, split_size=hps['split_size'])
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=hps['n_epochs'],
            batch_size=hps['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        plot(history)
        test_datagen = ImageDataGenerator(rescale=1 / 255)
        test_generator = test_datagen.flow_from_directory(hps['dataset_dir'] + "val/",
                                                          target_size=(hps['img_size'], hps['img_size']),
                                                          batch_size=hps['batch_size'], class_mode='categorical')
        model_evaluation(model, test_generator)
    elif hps['framework'] == 'keras':
        train_generator, validation_generator = Dataset.keras_preprocess(
            dataset_dir=hps['dataset_dir'],
            img_size=hps['img_size'],
            batch_size=hps['batch_size'],
            augment=True, split_size=hps['split_size'])
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // hps['batch_size'],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // hps['batch_size'],
            epochs=hps['n_epochs'],
            callbacks=callbacks,
            verbose=2)
        plot(history)
        model_evaluation(model, validation_generator)


if __name__ == '__main__':
    train()
