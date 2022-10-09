"""
This module contains models for resent50
"""
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model


class Resnet50:
    def __init__(self, img_size=48, channels=3, **kwargs):
        self.input_shape = (img_size, img_size, channels)

    def get_model(self) -> Model:
        # using ResNet50
        inputs = Input(self.input_shape)
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        for layer in base_model.layers:
            layer.trainable = False
        x = Flatten()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(8, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        return model
