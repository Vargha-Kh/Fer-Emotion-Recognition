
"""
This module contains models for regnety320
"""
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import regnet
from tensorflow.keras.models import Sequential, Model


class RegNetY320:
    def __init__(self, img_w=200, img_h=200, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self) -> Model:
        # using RegNet
        INPUT_SHAPE = self.input_shape

        # get the pretrained model
        base_model = regnet.RegNetY320(
            model_name='regnety320',
            include_top=True,
            include_preprocessing=True,
            weights='imagenet',
            input_shape=INPUT_SHAPE,
            classes=62438,
        )
        base_model.add(Dense(62438, activation='softmax'))
        base_model.summary()
        return base_model
