
"""
This module contains models for RegNetX002
"""

from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import regnet
from tensorflow.keras.models import Sequential, Model


class RegNetX002:
    def __init__(self, img_size=48, channels=1, **kwargs):
        self.input_shape = (img_size, img_size, channels)

    def get_model(self) -> Model:
        # using RegNet
        INPUT_SHAPE = self.input_shape

        # get the pretrained model
        base_model = regnet.RegNetX002(
            model_name='regnetx002',
            include_top=False,
            include_preprocessing=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=INPUT_SHAPE,
        )
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))
        base_model.summary()
        return base_model
