import threading

from keras.applications import Xception
from keras.layers import Dense, MaxPool2D, Flatten
from keras.models import Sequential, load_model


class ModelXception:
    __instance_model = None  # Instance model
    model_path = None  # Model path model
    weight_path = None  # Weight path model
    labels = 5
    image_size = 224

    def __init__(self):
        if (ModelXception.__instance_model is None) \
                and ((ModelXception.model_path is not None) or (ModelXception.weight_path is not None)):
            if ModelXception.model_path is not None:
                ModelXception.__instance_model = load_model(self.model_path)
            else:
                model = ModelXception.__create_model()
                model.load_weights(ModelXception.weight_path)
                ModelXception.__instance_model = model
        else:
            ModelXception.__instance_model = self

    @staticmethod
    def set_path(model_path=None, weight_path=None, labels=5, image_size=224):
        ModelXception.labels = labels
        ModelXception.image_size = image_size
        ModelXception.model_path = model_path
        ModelXception.weight_path = weight_path
        if model_path is None and weight_path is None:
            print(r'The path is not correct!')

    @classmethod
    def __create_model(cls):
        pre_trained_model = Xception(input_shape=(cls.image_size, cls.image_size, 3),
                                     include_top=False,
                                     weights="imagenet")
        model = Sequential([
            pre_trained_model,
            MaxPool2D((2, 2), strides=2),
            Flatten(),
            Dense(cls.labels, activation='softmax')])
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def get_model():
        if ModelXception.__instance_model is None:
            with threading.Lock():
                if ModelXception.__instance_model is None:
                    ModelXception()
        return ModelXception.__instance_model
