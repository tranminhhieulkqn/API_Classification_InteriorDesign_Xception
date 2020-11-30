import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

from source.ModelXception import ModelXception


class ModelGeneral:
    image_size = 224
    labels = None
    floatx = r'float64'
    labelspath = r'static/label/Labels.npy'
    model_path = r'static/model_cnn/models/'
    weights_path = r'static/model_cnn/weights/'

    def __init__(self):
        self.__load_labels()
        if False:
            self.__load_model()
        else:
            self.__load_weight()

    @classmethod
    def __load_labels(cls):
        cls.labels = np.load(cls.labelspath)
        print("Load labels successfully!")

    @classmethod
    def get_lables(cls):
        return cls.labels

    @classmethod
    def __load_weight(cls):
        # Load model Xception
        ModelXception.set_path(model_path=None,
                               weight_path=cls.weights_path + r'WeightsXception.h5')
        cls.model_xception = ModelXception.get_model()
        print("Load models successfully!")

    @classmethod
    def __load_model(cls):
        # Load model Xception
        ModelXception.set_path(model_path=cls.model_path + r'ModelXception.h5',
                               weight_path=None)
        cls.model_xception = ModelXception.get_model()
        print("Load models successfully!")

    # @classmethod # Remember to install opencv
    # def __resize_image(cls, image_path, image_size=224):
    #     img_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     img_arr = cv2.resize(img_arr, (image_size, image_size))  # Reshaping images to preferred size
    #     img_arr = np.array(img_arr, dtype=cls.floatx) / 255
    #     img_arr = img_arr.reshape(-1, image_size, image_size, 3)
    #     return img_arr

    @classmethod
    def __resize_image(cls, image_request, image_size=224):
        # resize the input image and preprocess it
        img_arr = Image.open(image_request).convert("RGB").resize((image_size, image_size))
        img_arr = np.array([img_to_array(img_arr)[..., ::-1]], dtype=cls.floatx) / 255
        return img_arr

    @classmethod
    def __prediction_classify(cls, model, image_request):
        img_arr = cls.__resize_image(image_request, cls.image_size)
        predictions = model.predict(img_arr)
        classes = np.argmax(predictions, axis=1)
        return [np.round(predictions[0] * 100, 2), cls.labels[classes[0]]]

    @classmethod
    def prediction(cls, model="Xception", image_request=None):
        if model == "Xception":
            return cls.__prediction_classify(model=ModelXception.get_model(),
                                             image_request=image_request)
        else:
            return None
