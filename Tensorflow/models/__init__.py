from .resnet50 import Resnet50
from .InceptionResNetV2 import InceptionResnetV2
from .custom_model import CustomClassifier
from .regnetx002 import RegNetX002
from .transformers import Transformers
from .VIT import create_vit_classifier

MODELS = dict(resnet50=Resnet50,
              inception_resnetv2=InceptionResnetV2,
              custom_model=CustomClassifier,
              regnetx002=RegNetX002,
              transformers=Transformers,
              VIT=create_vit_classifier(vanilla=True)
              # other models
              )


def load_model(model_name, **kwargs):
    return MODELS[model_name](**kwargs).get_model()
