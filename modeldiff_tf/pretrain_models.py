from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.densenet import DenseNet121

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from pdb import set_trace as st

# model = ResNet50(weights='imagenet')
# model = MobileNetV2(weights='imagenet')
# model = NASNetMobile(weights='imagenet')
# model = DenseNet121(weights='imagenet')

pretrain_models = {
    # "resnet50": ResNet50(weights='imagenet'),
    "mbnetv2": MobileNetV2(weights='imagenet'),
    "nasnetmb": NASNetMobile(weights='imagenet'),
    "densenet": DenseNet121(weights='imagenet'),
}

if __name__=="__main__":
    from dataset import MyDataset
    
    d = MyDataset("image")
    samples = d.sample(100)
    samples = np.concatenate(samples)
    samples = preprocess_input(samples)
    
    pred_dict = {}
    for name, model in pretrain_models.items():
        output = model.predict(samples)
        preds = output
        pred_dict[name] = output
        print(name, [x.argmax(axis=-1) for x in preds])

    """    
    for i in range(1):
        for k in pretrain_models.keys():
            out = pred_dict[k][i:i+1]
            top3 = decode_predictions(out, top=5)[0]
            print(k, top3)
    """
