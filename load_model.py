import argparse
from pdb import set_trace as st

import torch
from torchvision import transforms

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67
from dataset.stanford_dog import SDog120
from dataset.caltech256 import Caltech257Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_mobilenet import mbnetv2_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101
from model.fe_mobilenet import fembnetv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    args = parser.parse_args()
    
    return args

def main():
    args = get_args()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
    )
    
    
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=test_set.num_classes
    )

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    
if __name__=="__main__":
    main()