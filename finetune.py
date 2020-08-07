import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib

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
from model.fe_vgg16 import *


from utils import *
from finetuner import Finetuner
from weight_pruner import WeightPruner


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--iterations", type=int, default=30000, help='Iterations to train')
    parser.add_argument("--print_freq", type=int, default=100, help='Frequency of printing training logs')
    parser.add_argument("--test_interval", type=int, default=1000, help='Frequency of testing')
    parser.add_argument("--adv_test_interval", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test', help='Name for the checkpoint')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--const_lr", action='store_true', default=False, help='Use constant learning rate')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1e-2, help='The strength of the L2 regularization on the last linear layer')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--l2sp_lmda", type=float, default=0)
    parser.add_argument("--feat_lmda", type=float, default=0)
    parser.add_argument("--feat_layers", type=str, default='1234', help='Used for DELTA (which layers or stages to match), ResNets should be 1234 and MobileNetV2 should be 12345')
    parser.add_argument("--reinit", action='store_true', default=False, help='Reinitialize before training')
    parser.add_argument("--no_save", action='store_true', default=False, help='Do not save checkpoints')
    parser.add_argument("--swa", action='store_true', default=False, help='Use SWA')
    parser.add_argument("--swa_freq", type=int, default=500, help='Frequency of averaging models in SWA')
    parser.add_argument("--swa_start", type=int, default=0, help='Start SWA since which iterations')
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--teacher_ckpt", type=str, default='')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--shot", type=int, default=-1, help='Number of training samples per class for the training dataset. -1 indicates using the full dataset.')
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--method", default=None, 
        choices=[None, "weight"]
    )
    parser.add_argument("--train_all", default=False, action="store_true")
    parser.add_argument("--ft_begin_module", default=None)
    parser.add_argument("--vgg_output_distill", default=False, action="store_true")
    # Weight prune
    parser.add_argument("--weight_ratio", default=-1, type=float)
    args = parser.parse_args()

    if args.feat_lmda > 0:
        assert args.teacher_ckpt is not None
    args.family_output_dir = args.output_dir
    args.output_dir = osp.join(
        args.output_dir,
        args.name
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params_out_path = osp.join(args.output_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(vars(args), jf, indent=True)
    print(args)

    return args

if __name__=="__main__":
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios
    seed = 98

    train_set = eval(args.dataset)(
        args.datapath, True, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )
    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    # Pre-trained model
    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()
    if args.teacher_ckpt is not "":
        checkpoint = torch.load(args.teacher_ckpt)
        teacher.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded teacher_ckpt from {args.teacher_ckpt}")

    if args.reinit:
        for m in model.modules():
            if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                m.reset_parameters()

    if args.method is None:
        finetune_machine = Finetuner(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    elif args.method == "weight":
        finetune_machine = WeightPruner(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    else:
        raise RuntimeError
    
    
    finetune_machine.train()
    # finetune_machine.adv_eval()

    if args.method is not None:
        finetune_machine.final_check_param_num()


