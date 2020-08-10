import os
import sys
import time
import argparse
import json
import random
import logging
import pathlib
import functools
import torch
import torchvision
import torchvision.models as models
import numpy as np
from pdb import set_trace as st
import copy


from dataset.mit67 import MIT67
from dataset.stanford_dog import SDog120
from dataset.flower102 import Flower102
from dataset.caltech256 import Caltech257Data
from dataset.stanford_40 import Stanford40Data
from dataset.cub200 import CUB200Data

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_mobilenet import mbnetv2_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101
from model.fe_mobilenet import fembnetv2
from model.fe_vgg16 import *

SEED = 98
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 64
TRAIN_ITERS = 100000    # TODO update the number of iterations
TRANSFER_ITERS = 30000
QUANTIZATION_ITERS = 30000  # may be useless
PRUNE_ITERS = 30000
DISTILL_ITERS = 30000
STEAL_ITERS = 30000

# for debug
# TRANSFER_ITERS = 100
# PRUNE_ITERS = 100
# DISTILL_ITERS = 100
# STEAL_ITERS = 10000

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


def base_args():
    args = argparse.Namespace()
    args.const_lr = False
    args.batch_size = BATCH_SIZE
    args.lr = 5e-3
    args.print_freq = 100
    args.label_smoothing = 0
    args.vgg_output_distill = False
    args.reinit = False
    args.l2sp_lmda = 0
    args.train_all = False
    args.ft_begin_module = None
    args.momentum = 0
    args.weight_decay = 1e-4
    args.beta = 1e-2
    args.feat_lmda = 0
    args.test_interval = 1000
    args.adv_test_interval = -1
    args.feat_layers = '1234'
    args.no_save = False
    args.steal = False
    
    return args

class ModelWrapper:
    def __init__(self, benchmark, teacher_wrapper, trans_str,
                 arch_id=None, dataset_id=None, gen_if_not_exist=True):
        self.logger = logging.getLogger('ModelWrapper')
        self.benchmark = benchmark
        self.teacher_wrapper = teacher_wrapper
        self.trans_str = trans_str
        self.arch_id = arch_id if arch_id else teacher_wrapper.arch_id
        self.dataset_id = dataset_id if dataset_id else teacher_wrapper.dataset_id
        self.gen_if_not_exist = teacher_wrapper.gen_if_not_exist if teacher_wrapper else gen_if_not_exist
        self.torch_model_path = os.path.join(benchmark.models_dir, f'{self.__str__()}')
        assert self.arch_id is not None
        assert self.dataset_id is not None
        if self.gen_if_not_exist and not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)

    def __str__(self):
        teacher_str = '' if self.teacher_wrapper is None else self.teacher_wrapper.__str__()
        return f'{teacher_str}{self.trans_str}-'

    def torch_model_exists(self):
        ckpt_path = os.path.join(self.torch_model_path, 'final_ckpt.pth')
        return os.path.exists(ckpt_path)

    def save_torch_model(self, torch_model):
        if not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        ckpt_path = os.path.join(self.torch_model_path, 'final_ckpt.pth')
        torch.save(
            {'state_dict': torch_model.state_dict()},
            ckpt_path,
        )

    @lazy_property
    def torch_model(self):
        """
        load the model object from torch_model_path
        :return: torch.nn.Module object
        """
        if self.dataset_id == 'ImageNet':
            num_classes = 1000
        else:
            num_classes = self.benchmark.get_dataloader(self.dataset_id).dataset.num_classes
        torch_model = eval(f'{self.arch_id}_dropout')(
            pretrained=False,
            num_classes=num_classes
        )
        
        ckpt = torch.load(os.path.join(self.torch_model_path, 'final_ckpt.pth'))
        torch_model.load_state_dict(ckpt['state_dict'])
        return torch_model

    def load_saved_weights(self, torch_model):
        """
        load weights in the latest checkpoint to torch_model
        """
        ckpt_path = os.path.join(self.torch_model_path, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            torch_model.load_state_dict(ckpt['state_dict'])
            self.logger.info('load_saved_weights: loaded a previous checkpoint')
        else:
            self.logger.info('load_saved_weights: no previous checkpoint found')
        return torch_model

    def transfer(self, dataset_id, tune_ratio=0.1, iters=TRANSFER_ITERS):
        trans_str = f'transfer({dataset_id},{tune_ratio})'
        # model_wrapper is the wrapper of the student model
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            dataset_id=dataset_id
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            teacher_model = self.torch_model
            # transfer the model to another dataset as specified by dataset_id, fine-tune the last tune_ratio% layers
            train_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='train')
            test_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='test')
            student_model = eval(f'{model_wrapper.arch_id}_dropout')(
                pretrained=True,
                num_classes=train_loader.dataset.num_classes
            )
            # FIXME copy state_dict from teacher to student, ignore the final layer
            # student_model.load_state_dict(teacher_model.state_dict(), strict=False)
            
            from finetuner import Finetuner
            args = base_args()
            args.iterations = iters
            args.output_dir = model_wrapper.torch_model_path
            args.network = model_wrapper.arch_id
            args.ft_ratio = tune_ratio
            
            student_model = model_wrapper.load_saved_weights(student_model) # continue training
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            model_wrapper.save_torch_model(student_model)
        return model_wrapper

    def quantize(self, dtype='qint8'):
        """
        do post-training quantization on the model
        :param method: int8 or float16
        :return:
        """
        trans_str = f'quantize({dtype})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            dtype = torch.qint8 if dtype == 'qint8' else torch.float16
            torch_model = torch.quantization.quantize_dynamic(self.torch_model, dtype=dtype)
            model_wrapper.save_torch_model(torch_model)
        return model_wrapper

    def prune(self, prune_ratio=0.1, iters=PRUNE_ITERS):
        trans_str = f'prune({prune_ratio})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            teacher_model = self.torch_model
            # prune prune_ratio% weights of the teacher model
            train_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='train')
            test_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='test')
            student_model = copy.deepcopy(teacher_model)
            
            from finetuner import Finetuner
            args = base_args()
            args.output_dir = model_wrapper.torch_model_path
            args.network = model_wrapper.arch_id
            args.method = "weight"
            args.weight_ratio = prune_ratio
            args.iterations = iters
            
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            model_wrapper.save_torch_model(student_model)
            
            
        return model_wrapper

    def distill(self, iters=DISTILL_ITERS):
        trans_str = f'distill()'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            teacher_model = self.torch_model
            # distill the knowledge of teacher to a reinitialized model
            train_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='train')
            test_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='test')
            student_model = eval(f'{model_wrapper.arch_id}_dropout')(
                pretrained=True,
                num_classes=train_loader.dataset.num_classes
            )
            
            from finetuner import Finetuner
            args = base_args()
            args.iterations = iters
            args.output_dir = model_wrapper.torch_model_path
            args.network = model_wrapper.arch_id
            args.feat_lmda = 5e0
            
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            model_wrapper.save_torch_model(student_model)

        return model_wrapper

    def steal(self, arch_id, iters=STEAL_ITERS):
        trans_str = f'steal({arch_id})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            arch_id=arch_id
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            teacher_model = self.torch_model
            # use output distillation to transfer teacher knowledge to another architecture
            train_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='train')
            test_loader = self.benchmark.get_dataloader(model_wrapper.dataset_id, split='test')
            student_model = eval(f'{arch_id}_dropout')(
                pretrained=True,
                num_classes=train_loader.dataset.num_classes
            )
            
            from finetuner import Finetuner
            args = base_args()
            args.iterations = iters
            args.output_dir = model_wrapper.torch_model_path
            args.network = arch_id
            args.steal = True
            args.steal_alpha = 1
            args.temperature = 1
            
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            model_wrapper.save_torch_model(student_model)
        return model_wrapper

    @lazy_property
    def accuracy(self):
        """
        evaluate the model accuracy on the dataset
        :return: a float number
        """
        # TODO implement this
        model = self.torch_model.to('cuda')
        test_loader = self.benchmark.get_dataloader(self.dataset_id)

        with torch.no_grad():
            model.eval()
            total = 0
            top1 = 0
            for i, (batch, label) in enumerate(test_loader):
                batch, label = batch.to('cuda'), label.to('cuda')
                total += batch.size(0)
                out = model(batch)
                _, pred = out.max(dim=1)
                top1 += int(pred.eq(label).sum().item())
        return float(top1) / total * 100


class ImageBenchmark:
    def __init__(self, datasets_dir='data', models_dir='models'):
        self.logger = logging.getLogger('ImageBench')
        self.datasets_dir = datasets_dir
        self.models_dir = models_dir
        self.datasets = ['MIT67', 'Flower102', 'SDog120']
        self.archs = ['mbnetv2', 'resnet18', 'vgg16_bn']
        # For debug
        # self.datasets = ['MIT67']
        # self.archs = ['resnet18']

    def get_dataloader(self, dataset_id, split='train', batch_size=BATCH_SIZE, shot=-1):
        """
        Get the torch Dataset object
        :param dataset_id: the name of the dataset, should also be the dir name and the class name
        :param split: train or test
        :param batch_size: batch size
        :param shot: number of training samples per class for the training dataset. -1 indicates using the full dataset
        :return: torch.utils.data.DataLoader instance
        """
        datapath = os.path.join(self.datasets_dir, dataset_id)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        from torchvision import transforms
        if split == 'train':
            dataset = eval(dataset_id)(
                datapath, True, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                shot, seed, preload=False
            )
        else:
            dataset = eval(dataset_id)(
                datapath, False, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                shot, seed, preload=False
            )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=False
        )
        return data_loader

    def load_pretrained(self, arch_id, gen_if_not_exist=True):
        """
        Get the model pretrained on imagenet
        :param arch_id: the name of the arch
        :param gen_if_not_exist: generate a new model if the model does not exist
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'pretrain({arch_id},ImageNet)',
            arch_id=arch_id,
            dataset_id='ImageNet',
            gen_if_not_exist=gen_if_not_exist
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            # load pretrained model as specified by arch_id and save it to model path
            torch_model = eval(f'{arch_id}_dropout')(
                pretrained=True,
                num_classes=1000
            )
            model_wrapper.save_torch_model(torch_model)
        return model_wrapper

    def load_trained(self, arch_id, dataset_id, gen_if_not_exist=True, iters=TRAIN_ITERS):
        """
        Get the model with architecture arch_id trained on dataset dataset_id
        :param arch_id: the name of the arch
        :param dataset_id: the name of the dataset
        :param gen_if_not_exist: generate a new model if the model does not exist
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'train({arch_id},{dataset_id})',
            arch_id=arch_id,
            dataset_id=dataset_id,
            gen_if_not_exist=gen_if_not_exist
        )
        if not model_wrapper.torch_model_exists() and model_wrapper.gen_if_not_exist:
            self.logger.info(f'generating: {model_wrapper.__str__()}')
            train_loader = self.get_dataloader(model_wrapper.dataset_id, split='train')
            test_loader = self.get_dataloader(model_wrapper.dataset_id, split='test')
            torch_model = eval(f'{arch_id}_dropout')(
                pretrained=False,
                num_classes=train_loader.dataset.num_classes
            )
            # train the model from scratch
            from finetuner import Finetuner
            args = base_args()
            args.iterations = iters
            args.output_dir = model_wrapper.torch_model_path
            args.network = model_wrapper.arch_id
            args.ft_ratio = 1

            torch_model = model_wrapper.load_saved_weights(torch_model)  # continue training
            finetuner = Finetuner(
                args,
                torch_model, torch_model,
                train_loader, test_loader,
            )
            finetuner.train()
            model_wrapper.save_torch_model(torch_model)
        return model_wrapper

    def build_models(self):
        """
        build the benchmark dataset
        :return: a stream of ModelWrapper instances
        """
        source_models = []
        # load pretrained source models
        for arch in self.archs:
            source_model = self.load_pretrained(arch, gen_if_not_exist=True)
            source_models.append(source_model)
        quantization_dtypes = ['qint8', 'float16']
        prune_ratios = [0.2, 0.5, 0.8]
        transfer_tune_ratios = [0.1, 0.5, 1]
        # for debug
        # prune_ratios = [0.2]
        # transfer_tune_ratios = [0.1]

        transfer_models = []
        # - M_{i,x}/{trans-y,l} -- Transfer M_{i,x} to D_y by fine-tuning from l-st layer
        for source_model in source_models:
            for dataset_id in self.datasets:
                if dataset_id == source_model.dataset_id:
                    continue
                for tune_ratio in transfer_tune_ratios:
                    transfer_model = source_model.transfer(dataset_id=dataset_id, tune_ratio=tune_ratio)
                    transfer_models.append(transfer_model)
                    yield transfer_model
        
        # - M_{i,x}/{quant-dyn/int/flo} -- Compress M_{i,x} with dynamic_range / integer / float16 quantization
        # for debug
        for transfer_model in transfer_models:
            for quantization_dtype in quantization_dtypes:
                yield transfer_model.quantize(dtype=quantization_dtype)

        # - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
        for transfer_model in transfer_models:
            for pr in prune_ratios:
                yield transfer_model.prune(prune_ratio=pr)
        
        # - M_{i,x}/{distill} -- Distill M_{i,x}
        for transfer_model in transfer_models:
            yield transfer_model.distill()

        # - M_{i,x}/{steal-j} -- Steal M_{i,x} to A_j
        for transfer_model in transfer_models:
            for arch_id in self.archs:
                yield transfer_model.steal(arch_id=arch_id)


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Build micro benchmark.")

    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default='data',
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default='models',
                        help="Path to the dir of benchmark models.")
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = parse_args()
    bench = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    for model in bench.build_models():
        print(f'loaded model: {model}')
    # print(benchmark.model2variations)

