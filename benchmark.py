import os
import sys
import time
import argparse
import json
import random
import logging
import pathlib
import re
import functools
import torch
import torch.nn as nn
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
from finetuner import Finetuner
from weight_pruner import WeightPruner


SEED = 98
INPUT_SHAPE = (3, 224, 224)
BATCH_SIZE = 64
TRAIN_ITERS = 100000    # TODO update the number of iterations
DEFAULT_ITERS = 10000   # TODO change back to 30000
TRANSFER_ITERS = DEFAULT_ITERS
QUANTIZE_ITERS = DEFAULT_ITERS  # may be useless
PRUNE_ITERS = DEFAULT_ITERS
DISTILL_ITERS = DEFAULT_ITERS
STEAL_ITERS = DEFAULT_ITERS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONTINUE_TRAIN = False  # whether to continue previous training

# for debug
TRAIN_ITERS = 10000
TRANSFER_ITERS = 10000
PRUNE_ITERS = 100
DISTILL_ITERS = 10000
STEAL_ITERS = 10000

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
                 arch_id=None, dataset_id=None, iters=100, fc=True):
        self.logger = logging.getLogger('ModelWrapper')
        self.benchmark = benchmark
        self.teacher_wrapper = teacher_wrapper
        self.trans_str = trans_str
        self.arch_id = arch_id if arch_id else teacher_wrapper.arch_id
        self.dataset_id = dataset_id if dataset_id else teacher_wrapper.dataset_id
        self.torch_model_path = os.path.join(benchmark.models_dir, f'{self.__str__()}')
        self.iters = iters
        self.fc = fc
        assert self.arch_id is not None
        assert self.dataset_id is not None

    def __str__(self):
        teacher_str = '' if self.teacher_wrapper is None else self.teacher_wrapper.__str__()
        return f'{teacher_str}{self.trans_str}-'
    
    def name(self):
        return self.__str__()

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
        
        if self.fc:
            torch_model = eval(f'{self.arch_id}_dropout')(
                pretrained=False,
                num_classes=num_classes
            )
        else:
            torch_model = eval(f'fe{self.arch_id}')(
                pretrained=False,
                num_classes=num_classes
            )
        
        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        params = m.group(2).split(',')
        if method == 'quantize':
            dtype = params[0]
            dtype = torch.qint8 if dtype == 'qint8' else torch.float16
            torch_model = torch.quantization.quantize_dynamic(torch_model, dtype=dtype)
        ckpt = torch.load(os.path.join(self.torch_model_path, 'final_ckpt.pth'))
        torch_model.load_state_dict(ckpt['state_dict'])
        return torch_model
    
    @lazy_property
    def torch_model_on_device(self):
        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        if method == "quantize":
            return self.torch_model.to("cpu")
        else:
            return self.torch_model.to(DEVICE)

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

    @lazy_property
    def input_shape(self):
        return INPUT_SHAPE

    def get_seed_inputs(self, n, rand=False):
        if rand:
            batch_input_size = (n, *INPUT_SHAPE)
            images = np.random.normal(size=batch_input_size).astype(np.float32)
        else:
            dataset_id = 'MIT67' if self.dataset_id == 'ImageNet' else self.dataset_id
            train_loader = self.benchmark.get_dataloader(
                dataset_id, split='train', batch_size=n, shuffle=True)
            images, labels = next(iter(train_loader))
            images = images.to('cpu').numpy()
        return images

    def batch_forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        m = re.match(r'(\S+)\((\S*)\)', self.trans_str)
        method = m.group(1)
        if method == "quantize":
            inputs = inputs.to("cpu")
        else:
            inputs = inputs.to(DEVICE)
        self.torch_model_on_device.eval()
        with torch.no_grad():
            return self.torch_model_on_device(inputs)

    def list_tensors(self):
        pass

    def batch_forward_with_ir(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        idx = 0
        hook_handles = []
        module_ir = {}
        model = self.torch_model

        def register_hooks(module):
            def hook(module, input, output):
                global idx
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_name = f"{class_name}/{idx:03d}"
                idx += 1
                module_ir[module_name] = output.numpy()

            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(hook)
                hook_handles.append(handle)

        def remove_hooks():
            for h in hook_handles:
                h.remove()

        model.eval()
        with torch.no_grad():
            model.apply(register_hooks)
            outputs = model(inputs)
            remove_hooks()
        return module_ir

    def gen_model(self, regenerate=False):
        """
        generate the torch model
        :return:
        """
        trans_str = self.trans_str
        if not regenerate and self.torch_model_exists():
            self.logger.info(f'model already exists: {self.__str__()}')
            return
        self.logger.info(f'generating model for: {self.__str__()}')
        m = re.match(r'(\S+)\((\S*)\)', trans_str)
        method = m.group(1)
        params = m.group(2).split(',')

        if regenerate and os.path.exists(self.torch_model_path):
            import shutil
            shutil.rmtree(self.torch_model_path)
        if not os.path.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)

        teacher_model = None
        if self.teacher_wrapper:
            self.teacher_wrapper.gen_model()
            teacher_model = self.teacher_wrapper.torch_model
        train_loader = self.benchmark.get_dataloader(self.dataset_id, split='train')
        test_loader = self.benchmark.get_dataloader(self.dataset_id, split='test')

        args = base_args()
        args.iterations = self.iters
        args.output_dir = self.torch_model_path

        if method == 'pretrain':
            # load pretrained model as specified by arch_id and save it to model path
            arch_id = params[0]
            dataset_id = params[1]
            if dataset_id != 'ImageNet':
                self.logger.warning(f'gen_model: pretrained model on {dataset_id} not supported')
            torch_model = eval(f'{arch_id}_dropout')(
                pretrained=True,
                num_classes=1000
            )
            self.save_torch_model(torch_model)
        elif method == 'train':
            # train the model from scratch
            arch_id = params[0]
            dataset_id = params[1]
            torch_model = eval(f'{arch_id}_dropout')(
                pretrained=False,
                num_classes=train_loader.dataset.num_classes
            )
            args.network = self.arch_id
            args.ft_ratio = 1
            args.reinit = True
            args.lr = 1e-2
            args.weight_decay = 5e-3
            args.momentum = 0.9

            if CONTINUE_TRAIN:
                torch_model = self.load_saved_weights(torch_model)  # continue training
            finetuner = Finetuner(
                args,
                torch_model, torch_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(torch_model)
        elif method == 'transfer':
            # transfer the teacher to a dataset as specified by dataset_id, fine-tune the last tune_ratio% layers
            dataset_id = params[0]
            tune_ratio = float(params[1])
            student_model = eval(f'{self.arch_id}_dropout')(
                pretrained=True,
                num_classes=train_loader.dataset.num_classes
            )
            # FIXME copy state_dict from teacher to student, ignore the final layer
            # student_model.load_state_dict(teacher_model.state_dict(), strict=False)

            args.network = self.arch_id
            args.ft_ratio = tune_ratio

            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model)
        elif method == 'quantize':
            dtype = params[0]
            dtype = torch.qint8 if dtype == 'qint8' else torch.float16
            student_model = torch.quantization.quantize_dynamic(teacher_model, dtype=dtype)
            self.save_torch_model(student_model)
        elif method == 'prune':
            prune_ratio = float(params[0])
            student_model = copy.deepcopy(teacher_model)

            args.network = self.arch_id
            args.method = "weight"
            args.weight_ratio = prune_ratio

            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training

            finetuner = WeightPruner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model)
            finetuner.final_check_param_num()
        elif method == 'distill':
            student_model = eval(f'{self.arch_id}_dropout')(
                pretrained=False,
                num_classes=train_loader.dataset.num_classes
            )
            args.network = self.arch_id
            args.feat_lmda = 5e0
            args.reinit = True
            args.lr = 1e-2
            args.weight_decay = 5e-3
            args.momentum = 0.9

            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training

            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model)
        elif method == 'steal':
            arch_id = params[0]
            # use output distillation to transfer teacher knowledge to another architecture
            student_model = eval(f'{arch_id}_dropout')(
                pretrained=False,
                num_classes=train_loader.dataset.num_classes
            )

            args.network = arch_id
            args.steal = True
            args.reinit = True
            args.steal_alpha = 1
            args.temperature = 1
            args.lr = 1e-2
            args.weight_decay = 5e-3
            args.momentum = 0.9

            if CONTINUE_TRAIN:
                student_model = self.load_saved_weights(student_model)  # continue training
                
            finetuner = Finetuner(
                args,
                student_model, teacher_model,
                train_loader, test_loader,
            )
            finetuner.train()
            self.save_torch_model(student_model)
        else:
            raise RuntimeError(f'unknown transformation: {method}')

    def transfer(self, dataset_id, tune_ratio=0.1, iters=TRANSFER_ITERS):
        trans_str = f'transfer({dataset_id},{tune_ratio})'
        # model_wrapper is the wrapper of the student model
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            dataset_id=dataset_id,
            iters=iters
        )
        return model_wrapper

    def quantize(self, dtype='qint8'):
        """
        do post-training quantization on the model
        :param dtype: qint8 or float16
        :return:
        """
        trans_str = f'quantize({dtype})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str
        )
        return model_wrapper

    def prune(self, prune_ratio=0.1, iters=PRUNE_ITERS):
        trans_str = f'prune({prune_ratio})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            iters=iters
        )
        return model_wrapper

    def distill(self, iters=DISTILL_ITERS):
        trans_str = f'distill()'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            iters=iters
        )
        return model_wrapper

    def steal(self, arch_id, iters=STEAL_ITERS):
        trans_str = f'steal({arch_id})'
        model_wrapper = ModelWrapper(
            benchmark=self.benchmark,
            teacher_wrapper=self,
            trans_str=trans_str,
            arch_id=arch_id,
            iters=iters
        )
        return model_wrapper

    @lazy_property
    def accuracy(self):
        """
        evaluate the model accuracy on the dataset
        :return: a float number
        """
        # TODO implement this
        model = self.torch_model.to(DEVICE)
        test_loader = self.benchmark.get_dataloader(self.dataset_id, split='test')

        with torch.no_grad():
            model.eval()
            total = 0
            top1 = 0
            for i, (batch, label) in enumerate(test_loader):
                batch, label = batch.to(DEVICE), label.to(DEVICE)
                total += batch.size(0)
                out = model(batch)
                _, pred = out.max(dim=1)
                top1 += int(pred.eq(label).sum().item())
        # print(top1, total)
        return float(top1) / total * 100


class ImageBenchmark:
    def __init__(self, datasets_dir='data', models_dir='models'):
        self.logger = logging.getLogger('ImageBench')
        self.datasets_dir = datasets_dir
        self.models_dir = models_dir
        # self.datasets = ['MIT67', 'Flower102', 'SDog120']
        self.datasets = ['Flower102', 'SDog120']
        # self.archs = ['mbnetv2', 'resnet18', 'vgg16_bn']
        self.archs = ['mbnetv2', 'resnet18']
        # For debug
        # self.datasets = ['MIT67', 'Flower102', 'SDog120']
        # self.archs = ['mbnetv2', 'resnet18']

    def get_dataloader(self, dataset_id, split='train', batch_size=BATCH_SIZE, shuffle=True, seed=SEED, shot=-1):
        """
        Get the torch Dataset object
        :param dataset_id: the name of the dataset, should also be the dir name and the class name
        :param split: train or test
        :param batch_size: batch size
        :param shot: number of training samples per class for the training dataset. -1 indicates using the full dataset
        :return: torch.utils.data.DataLoader instance
        """
        try:
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
                batch_size=batch_size, shuffle=shuffle,
                num_workers=8, pin_memory=False
            )
            return data_loader
        except Exception as e:
            self.logger.warning(f'get_dataloader failed: {e}')
            return None

    def load_pretrained(self, arch_id, fc=True):
        """
        Get the model pretrained on imagenet
        :param arch_id: the name of the arch
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'pretrain({arch_id},ImageNet)',
            arch_id=arch_id,
            dataset_id='ImageNet',
            fc=fc,
        )
        return model_wrapper

    def load_trained(self, arch_id, dataset_id, iters=TRAIN_ITERS, fc=True):
        """
        Get the model with architecture arch_id trained on dataset dataset_id
        :param arch_id: the name of the arch
        :param dataset_id: the name of the dataset
        :param iters: number of iterations
        :return: a ModelWrapper instance
        """
        model_wrapper = ModelWrapper(
            benchmark=self,
            teacher_wrapper=None,
            trans_str=f'train({arch_id},{dataset_id})',
            arch_id=arch_id,
            dataset_id=dataset_id,
            iters=iters,
            fc=fc,
        )
        return model_wrapper

    def list_models(self, fc=True):
        """
        list the models in the benchmark dataset
        :return: a stream of ModelWrapper instances
        """
        source_models = []

        quantization_dtypes = ['qint8', 'float16']
        prune_ratios = [0.2, 0.5, 0.8]
        transfer_tune_ratios = [0.1, 0.5, 1]

        # load pretrained source models
        for arch in self.archs:
            source_model = self.load_pretrained(arch, fc=fc)
            source_models.append(source_model)
            yield source_model

        # retrain models
        retrain_models = []
        for arch_id in self.archs:
            for dataset_id in self.datasets:
                retrain_model = self.load_trained(arch_id, dataset_id, TRAIN_ITERS, fc=fc)
                retrain_models.append(retrain_model)
                yield retrain_model

        # for debug
        # prune_ratios = [0.2]
        # transfer_tune_ratios = [0.5, 1]

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
        
        # - M_{i,x}/{quant-qint8/float16} -- Compress M_{i,x} with integer / float16 quantization
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

        # variations of retrained models
        # - M_{i,x}/{prune-p} -- Prune M_{i,x} with pruning ratio = p
        for retrain_model in retrain_models:
            for pr in prune_ratios:
                yield retrain_model.prune(prune_ratio=pr)

        # - M_{i,x}/{distill} -- Distill M_{i,x}
        for retrain_model in retrain_models:
            yield retrain_model.distill()

        # - M_{i,x}/{steal-j} -- Steal M_{i,x} to A_j
        for retrain_model in retrain_models:
            for arch_id in self.archs:
                yield retrain_model.steal(arch_id=arch_id)


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
    parser.add_argument("-mask", action="store", dest="mask", default="",
                        help="The mask to filter the models to generate, split with +")
    parser.add_argument("-phase", action="store", dest="phase", type=str, default="",
                        help="The phase to run. Use a prefix to filter the phases.")
    parser.add_argument("-regenerate", action="store_true", dest="regenerate", default=False,
                        help="Whether to regenerate the models.")
    args, unknown = parser.parse_known_args()
    return args

def check_param_num(model, name):
    total = sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d) ])
    num = total
    for m in model.modules():
        if ( isinstance(m, nn.Conv2d) ):
            num -= int((m.weight.data == 0).sum())
    ratio = (total - num) / total
    log = f"===>{name}: Total {total}, current {num}, prune ratio {ratio:2f}"
    print(log)

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
    models_to_gen = []
    mask_substrs = args.mask.strip().split('+')
    for model_wrapper in bench.list_models():
        # print(f'loaded model: {model_wrapper}')
        model_str_tokens = model_wrapper.__str__().split('-')
        if len(model_str_tokens) >= 2 and model_str_tokens[-2].startswith(args.phase):
            to_gen = True
            model_str = re.sub(r'[^A-Za-z0-9.]+', '_', model_wrapper.__str__())
            for mask_substr in mask_substrs:
                if not mask_substr:
                    continue
                if mask_substr not in f'_{model_str}_':
                    to_gen = False
                    break
            if to_gen:
                models_to_gen.append(model_wrapper)
    models_to_gen_str = "\n".join([model_wrapper.__str__() for model_wrapper in models_to_gen])
    print(f'{len(models_to_gen)} models to generate: \n{models_to_gen_str}')
    for model_wrapper in models_to_gen:
            
        model_wrapper.gen_model(regenerate=args.regenerate)
    # print(benchmark.model2variations)

