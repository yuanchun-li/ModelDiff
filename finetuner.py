import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
from functools import partial

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class Finetuner(object):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        self.args = args
        self.model = model.to('cuda')
        self.teacher = teacher.to('cuda')
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.init_models()

    def init_models(self):
        args = self.args
        model = self.model
        teacher = self.teacher

        # Used to matching features
        def record_act(self, input, output):
            self.out = output

        if 'mbnetv2' in args.network:
            reg_layers = {0: [model.layer1], 1: [model.layer2], 2: [model.layer3], 3: [model.layer4]}
            model.layer1.register_forward_hook(record_act)
            model.layer2.register_forward_hook(record_act)
            model.layer3.register_forward_hook(record_act)
            model.layer4.register_forward_hook(record_act)
            if '5' in args.feat_layers:
                reg_layers[4] = [model.layer5]
                model.layer5.register_forward_hook(record_act)
        elif 'resnet' in args.network:
            reg_layers = {0: [model.layer1], 1: [model.layer2], 2: [model.layer3], 3: [model.layer4]}
            model.layer1.register_forward_hook(record_act)
            model.layer2.register_forward_hook(record_act)
            model.layer3.register_forward_hook(record_act)
            model.layer4.register_forward_hook(record_act)
        elif 'vgg' in args.network:
            cnt = 0
            reg_layers = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.MaxPool2d) :
                    reg_layers[name] = [module]
                    module.register_forward_hook(record_act)
                    print(name, module)
        
        # Stored pre-trained weights for computing L2SP
        for m in model.modules():
            if hasattr(m, 'weight') and not hasattr(m, 'old_weight'):
                m.old_weight = m.weight.data.clone().detach()
                # all_weights = torch.cat([all_weights.reshape(-1), m.weight.data.abs().reshape(-1)], dim=0)
            if hasattr(m, 'bias') and not hasattr(m, 'old_bias') and m.bias is not None:
                m.old_bias = m.bias.data.clone().detach()

        if args.reinit:
            for m in model.modules():
                if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                    m.reset_parameters()

        if 'vgg' not in args.network:
            reg_layers[0].append(teacher.layer1)
            teacher.layer1.register_forward_hook(record_act)
            reg_layers[1].append(teacher.layer2)
            teacher.layer2.register_forward_hook(record_act)
            reg_layers[2].append(teacher.layer3)
            teacher.layer3.register_forward_hook(record_act)
            reg_layers[3].append(teacher.layer4)
            teacher.layer4.register_forward_hook(record_act)

            if '5' in args.feat_layers:
                reg_layers[4].append(teacher.layer5)
                teacher.layer5.register_forward_hook(record_act)
        else:
            cnt = 0
            for name, module in teacher.named_modules():
                if isinstance(module, nn.MaxPool2d) :
                    reg_layers[name].append(module)
                    module.register_forward_hook(record_act)
                    # print(name, module)

        self.reg_layers = reg_layers
        # Check self.model
        # st()

    def compute_steal_loss(self, batch, label):
        def CXE(predicted, target):
            return -(target * torch.log(predicted)).sum(dim=1).mean()
        model = self.model
        teacher = self.teacher
        alpha = self.args.steal_alpha
        T = self.args.temperature
        
        teacher_out = teacher(batch)
        out = model(batch)
        _, pred = out.max(dim=1)
        
        # _, teacher_pred = teacher_out.max(dim=1)
        # KD_loss = F.cross_entropy(out, teacher_pred)
        # soft_loss, hard_loss = 0, 0
        
        out = F.softmax(out)
        teacher_out = F.softmax(teacher_out)
        KD_loss = CXE(out, teacher_out)
        soft_loss, hard_loss = 0, 0
        
        # soft_loss = nn.KLDivLoss()(
        #     F.log_softmax(out/T, dim=1),
        #     F.softmax(teacher_out/T, dim=1)
        # ) * (alpha * T * T)
        # hard_loss = F.cross_entropy(out, label) * (1. - alpha)
        # KD_loss = soft_loss + hard_loss
        
        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.

        return KD_loss, top1, soft_loss, hard_loss

        
    def compute_loss(self, batch, label, ce, featloss):
        model = self.model
        teacher = self.teacher
        args = self.args
        l2sp_lmda = self.args.l2sp_lmda
        reg_layers = self.reg_layers
        feat_loss, l2sp_loss = 0, 0

        out = model(batch)
        _, pred = out.max(dim=1)

        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.
        # top1_meter.update(float(pred.eq(label).sum().item()) / label.shape[0] * 100.)

        loss = 0.
        loss += ce(out, label)

        ce_loss = loss.item()
        # ce_loss_meter.update(loss.item())

        with torch.no_grad():
            tout = teacher(batch)

        # Compute the feature distillation loss only when needed
        if args.feat_lmda != 0:
            regloss = 0
            for key in reg_layers.keys():
                # key = int(layer)-1

                src_x = reg_layers[key][0].out
                tgt_x = reg_layers[key][1].out
                regloss += featloss(src_x, tgt_x.detach())

            regloss = args.feat_lmda * regloss
            loss += regloss
            feat_loss = regloss.item()
            # feat_loss_meter.update(regloss.item())

        beta_loss, linear_norm = linear_l2(model, args.beta)
        loss = loss + beta_loss 
        linear_loss = beta_loss.item()
        # linear_loss_meter.update(beta_loss.item())

        if l2sp_lmda != 0:
            reg, _ = l2sp(model, l2sp_lmda)
            l2sp_loss = reg.item()
            # l2sp_loss_meter.update(reg.item())
            loss = loss + reg

        total_loss = loss.item()
        # total_loss_meter.update(loss.item())

        return loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss

    def steal_test(self):
        model = self.model
        teacher = self.teacher
        loader = self.test_loader
        alpha = self.args.steal_alpha
        T = self.args.temperature
        
        with torch.no_grad():
            model.eval()
            teacher.eval()
            
            total_soft, total_hard, total_kd = 0, 0, 0
            total = 0
            top1 = 0
            
            for i, (batch, label) in enumerate(loader):
                batch, label = batch.to('cuda'), label.to('cuda')
                total += batch.size(0)
                
                teacher_out = teacher(batch)
                out = model(batch)
                _, pred = out.max(dim=1)
                
                soft_loss = nn.KLDivLoss()(
                    F.log_softmax(out/T, dim=1),
                    F.softmax(teacher_out/T, dim=1)
                ) * (alpha * T * T)
                hard_loss = F.cross_entropy(out, label) * (1. - alpha)
                KD_loss = soft_loss + hard_loss
                
                total_soft += soft_loss.item()
                total_hard += hard_loss.item()
                total_kd += KD_loss.item()
                top1 += int(pred.eq(label).sum().item())
                
        return float(top1)/total*100, total_kd/(i+1), total_soft/(i+1), total_hard/(i+1)
        
    def test(self):
        model = self.model
        teacher = self.teacher
        loader = self.test_loader
        reg_layers = self.reg_layers
        args = self.args
        loss = True

        with torch.no_grad():
            model.eval()

            if loss:
                teacher.eval()

                ce = CrossEntropyLabelSmooth(loader.dataset.num_classes, args.label_smoothing).to('cuda')
                featloss = torch.nn.MSELoss(reduction='none')

            total_ce = 0
            total_feat_reg = np.zeros(len(reg_layers))
            total_l2sp_reg = 0

            total = 0
            top1 = 0
            for i, (batch, label) in enumerate(loader):
                batch, label = batch.to('cuda'), label.to('cuda')

                total += batch.size(0)
                out = model(batch)
                _, pred = out.max(dim=1)
                top1 += int(pred.eq(label).sum().item())

                if loss:
                    total_ce += ce(out, label).item()
                    if teacher is not None:
                        with torch.no_grad():
                            tout = teacher(batch)

                        for i, key in enumerate(reg_layers):
                            # print(key, len(reg_layers[key]))
                            src_x = reg_layers[key][0].out
                            tgt_x = reg_layers[key][1].out
                            # print(src_x.shape, tgt_x.shape)

                            regloss = featloss(src_x, tgt_x.detach()).mean()

                            total_feat_reg[i] += regloss.item()

                    _, unweighted = l2sp(model, 0)
                    total_l2sp_reg += unweighted.item()
                # break

        return float(top1)/total*100, total_ce/(i+1), np.sum(total_feat_reg)/(i+1), total_l2sp_reg/(i+1), total_feat_reg/(i+1)

    def get_fine_tuning_parameters(self):
        model = self.model
        parameters = []

        ft_begin_module = self.args.ft_begin_module
        ft_ratio = self.args.ft_ratio if 'ft_ratio' in self.args else None
        
        if ft_ratio:
            all_params = [param for param in model.parameters()]
            num_tune_params = int(len(all_params) * ft_ratio)
            for v in all_params[-num_tune_params:]:
                parameters.append({'params': v})

            all_names = [name for name, _ in model.named_parameters()]
            with open(osp.join(self.args.output_dir, "finetune.log"), "w") as f:
                f.write(f"Fixed layers:\n")
                for name in all_names[:-num_tune_params]:
                    f.write(name+"\n")
                f.write(f"\n\nFinetuned layers:\n")
                for name in all_names[-num_tune_params:]:
                    f.write(name+"\n")
            
            return parameters

        if not ft_begin_module:
            return model.parameters()

        add_flag = False
        for k, v in model.named_parameters():
            # if ft_begin_module == k:
            if ft_begin_module in k:
                add_flag = True

            if add_flag:
                # print(k)
                parameters.append({'params': v})
        if ft_begin_module and not add_flag:
            raise RuntimeError("wrong ft_begin_module, no module to finetune")

        return parameters
            
    def train(self):
        model = self.model
        train_loader = self.train_loader
        test_loader = self.test_loader
        iterations = self.args.iterations
        lr = self.args.lr
        output_dir = self.args.output_dir
        l2sp_lmda = self.args.l2sp_lmda
        teacher = self.teacher
        reg_layers = self.reg_layers
        args = self.args

        model_params = self.get_fine_tuning_parameters()
        
        if l2sp_lmda == 0:
            optimizer = optim.SGD(
                model_params, 
                lr=lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                model_params, 
                lr=lr, 
                momentum=args.momentum, 
                weight_decay=0,
            )

        end_iter = iterations

        if args.const_lr:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                end_iter,
            )

        teacher.eval()
        ce = CrossEntropyLabelSmooth(train_loader.dataset.num_classes, args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        batch_time = MovingAverageMeter('Time', ':6.3f')
        data_time = MovingAverageMeter('Data', ':6.3f')
        ce_loss_meter = MovingAverageMeter('CE Loss', ':6.3f')
        feat_loss_meter  = MovingAverageMeter('Feat. Loss', ':6.3f')
        l2sp_loss_meter  = MovingAverageMeter('L2SP Loss', ':6.3f')
        linear_loss_meter  = MovingAverageMeter('LinearL2 Loss', ':6.3f')
        total_loss_meter  = MovingAverageMeter('Total Loss', ':6.3f')
        top1_meter  = MovingAverageMeter('Acc@1', ':6.2f')

        train_path = osp.join(output_dir, "train.tsv")
        with open(train_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
            wf.write('\t'.join(columns) + '\n')
        test_path = osp.join(output_dir, "test.tsv")
        with open(test_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
            wf.write('\t'.join(columns) + '\n')
        adv_path = osp.join(output_dir, "adv.tsv")
        with open(adv_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'AdvAcc', 'ASR']
            wf.write('\t'.join(columns) + '\n')
        
        dataloader_iterator = iter(train_loader)
        for i in range(iterations):

            model.train()
            optimizer.zero_grad()

            end = time.time()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            data_time.update(time.time() - end)

            if args.steal:
                loss, top1, soft_loss, hard_loss = self.compute_steal_loss(batch, label)
                total_loss = loss
                ce_loss = hard_loss
                feat_loss = soft_loss
                linear_loss, l2sp_loss = 0, 0
            else:
                loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                    batch, label, 
                    ce, featloss,
                )
            top1_meter.update(top1)
            ce_loss_meter.update(ce_loss)
            feat_loss_meter.update(feat_loss)
            linear_loss_meter.update(linear_loss)
            l2sp_loss_meter.update(l2sp_loss)
            total_loss_meter.update(total_loss)
            
            loss.backward()
            #-----------------------------------------
            for k, m in enumerate(model.modules()):
                # print(k, m)
                if isinstance(m, nn.Conv2d):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    m.weight.grad.data.mul_(mask)
                if isinstance(m, nn.Linear):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    m.weight.grad.data.mul_(mask)
            #-----------------------------------------
            optimizer.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            if scheduler is not None:
                scheduler.step()

            batch_time.update(time.time() - end)

            if (i % args.print_freq == 0) or (i == iterations-1):
                progress = ProgressMeter(
                    iterations,
                    [batch_time, data_time, top1_meter, total_loss_meter, ce_loss_meter, feat_loss_meter, l2sp_loss_meter, linear_loss_meter],
                    prefix="LR: {:6.3f}".format(current_lr),
                    output_dir=output_dir,
                )
                progress.display(i)

            if (i % args.test_interval == 0) or (i == iterations-1):
                if self.args.steal:
                    test_top1, test_ce_loss, test_feat_loss, test_weight_loss = self.steal_test(
                        # model, teacher, test_loader, loss=True
                    )
                    train_top1, train_ce_loss, train_feat_loss, train_weight_loss = self.steal_test(
                        # model, teacher, train_loader, loss=True
                    )
                    test_feat_layer_loss, train_feat_layer_loss = 0, 0
                else:
                    test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test(
                        # model, teacher, test_loader, loss=True
                    )
                    train_top1, train_ce_loss, train_feat_loss, train_weight_loss, train_feat_layer_loss = self.test(
                        # model, teacher, train_loader, loss=True
                    )
                
                print(
                    'Eval Train | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, train_top1, train_ce_loss, train_feat_loss, train_weight_loss))
                print(
                    'Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, test_top1, test_ce_loss, test_feat_loss, test_weight_loss))
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(train_path, 'a') as af:
                    train_cols = [
                        localtime,
                        i, 
                        round(train_top1,2), 
                        round(train_ce_loss,2), 
                        round(train_feat_loss,2),
                        round(train_weight_loss,2),
                    ]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                with open(test_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(test_top1,2), 
                        round(test_ce_loss,2), 
                        round(test_feat_loss,2),
                        round(test_weight_loss,2),
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')
                if not args.no_save:
                    ckpt_path = osp.join(
                        args.output_dir,
                        "ckpt.pth"
                    )
                    torch.save(
                        {'state_dict': model.state_dict()}, 
                        ckpt_path,
                    )

            if ( hasattr(self, "iterative_prune") and i % args.prune_interval == 0 ):
                self.iterative_prune(i)

        return model.to('cpu')

    def countWeightInfo(self):
        ...
