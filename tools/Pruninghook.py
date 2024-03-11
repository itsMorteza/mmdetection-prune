import os.path as osp
import re
import sys
import scipy
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
import datasetdep
import pickle
import numpy as np
import json
import config_loader
import copy 
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from math import ceil

from mmcv.runner import HOOKS
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import Hook
from torch.nn import Conv2d, Linear
from torch.nn.modules.batchnorm import _BatchNorm
from types import MethodType
from mmcv.cnn import get_model_complexity_info
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.apis import single_gpu_test
from config_loader import load_config2
from mmdet.datasets.coco import CocoDataset
from compression.model_compression import compress_model
#from quantization_utils.quant_modules import *
# These grad_fn pattern are flags of specific a nn.Module
CONV = ('ThnnConv2DBackward', 'CudnnConvolutionBackward')
FC = ('ThAddmmBackward', 'AddmmBackward', 'MmBackward')
BN = ('ThnnBatchNormBackward', 'CudnnBatchNormBackward')
# the modules which contains NON_PASS grad_fn need to change the parameter size
# according to channels after pruning
NON_PASS = CONV + FC

@HOOKS.register_module()
class BrushPruningHook(Hook):
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops", prune the model by
            "acts" or flops. Default: "acts"
        batch_size (int): The batch_size when pruning model.
            Default: 2
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """
    def __init__(
        self,
        pruning=True,
        delta='acts',
        batch_size=2,
        test_load=None,
        interval=10,
        pr_type='l2',
        deploy_from=None,
        masking=False,
        arch_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):
        assert delta in ('acts', 'flops')
        self.pruning = pruning
        self.delta = delta
        self.interval = interval
        self.batch_size = batch_size
        # The key of self.input is conv module, and value of it
        # is list of conv' input_features in forward process
        self.conv_inputs = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}
        self.test_load = datasetdep.test_load0

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        self.pr_type=pr_type
        self.prunperc = 100.
        self.prunrate = 1.
        self.themask = {}
        self.pr_mask={} 
        self.keep_mask_li={}
        self.cookbook = {}
        self.cookfil = {}
        self.nonpass_inputs = {}
        self.maskdims = 0
        self.avg_loss = 0
        self.channel_times=0
        self.filter_times=0
        self.masking = masking
        self.avg_mask_loss = 0  
        self.pritt=0
        self.ctrloss=0
        self.arch_from=arch_from 
        self.deploy_from=deploy_from  
        self.conv_names = OrderedDict()
        self.bn_names = OrderedDict()
        self.fc_names = OrderedDict()
        self.nonpass_names = OrderedDict()
        self.maxflops=0.
        self.alpha=1  
    def after_build_model(self, model):
        """Remove all pruned channels in finetune stage.
        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """
        with open(self.arch_from,'r') as json_file:
            self.prunedarch = json.load(json_file)
        pr_block=0
        
        for name, module in model.named_modules():

            if (type(module).__name__ == 'Conv2d'):
                if(pr_block>0):
                        tp.prune_related_conv(module, [i for i in range(module.weight.data.shape[1]-pr_block)])
                if (module.weight.shape[0]!=self.prunedarch.get(f"module.{name}", None)):
                    keep_num=self.prunedarch.get(f"module.{name}", None)
                    self.pr_mask[name]=[i for i in range(module.weight.data.shape[0]-keep_num)]
                    self.keep_mask_li[name]=[i for i in range(keep_num)]  
                    tp.prune_conv(module,self.pr_mask[name])
                    bnname='Nothingmatter'
                    #print(name,len(self.pr_mask[name]))
                    if(24>len(name)>20):
                        bnname=name[:18]+'bn'+name[22]
                    for name2, module2 in model.named_modules():
                        if (type(module2).__name__ != 'Conv2d'):
                            if(bnname in name2):
                                tp.prune_batchnorm(module2,self.pr_mask[name])
                    pr_block=keep_num
                else:
                    pr_block=0 
                if 'rpn_head.rpn_reg' in name:
                    tp.prune_related_conv(module, [i for i in range(module.weight.data.shape[1]-64)])
            if (type(module).__name__ == 'Linear'):
                if (module.weight.shape[0]!=self.prunedarch.get(f"module.{name}", None)):
                    keep_num=self.prunedarch.get(f"module.{name}", None)
                    self.pr_mask[name]=[i for i in range(module.weight.data.shape[0]-keep_num)]
                    self.keep_mask_li[name]=[i for i in range(keep_num)]  
                    tp.prune_linear(module,self.pr_mask[name]) 
                    print(name,len(self.pr_mask[name]))
                    if(pr_block>0):
                        tp.prune_related_linear(module, [i for i in range(module.weight.data.shape[1]-pr_block)])
                    pr_block=keep_num
                else:
                    pr_block=0 
                #if 'roi_head.bbox_head.fc_cls' in name:
                #    tp.prune_related_linear(module, [i for i in range(module.weight.data.shape[1]-1024)])     
                #if 'roi_head.bbox_head.fc_reg' in name:
                #    tp.prune_related_linear(module, [i for i in range(module.weight.data.shape[1]-1024)])                  
        self.pruning=False 
        #load_checkpoint(model, self.deploy_from) 
    def before_run(self, runner):
        self.logger = runner.logger
                
        model = runner.model
        
        print(getattr(model.module.backbone , 'layer1')[0].downsample)
        #buld dependecy graph
        self.quantize_config =  load_config2('/comm_dat/morteza/mmdetection/tools/compression_maskrcnn.yaml')["model"]["compression_parameters"]
        self.build_setting(model,mode=2)
        self.MAXP = count_params(model,True)
        
        if self.deploy_from:
            print("deploying")
            #load pruned arch if applicable
            #load masked arch if applicable
            if self.masking: 
                set_masks(runner,FT=masking)
            else:
                with open(self.arch_from,'r') as json_file:
                   self.prunedarch = json.load(json_file)
                for m in runner.model.modules():              
                  if (type(m).__name__ == 'Conv2d')or(type(m).__name__ == 'Linear'):  
                     if (m.weight.data.shape[0]!=self.prunedarch[m.name]):
                        self.pr_mask[m.name]=[i for i in range(m.weight.data.shape[0]-self.prunedarch[m.name])]
                        self.keep_mask_li[m.name]=[i for i in range(self.prunedarch[m.name])]
                self.set_del(model)   
                self.pruning=False 
                load_checkpoint(model, self.deploy_from)         
        
        if self.pruning:
            self.prevloss = 0.0
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            model.eval()
            #recognizing the convolution  
                 
            #counting the number of parameters
            #self.checkrun = runner.model
            
            model.train()

            self.prev_state_dict = runner.model.state_dict()
            #initilize pruning parameters
            self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_os(model,0)

           
    def after_train_iter(self, runner):
        if not self.pruning:
            if (runner.iter + 1) % (20* self.interval) == 0 :
               p_rate = prune_rate(runner.model,False)
               CRT = 100*(1-(count_params(runner.model,True)/self.MAXP))
               _,self.prflops = self.compute_flops()
               if self.maxflops==0:
                  self.maxflops=  self.prflops/CRT
               print('%.2f > zero count: %s : Real pruned %.2f params -- %.2f flops ' % (self.prunperc, p_rate,CRT,100*(1-(float(self.prflops)/float(self.maxflops))))) 
            self.init_flops_acts()
            return
        model = runner.model
        self.avg_loss+=runner.outputs['loss'].item()
        self.avg_mask_loss += runner.outputs['log_vars']['loss_mask']
        self.ctrloss+=1
        if (runner.iter + 1) % (5* self.interval) == 0 :
                difloss=(self.avg_mask_loss/self.ctrloss)-self.prevloss
                if self.ctrloss == (runner.iter + 1):
                   _,self.maxflops = self.compute_flops()
                   print(self.maxflops)
                   self.prevloss=(self.avg_mask_loss/self.ctrloss)
                   self.prev_state_dict = runner.model.state_dict()
                elif difloss<=0.1:
                   self.prev_state_dict = runner.model.state_dict()
                """elif difloss>0.1:
                   runner.model.load_state_dict(self.prev_state_dict) """

                print("loss_mask : {}".format(float(self.avg_mask_loss/(self.ctrloss))))   
                self.prevloss=float(self.avg_mask_loss/self.ctrloss)
                self.ctrloss=0
                self.avg_mask_loss=0   
                #for j in range(10):

                #important
                #pruning selection
                """"""
                #self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1(model,0)
                #self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_os(model,0)
                #self.prunperc = 100 
                if self.pr_type == 'l2':
                        masks = quick_filter_prune_l2(model, self.prunperc)
                elif self.pr_type == 'block':
                        #getattr(model.module.backbone , 'layer1')[2]=nn.Identity()
                        #getattr(model.module.backbone , 'layer1')[1]=nn.Identity()
                        #getattr(model.module.backbone , 'layer2')[2]=nn.Identity()
                        getattr(model.module.backbone , 'layer3')[4]=nn.Identity()
                        #getattr(model.module.backbone , 'layer3')[5]=nn.Identity()
                        getattr(model.module.backbone , 'layer4')[2]=nn.Identity()
                        #getattr(model.module.backbone , 'layer4')[0]= getattr(model.module.backbone , 'layer4')[0].downsample
                        print(getattr(model.module.backbone , 'layer4')[0])
                        #self.themask = quick_block_prune_mask(model, self.prunperc)
                        #set_masks(runner.model,self.themask,False)
                        #self.set_B_del(runner.model) 
                elif self.pr_type == 'sfp':
                        masks = quick_sfp_filter_prune(model, self.prunperc)
                elif self.pr_type == 'onesh':
                        self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_os(model,0)
                        self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_vst( model, self.prunperc, prev_msk=self.keep_mask_li, pt_mask=self.themask, prev_indi=self.pr_mask)
                elif self.pr_type == 'l1':
                        if(self.channel_times%10==0):
                            self.prunperc=10
                            self.themask = quick_channel_prune_l1_v1( model, self.prunperc)
                            #set_masks(runner.model,self.themask,False) 
                        else:
                            #self.prunperc=100/32
                            self.prunperc=100
                            self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_os(model,0)
                            self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_vst( model, self.prunperc, prev_msk=self.keep_mask_li, pt_mask=self.themask, prev_indi=self.pr_mask)
                            self.set_del(runner.model)
                            #model = self.quantize_model(model)
                            #tmodel = compress_model(model, **self.quantize_config).cuda()
                            #torch.cuda.empty_cache()
                            #del runner.model
                            #runner.model=tmodel
                            #donothing=0
                            #for param in model.parameters():
                            #   try:
                            #     param.requires_grad = True
                            #   except:
                            #     donothing+=1
                            #print(donothing)
                            #torch.set_grad_enabled(True)
                        
                        donothing=0

                        #self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_vst( model, self.prunperc, prev_msk=self.keep_mask_li, pt_mask=self.themask, prev_indi=self.pr_mask)
                        #self.themask = quick_channel_prune_l1_v1( model, self.prunperc)
                        #self.themask, self.pr_mask, self.keep_mask_li = quick_filter_prune_l1_os( model, self.prunperc, prev_msk=self.keep_mask_li, pt_mask=self.themask, prev_indi=self.pr_mask)
                        self.channel_times+=1
                        

                elif self.pr_type == 'lth':
                   while self.prunperc < 1:
                         masks,self.maskdims = quick_prune_lth(model, 1-self.prunperc)
                         self.themask=masks
                         Check_point2 = runner.model
                         apply_mask(runner.model,self.themask)
                         p_rate = prune_rate(runner.model,False)
                         print('%.2f > %s pruned: %s' % (self.prunperc,self.pr_type, p_rate))
                         #self.Eval_func(runner)
                         self.Model_saver(runner,p_rate)
                         runner.model = Check_point2
                
 
 
                #self.set_del(runner.model)
                p_rate = prune_rate(runner.model,False,self.pr_mask,True)
                Cu_Par = count_params(model)
                CPR = (1-float(Cu_Par/ self.MAXP))*100
                if CPR>60 or self.prunperc==100 or self.pr_type == 'block' :
                   self.Eval_func(runner)
                   self.Model_saver(runner,CPR)
                   self.pruning=False  
                #printing function  
                print('%d / %d pruned: %.2f' % (Cu_Par,self.MAXP, CPR)) 
                #print('%.2f > %s pruned: %s' % (self.prunperc,self.pr_type, p_rate)) 
                #self.prunperc = self.prunrate + self.prunperc   
                self.avg_loss = 0
                self.prunperc+=self.prunrate

        self.init_flops_acts()
                      
    def after_epoch(self, runner):
        model = runner.model
        p_rate = prune_rate(model,False)
        Cu_Par = count_params(model)
        CPR = (1-float(Cu_Par/ self.MAXP))*100
        path = osp.join(runner.work_dir, 
                        'ncountzero_{:.0f}_epoch{}.pth'.
                        format(p_rate,runner.epoch))
        save_checkpoint(runner.model, filename=path)
        pass
    def Model_saver(self, runner,p_rate):
         path = osp.join(runner.work_dir, 
                         'ncountzero_{:.0f}_filterPercent{:.0f}_epoch{}.pth'.
                         format(p_rate,self.prunrate,runner.epoch))
         save_checkpoint(runner.model,  filename=path)  
         Arch ={}
         for m in runner.model.modules():              
            if (type(m).__name__ == 'Conv2d')or(type(m).__name__ == 'Linear'):  
               Arch[m.name]=m.weight.data.shape[0]
               print(m.name,Arch[m.name])
         with open('pruned_Arch.txt','w') as fp:
              fp.write(json.dumps(Arch))      
               
    def Eval_func(self, runner):
        val_dataset = build_dataset(self.test_load, dict(test_mode=True))
        val_dataloader = build_dataloader(
                                val_dataset,
                                samples_per_gpu=1,
                                workers_per_gpu=2,
                                shuffle=False)
        results = single_gpu_test(runner.model, val_dataloader, show=False)
        key_score = val_dataloader.dataset.evaluate(results,metric='segm')
     
    def register_hooks(self):
        """Register forward and backward hook to Conv module."""
        for module, name in self.nonpass_names.items():
            module.register_forward_hook(self.save_input_forward_hook)
            #module.register_backward_hook(self.compute_out_backward_hook)
    def compute_out_backward_hook(self, module, grad_input, *args):
        if module in self.nonpass_names:
           layer_name = type(module).__name__
           grad_feature = grad_input[0] if layer_name == "Conv2d" else grad_input[1]
           if grad_feature is not None:
               feature = self.nonpass_inputs[module].pop(-1)[0]
               grads = feature * grad_feature
               while grads.dim() > 2:
                   grads = grads.sum(dim=-1)
               if grads.size(0) % self.batch_size == 0:
                   grads = grads.view(self.batch_size, -1, grads.size(1)).sum(dim=1)
               #self.temp_fisher_info[module][:grad_feature.size(0)] += grads

    def save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts.
        Args:
            module (nn.Module): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """
        if type(module).__name__ == 'Conv2d':
            n, oc, oh, ow = outputs.shape
            ic = module.in_channels // module.groups
            kh, kw = module.kernel_size
            self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
        elif type(module).__name__ == 'Linear':
            n, oc = outputs.shape
            self.flops[module] += np.prod([n, oc])    

    
    def build_setting(self,model, mode = 0):
     for n, m in model.named_modules():
            if isinstance(m, Conv2d):
                m.name = n
                self.conv_names[m] = n
                self.nonpass_names[m] = n
                self.flops[m] = 0
            if isinstance(m, _BatchNorm):
                m.name = n
                self.bn_names[m] = n
                self.flops[m] = 0
            if isinstance(m, Linear):
                m.name = n
                self.fc_names[m] = n
                self.nonpass_names[m] = n
                self.flops[m] = 0
     if mode == 1:
        for name in datasetdep.data:
                if name is 'train':
                   datasetstr = build_dataset(datasetdep.data[name])
        train_dataloader = build_dataloader(
                                datasetstr,
                                samples_per_gpu=1,
                                workers_per_gpu=1,
                                shuffle=True) 
        for item in train_dataloader:
            self.trainsample= item
            break
        del datasetstr
        del train_dataloader
     if mode == 3:
        for n, m in model.named_modules():
             add_pruning_attrs(m, pruning=self.pruning) 
        self.construct_outchannel_masks()  
        self.register_hooks()     
     if mode == 2:
        #for n, m in runner.model.named_modules():
        #    add_pruning_attrs(m, pruning=self.pruning)        
        for module, name in self.nonpass_names.items():
             self.nonpass_inputs[module] = []                 
        inputs = torch.zeros(1, 3, 256, 256).cuda()
        inputs_meta = [{"img_shape": (256, 256, 3), "scale_factor": np.zeros(4, dtype=np.float32)}]
        backbone_out=model.module.backbone(inputs)
        neck_out = model.module.neck(backbone_out)
        from mmdet.core import bbox2roi
        rpn_out = model.module.rpn_head(neck_out)
        proposals = model.module.rpn_head.get_bboxes(*rpn_out, inputs_meta)
        rois = bbox2roi(proposals)
        roi_out = model.module.roi_head._bbox_forward(neck_out, rois)
        mask_results = model.module.roi_head._mask_forward(neck_out, rois)
        loss = sum([sum([level.sum() for level in levels]) for levels in backbone_out])
        loss += sum([sum([level.sum() for level in levels]) for levels in neck_out])
        loss += sum([sum([level.sum() for level in levels]) for levels in rpn_out])
        loss += roi_out['cls_score'].sum()+ roi_out['bbox_pred'].sum() 
        loss += mask_results['mask_pred'].sum()
        self.conv2ancest = self.find_module_ancestors(loss, CONV)
        #print("conv done")
        #self.nonpass2ancest = self.find_module_ancestors(loss, NON_PASS) 
        self.conv_link = {k.name: [item.name for item in v] for k, v in self.conv2ancest.items()}
        self.bn2ancest = self.find_module_ancestors(loss, BN)
        print("bn done")
        self.fc2ancest = self.find_module_ancestors(loss, FC)
        print("fc done")
        self.fc_link = {k.name: [item.name for item in v] for k, v in self.fc2ancest.items()}
        self.nonpass2ancest = {**self.conv2ancest, **self.fc2ancest}
        self.nonpass_link = {**self.conv_link, **self.fc_link}  
        loss.sum().backward()
        self.register_hooks()
        #self.construct_outchannel_masks()  
        #self.register_hooks() 
    def construct_outchannel_masks(self):
        """Register the `input_mask` of one conv to it's nearest ancestor conv,
        and name it as `out_mask`, which means the actually number of output
        feature map after pruning."""

        for conv, name in self.conv_names.items():
            assigned = False
            for m, ancest in self.conv2ancest.items():
                if conv in ancest:
                    conv.out_mask = m.in_mask
                    assigned = True
                    break
            # may be the last conv of network
            if not assigned:
                conv.register_buffer(
                    'out_mask',
                    torch.ones((1, conv.out_channels, 1, 1),
                               dtype=torch.bool).cuda())

        for bn, name in self.bn_names.items():
            conv_module = self.bn2ancest[bn][0]
            bn.out_mask = conv_module.out_mask 
    def find_module_ancestors(self, loss, pattern):
        """find the nearest Convolution of the module
        matching the pattern
        Args:
            loss(Tensor): the output of the network
            pattern(Tuple[str]): the pattern name
        Returns:
            dict: the key is the module match the pattern(Conv or Fc),
             and value is the list of it's nearest ancestor Convolution
        """

        # key is the op (indicate a Conv or Fc) and value is a list
        # contains all the nearest ops (indicate a Conv or Fc)
        op2parents = {}
        self.traverse(loss.grad_fn, op2parents, pattern)

        var2module = {}
        if pattern is BN:
            module_names = self.bn_names
        elif pattern is CONV:
            module_names = self.conv_names
        elif pattern is NON_PASS:
            module_names = {**self.conv_names, **self.fc_names}
        else:
            module_names = self.fc_names

        if pattern is FC:
            for module, name in module_names.items():
                #var2module[id(module.bias)] = module
                var2module[id(module.weight)] = module
        else:
            for module, name in module_names.items():
              #if 'fc' in name: var2module[id(module.bias)] = module
              var2module[id(module.weight)] = module

        # same module may appear several times in computing graph,
        # so same module can correspond to several op, for example,
        # different feature pyramid level share heads.
        # op2module select one op as the flag of module.
        op2module = {}
        for op, parents in op2parents.items():
            # TODO bfs to get variable
            try: var_id = id(op.next_functions[1][0].variable)
            except: 
                 tbackward_op = filter(lambda x: x[0].name().startswith("TBackward"), op.next_functions)
                 param_op = next(tbackward_op)[0].next_functions[0][0]
                 var_id = id(param_op.variable)
                 #print(var_id)
            #if pattern is FC:
            #    var_id = id(op.next_functions[0][0].variable)
            #else:
            #    var_id = id(op.next_functions[1][0].variable)
            module = var2module[var_id]
            exist = False
            # may several op link to same module
            for temp_op, temp_module in op2module.items():
                # temp_op(has visited in loop) and op
                # link to same module, so their should share
                # all parents, so we need extend the value of
                # op to value of temp_op
                if temp_module is module:
                    op2parents[temp_op].extend(op2parents[op])
                    exist = True
                    break
            if not exist:
                op2module[op] = module

        if not hasattr(self, 'nonpass_module'):
            # save for find bn's ancestor convolutions
            self.nonpass_module = op2module
        else:
            self.nonpass_module.update(op2module)
        return {
            module: [
                self.nonpass_module[parent] for parent in op2parents[op]
                if parent in self.nonpass_module
            ]
            for op, module in op2module.items()
        }
    def traverse(self, op, op2parents, pattern=NON_PASS):
        """to get a dict which can describe the computer Graph,
        Args:
            op (grad_fn): as a root of DFS
            op2parents (dict): key is the grad_fn match the patter,and
                value is first grad_fn match NON_PASS when DFS from Key
            pattern (Tuple[str]): the patter of grad_fn to match
        """

        if op is not None:
            parents = op.next_functions
            if parents is not None:
                if self.match(op, pattern):
                  if pattern is FC:
                    op2parents[op] = self.dfs(parents[1][0], [])
                  else:
                    op2parents[op] = self.dfs(parents[0][0], [])
                if len(op2parents.keys()) == -1:
                    return    
                for parent in parents:
                    parent = parent[0]
                    if parent not in op2parents:
                        self.traverse(parent, op2parents, pattern)
    def dfs(self, op, visited):
        """DFS from a op,return all op when find a op match the patter
        NON_PASS.
        Args:
            op (grad_fn): the root of DFS
            visited (list[grad_fn]): contains all op has been visited
        Returns:
            list : all the ops  match the patter NON_PASS
        """

        ret = []
        if op is not None:
            visited.append(op)
            if self.match(op, NON_PASS):
                return [op]
            parents = op.next_functions
            if parents is not None:
                for parent in parents:
                    parent = parent[0]
                    if parent not in visited:
                        ret.extend(self.dfs(parent, visited))
        return ret
    def match(self, op, op_to_match):
        """Match an operation to a group of operations; In pytorch graph, there
        may be an additional '0' or '1' (e.g. Addbackward1) after the ops
        listed above.
        Args:
            op (grad_fn): the grad_fn to match the pattern
            op_to_match (list[str]): the pattern need to match
        Returns:
            bool: return True when match the pattern else False
        """

        for to_match in op_to_match:
            if re.match(to_match + '[0-1]?$', type(op).__name__):
                return True
        return False    
    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.nonpass_names.items():
            self.flops[module] = 0
            self.acts[module] = 0
    def compute_flops(self):
        """Computing the flops remains."""
        flops = 0
        max_flops = 0

        for module, name in self.nonpass_names.items():
            max_flop = self.flops[module]
            max_flops += max_flop
            try:
                maskshape=module.mask.data.size()
                chnl_prams = np.prod(maskshape[1:])
                F_count=int(torch.sum(module.mask.data==0).item()/chnl_prams)
                total_fil=module.mask.data.size()[0]
                remain_fil = total_fil - F_count
                flops += max_flop * remain_fil / total_fil
            except:
                flops += max_flop
        print(max_flops)
        return float(flops) / float(max_flops), max_flops
                                       

    def set_del(self, model):
        for key in self.pr_mask:
          if self.pr_mask[key]:
             if (len(self.pr_mask[key])%self.alpha!=0) and ('fc' not in key):
               self.pr_mask[key] = quick_filter_prune_l1_v2(model, key, 
                                                             self.pr_mask[key],self.alpha) 
             print((key,len(self.pr_mask[key])))
        for m in model.modules():              
            if type(m).__name__ == 'Conv2d':  
               for key in self.pr_mask:
                    if key in m.name:   
                         if self.pr_mask[key]:
                             #if m.mask is not None:
                             #     del m.mask
                             tp.prune_conv(m,self.pr_mask[key])
                             if 'conv3' in m.name and 'backbone' in m.name:
                                 ds=m.name[:25]+'downsample.0'
                                 for m2 in model.modules():
                                   if type(m2).__name__ == 'Conv2d':    
                                     if ds in m2.name:
                                         tp.prune_conv(m2,self.pr_mask[key])
                                         for key2 in self.bn2ancest:
                                              for j in self.bn2ancest[key2]:
                                                 if j is m2:
                                                    tp.prune_batchnorm(key2, self.pr_mask[key] )
                             for key2 in self.bn2ancest:
                                 for j in self.bn2ancest[key2]:
                                     if j is m:
                                         tp.prune_batchnorm(key2, self.pr_mask[key] )
                             for key2 in self.conv2ancest:
                                 for j in self.conv2ancest[key2]:
                                     if (j is m) and (key2.weight.data.shape[1]>len(self.keep_mask_li[key])): 
                                         tp.prune_related_conv(key2, self.pr_mask[key])
                                         
            if type(m).__name__ == 'Linear':  
               prune_linear=0
               for key in self.pr_mask:
                    if key in m.name:   
                         if self.pr_mask[key]: 
                           if m.mask is not None:
                               del m.mask    
                           tp.prune_linear(m, self.pr_mask[key])
                           for key2 in self.fc2ancest:
                              for j in self.fc2ancest[key2]:
                                  if j is m:
                                     tp.prune_related_linear(key2,self.pr_mask[key])
    def set_B_del(self, model):
        for m in model.modules():              
            if isinstance(m, Conv2d):
                    #for key in self.pr_mask:
                    if 'module.backbone.layer1.0' in m.name:
                         print(m.name)   
                         m=nn.Identity()
                         #del m
        for m in model.modules():              
            if isinstance(m, _BatchNorm):
                    #for key in self.pr_mask:
                    if 'module.backbone.layer1.0' in m.name:   
                         print(m.name) 
                         #del m
                         m=nn.Identity()    
                       
def count_params(model,detail=False,conv_only=False):
    n_params = 0
    bbon_n_params = 0
    conv2_bb_n_params = 0
    neck_n_params = 0
    rpn_n_params = 0
    roi_BB_n_params = 0
    roi_mask_n_params = 0    
    n_zeros = 0
    nn_params = 0
    cn_params = 0
    nconv=0
    nclayers=0
    nnlayers=0
    for name, param in model.named_parameters():
        if "backbone" in name: 
           bbon_n_params+= param.numel()
           if "conv2" in name:
              conv2_bb_n_params += param.numel()
        elif "neck" in name:
           neck_n_params+= param.numel()
        elif "rpn" in name:
           rpn_n_params += param.numel()
        elif "roi" in name:
           if "bbox" in name:
              roi_BB_n_params+=param.numel()
           if "mask" in name:
              roi_mask_n_params+= param.numel()
    for name, param in model.named_parameters():
        #print(name, param.size())
        n_params += param.numel()
    for m in model.modules():
        if type(m).__name__ == 'Conv2d':
                cn_params += m.weight.data.numel()
                nconv+=m.weight.size()[0]
                nclayers +=1
                n_zeros += torch.sum(m.weight.data==0).item()
        if type(m).__name__ == 'Linear':
                nn_params += m.weight.data.numel()
                nnlayers +=1
                n_zeros += torch.sum(m.weight.data==0).item()
    if detail:
        print('Conv percentage: {}'.format(cn_params/n_params))
        print('Line percentage: {}'.format(nn_params/n_params))
        print('Parameter Number: {}'.format(n_params))
        print('backbone Parameter percentage: {} ,{}'.format(bbon_n_params/n_params, conv2_bb_n_params/n_params))
        print('neck Parameter percentage: {}'.format(neck_n_params/n_params))
        print('rpn Parameter percentage: {}'.format(rpn_n_params/n_params))
        print('roi Parameter percentage: {}'.format(roi_BB_n_params/n_params))
        print('roi Parameter percentage: {}'.format(roi_mask_n_params/n_params))    
        print('nconv : {}'.format(nconv))
        print('nnlayers : {}'.format(nnlayers))
        print('nclayers : {}'.format(nclayers))
        print('Zero percentage: {}'.format(n_zeros/n_params))
    print('Parameter Number: {}'.format(n_params))
    return n_params
def modified_forward(self, feature):                 
                        p = torch.ones(1).cuda()
                        p.requires_grad = False
                        p=p.cuda()
                        try: 
                           if self.mapflag == p:           
                                self.weight.data = self.weight.data*self.mask.data
                        except: 
                           del p
                        return F.conv2d(feature, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


def set_masks(self, masks,FT=False):
        count = 0
        F_count = 0
        for m in self.modules():
            try:
                 if type(m).__name__ == 'Conv2d':
                         m.register_buffer('mask', masks[count]) 
                         m.mask.requires_grad = False
                         m.mask=m.mask.cuda()
                         #print('mask shape: {}'.format(m.mask.data.size()))
                         #chnl_prams = np.prod(maskshape[1:])
                         #print('weight shape {}'.format(m.weight.data.size()))
                         count += 1
                         m.weight=nn.Parameter(m.weight*m.mask)
                         #if(torch.sum(m.mask.data==0).item()):
                         #   print((m.name,torch.sum(m.mask.data==0).item()/chnl_prams))
                         #m.orig = m.weight
                         #del m.weight
                         #m.register_forward_pre_hook(repop_weight)
                         """m.weight = nn.Parameter(m.weight*m.mask)
                         #m.weight.data = m.weight.data*m.mask.data
                         if(torch.sum(m.mask.data==0).item()):
                            print((m.name,torch.sum(m.mask.data==0).item()/chnl_prams))
                            #print((m.weight.data.numel(),m.weight.data.nonzero().size(0)))
                            F_count+=int(torch.sum(m.mask.data==0).item()/chnl_prams)
                         m.register_buffer('mapflag', torch.ones(1).cuda())

                         m.mapflag.requires_grad = False
                         m.mapflag=m.mapflag.cuda()
                         if type(m).__name__ == 'Linear':                 
                         #m.forward= MethodType(modified_forward, m)"""
            except:
                pass             


    
def prune_rate(model, verbose=True, pr_mask=None,verbose_filter=False):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    if verbose_filter:            
                prund_fil_num=0
                bb_fil_num=0
                neck_fil_num = 0
                rpn_fil_num= 0
                mask_fil_num = 0
                for m in model.modules():
                    if type(m).__name__ == 'Conv2d':  
                        for key in pr_mask:
                                if key is m.name:
                                   if pr_mask[key]:
                                      prund_fil_num+=len(pr_mask[key])
                                      if "backbone" in key: bb_fil_num+=len(pr_mask[key])
                                      elif "neck" in key:neck_fil_num+=len(pr_mask[key])
                                      elif "rpn" in key:rpn_fil_num+=len(pr_mask[key])
                                      elif "roi" in key:mask_fil_num+=len(pr_mask[key])
                print("prund_filter_num : {} : BB:{} - Neck:{} - RPn: {} - Roi: {} ".format(prund_fil_num,bb_fil_num,neck_fil_num,rpn_fil_num,mask_fil_num))  
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    
def add_pruning_attrs(module, pruning=False):
    """When module is conv, add `finetune` attribute, register `mask` buffer
    and change the origin `forward` function. When module is BN, add `out_mask`
    attribute to module.
    Args:
        conv (nn.Conv2d):  The instance of `torch.nn.Conv2d`
        pruning (bool): Indicating the state of model which
            will make conv's forward behave differently.
    """
    # TODO: mask  change to bool
    if type(module).__name__ == 'Conv2d':
        module.register_buffer(
            'in_mask', module.weight.new_ones((1, module.in_channels, 1, 1), ))
        module.register_buffer(
            'out_mask', module.weight.new_ones(
                (1, module.out_channels, 1, 1), ))
        module.finetune = not pruning
        def modified_forward_c(self, feature):
            if not self.finetune:
                feature = feature * self.in_mask
            return F.conv2d(feature, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        module.forward = MethodType(modified_forward_c, module)
    if 'BatchNorm' in type(module).__name__:
        #module.register_buffer('out_mask', module.weight.new_ones(len(module.weight)))
        module.register_buffer('out_mask', module.weight.new_ones((1, len(module.weight), 1, 1), ))

    if type(module).__name__ == "Linear":
        module.register_buffer('in_mask', module.weight.new_ones(module.in_features))
        module.register_buffer('out_mask', module.weight.new_ones(module.out_features))
        module.finetune = not pruning
        def modified_forward_l(self, feature):
            if not self.finetune:
                feature = feature * self.in_mask
            return F.linear(feature, self.weight, self.bias)

        module.forward = MethodType(modified_forward_l, module)
        module.finetune = not pruning
    


def quick_filter_prune_l2(model, pruning_perc):
    '''
    Prune pruning_perc% filters globally
    '''
    masks = []

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            masks.append(np.ones(p_np.shape).astype('float32'))                           

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))           
            max_value = np.max(value_this_layer)

            value_this_layer /= max_value

            values = np.concatenate((values, value_this_layer))

    threshold = np.percentile(values, pruning_perc)

    ind = 0
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))           
            max_value = np.max(value_this_layer)

            value_this_layer /= max_value

            masks[ind][value_this_layer < threshold] = 0.
            ind += 1
      
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks
def quick_block_prune_mask(model, pruning_perc):
    '''
    Prune pruning_perc% filters globally
    '''
    masks = []
    values = []
    ind = 0
    for p in model.parameters():
        if len(p.data.size()) == 4: # nasty way of selecting conv layer
          if type(p).__name__ == 'Conv2d':
            p_np = p.data.cpu().numpy()
            
            
            if 'module.backbone.layer4.2' in p.name:
               print(p.name)
               masks.append(np.zeros(p_np.shape).astype('float32')) 
            else:
               masks.append(np.ones(p_np.shape).astype('float32'))                
            """

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))           
            max_value = np.max(value_this_layer)

            value_this_layer /= max_value

            masks[ind][value_this_layer < threshold] = 0.
            ind += 1"""
      
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks

def quick_filter_prune_l1(model, pruning_perc, prev_msk=None,pt_mask=None,prev_indi=None, prunt=2):
    '''
    model, 
    pruning_perc, 
    prev_msk=None,
    pt_mask=None,
    prev_indi=None
    Prune pruning_perc% filters globally
    prunet: prune type
    '''

    masks = []
    masks_temp = {}
    keep_mask_temp = {}

    lock={'backbone':0.1,'rpn':1.0,'neck':1.0,'roi':0.5}
    values = []                 

    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
            #if len(p.weight.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.weight.data.cpu().numpy()
            masks.append(np.ones(p_np.shape).astype('float32'))
            if "downsample" not in p.name:
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                value_this_layer = np.sum(weight_copy, axis=(1, 2, 3))
                arg_max = np.argsort(value_this_layer)
                min_value, min_ind = arg_nonzero_min(list(value_this_layer))
                max_value = np.max(value_this_layer)
                value_this_layer /= max_value
                #if (p_np.shape[2]>1):
                if prune_ratio(p.name)>0.0:
                   values = np.concatenate((values, value_this_layer))
    threshold = np.percentile(values, pruning_perc)        
    ind = 0
    for p in model.modules():
        if prunt>1:
          if type(p).__name__ == 'Linear':
               value_this_layer2 = np.sum(p.weight.data.abs().clone().cpu().numpy(), axis=(1))
               if (pruning_perc==0) or (len(value_this_layer2)<256):
                    keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2)]
                    masks_temp[p.name]=[]
               elif 'bbox_head.shared_fcs' in p.name: 
                    masks_temp[p.name]= quick_linear_prune(model,p.name,pruning_perc)
                    keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2) if (i not in masks_temp[p.name])]
        if type(p).__name__ == 'Conv2d':
               #if len(p.weight.data.size()) == 4: # nasty way of selecting conv layer
                p_np = p.weight.data.cpu().numpy()
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                value_this_layer2 = np.sum(weight_copy, axis=(1, 2, 3))
                max_value = np.max(value_this_layer2)
                value_this_layer2 /= max_value
                if ("downsample" not in p.name) and ("conv_logits" not in p.name):
                  lc = 1.0-prune_ratio(p.name)
                  if (lc == 1.0) or (pruning_perc==0) or (len(value_this_layer2)<=32):
                      keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2)]
                      masks_temp[p.name]=[]
                  elif prev_msk:
                     if (len(prev_msk[p.name])>int(lc*len(value_this_layer2))):
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (elem>=threshold)]
                        ths=threshold
                        thsmax = np.percentile(value_this_layer2, (1-lc)*100)
                        if (len(keep_indices)%2!=0):
                           pcn = (1-((len(keep_indices)-1)/len(value_this_layer2)))*100
                           ths = np.percentile(value_this_layer2, pcn) 
                        if (threshold>thsmax):
                           ths = thsmax   
                        indices = [i for i, elem in enumerate(value_this_layer2) if (elem<ths)]
                        
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (i not in indices)]
                        mas = [value_this_layer2 >= ths]
                        masks[ind] = np.repeat(mas,p_np.shape[1]*p_np.shape[2]*p_np.shape[3]).astype('float32').reshape((p_np.shape))

                     elif (len(prev_msk[p.name])<=int(lc*len(value_this_layer2))):
                        masks[ind] = pt_mask[ind].detach().numpy()
                        indices = prev_indi[p.name]
                        keep_indices = prev_msk[p.name]
                     #if indices:      
                     #          print((len(indices),p_np.shape[0],p.name))
                     masks_temp[p.name]=indices
                     keep_mask_temp[p.name]=keep_indices
                if prunt>5:
                  if ("downsample" in p.name)or ("conv3" in p.name) and (p_np.shape[0]>32): 
                    ths = np.percentile(value_this_layer2, pruning_perc)
                    masks_temp[p.name]= [i for i, elem in enumerate(value_this_layer2) if (elem<ths)]
                    keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2) if (i not in masks_temp[p.name])]
                    
                    mas = np.ones(len(value_this_layer2)).astype('float32')
                    mas[masks_temp[p.name]]=0.
                    #print(mas.count_nonzero())
                    masks[ind] = np.repeat(mas,p_np.shape[1]*p_np.shape[2]*p_np.shape[3]).astype('float32').reshape((p_np.shape))
                ind += 1
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks , masks_temp,keep_mask_temp
def quick_filter_prune_l1_vst(model, pruning_perc, prev_msk=None,pt_mask=None,prev_indi=None, prunt=2):
    '''
    model, 
    pruning_perc, 
    prev_msk=None,
    pt_mask=None,
    prev_indi=None
    Prune pruning_perc% filters globally
    prunet: prune type
    '''

    masks = []
    masks_temp = {}
    keep_mask_temp = {}

    lock={'backbone':0.1,'rpn':1.0,'neck':1.0,'roi':0.5}
    values = []                 

    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
            #if len(p.weight.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.weight.data.cpu().numpy()
            masks.append(np.ones(p_np.shape).astype('float32'))
            if "downsample" not in p.name:
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                value_this_layer = np.sum(weight_copy, axis=(1, 2, 3))
                arg_max = np.argsort(value_this_layer)
                min_value, min_ind = arg_nonzero_min(list(value_this_layer))
                max_value = np.max(value_this_layer)
                value_this_layer /= max_value
                #if (p_np.shape[2]>1):
                if prune_ratio(p.name)>0.0:
                   values = np.concatenate((values, value_this_layer))    
    ind = 0
    for p in model.modules():
        if prunt>1:
          if type(p).__name__ == 'Linear':
               value_this_layer2 = np.sum(p.weight.data.abs().clone().cpu().numpy(), axis=(1))
               if (pruning_perc==0) or (len(value_this_layer2)<256):
                    keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2)]
                    masks_temp[p.name]=[]
               elif 'bbox_head.shared_fcs' in p.name: 
                    masks_temp[p.name]= quick_linear_prune(model,p.name,pruning_perc)
                    keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2) if (i not in masks_temp[p.name])]
        if type(p).__name__ == 'Conv2d':
                p_np = p.weight.data.cpu().numpy()
                masks.append(np.ones(p_np.shape).astype('float32'))
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                value_this_layer2 = np.sum(weight_copy, axis=(1, 2, 3))
                max_value = np.max(value_this_layer2)
                value_this_layer2 /= max_value
                threshold = np.percentile(value_this_layer2, pruning_perc)    
                if ("downsample" not in p.name) and ("conv_logits" not in p.name):
                  lc = 1.0-prune_ratio(p.name)
                  if (lc == 1.0) or (pruning_perc==0) or (len(value_this_layer2)<=32):
                      keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2)]
                      masks_temp[p.name]=[]
                  elif prev_msk:
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (elem>=threshold)]
                        ths=threshold
                        thsmax = np.percentile(value_this_layer2, (1-lc)*100)
                        if (len(keep_indices)%2!=0):
                           pcn = (1-((len(keep_indices)-1)/len(value_this_layer2)))*100
                           ths = np.percentile(value_this_layer2, pcn) 
                        if (ths>thsmax):
                           ths = thsmax   
                        indices = [i for i, elem in enumerate(value_this_layer2) if (elem<ths)]
                        print(p.name, len(indices))
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (i not in indices)]
                        mas = [value_this_layer2 >= ths]
                        masks[ind] = np.repeat(mas,p_np.shape[1]*p_np.shape[2]*p_np.shape[3]).astype('float32').reshape((p_np.shape))

                        masks_temp[p.name]=indices
                        keep_mask_temp[p.name]=keep_indices
                ind += 1
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks , masks_temp,keep_mask_temp    
def quick_filter_prune_l1_v2(model,layer,previous_indice,alpha=32):
    '''
    Prune pruning_perc% filters one layer
    '''
    desierd_fil = int((len(previous_indice)/alpha)+1)*alpha

    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
           if layer in p.name:
                p_np = p.weight.data.cpu().numpy()
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                shape= p.weight.data.shape
                score = p.weight.abs().mean(dim=[1, 2, 3]) 
                value_this_layer2 = np.mean(weight_copy, axis=(1, 2, 3))
                pruning_perc = float(desierd_fil/len(value_this_layer2))*100
                threshold = np.percentile(value_this_layer2, pruning_perc)   
                indices = [i for i, elem in enumerate(value_this_layer2) if (elem<threshold)]
                indices2 = _pick_pruned(score,pruning_perc/100)
                keep_indices = [i for i, elem in enumerate(value_this_layer2) if (elem>=threshold)]
                return indices
def quick_filter_prune_l1_os(model, pruning_perc, prev_msk=None,pt_mask=None,prev_indi=None, prunt=1):
    '''
    model, 
    pruning_perc, 
    prev_msk=None,
    pt_mask=None,
    prev_indi=None
    Prune pruning_perc% filters globally
    prunet: prune type
    '''
    masks = []
    masks_temp = {}
    keep_mask_temp = {}      
    ind = 0
    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
                p_np = p.weight.data.cpu().numpy()
                masks.append(np.ones(p_np.shape).astype('float32'))
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                value_this_layer2 = np.sum(weight_copy, axis=(1, 2, 3))
                max_value = np.max(value_this_layer2)
                value_this_layer2 /= max_value
                threshold = np.percentile(value_this_layer2, pruning_perc)    
                if ("downsample" not in p.name) and ("conv_logits" not in p.name):
                  lc = 1.0-prune_ratio(p.name)
                  if (lc == 1.0) or (pruning_perc==0) or (len(value_this_layer2)<=32):
                      keep_mask_temp[p.name] = [i for i, elem in enumerate(value_this_layer2)]
                      masks_temp[p.name]=[]
                  elif prev_msk:
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (elem>=threshold)]
                        ths=threshold
                        thsmax = np.percentile(value_this_layer2, (1-lc)*100)
                        if (len(keep_indices)%2!=0):
                           pcn = (1-((len(keep_indices)-1)/len(value_this_layer2)))*100
                           ths = np.percentile(value_this_layer2, pcn) 
                        if (ths>thsmax):
                           ths = thsmax   
                        indices = [i for i, elem in enumerate(value_this_layer2) if (elem<ths)]
                        print(p.name, len(indices))
                        keep_indices = [i for i, elem in enumerate(value_this_layer2) if (i not in indices)]
                        mas = [value_this_layer2 >= ths]
                        masks[ind] = np.repeat(mas,p_np.shape[1]*p_np.shape[2]*p_np.shape[3]).astype('float32').reshape((p_np.shape))

                        masks_temp[p.name]=indices
                        keep_mask_temp[p.name]=keep_indices
                ind += 1
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks , masks_temp,keep_mask_temp
def quick_channel_prune_l1_v1(model,pruning_perc):
    '''
    Prune pruning_perc% channels one layer
    '''
    masks = []
    masks_temp = {}
    keep_mask_temp = {}

    lock={'backbone':0.1,'rpn':1.0,'neck':1.0,'roi':0.5}
    values = []                 
    for p in model.modules():
        if type(p).__name__ == 'Conv2d':
            #if len(p.weight.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.weight.data.cpu().numpy()
            masks.append(np.ones(p_np.shape).astype('float32'))
            if "downsample" not in p.name:
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                for i in range(p_np.shape[0]):
                    weight_copy2 = weight_copy[i]
                    value_this_layer = np.sum(weight_copy2, axis=(1, 2))
                    arg_max = np.argsort(value_this_layer)
                    min_value, min_ind = arg_nonzero_min(list(value_this_layer))
                    max_value = np.max(value_this_layer)
                    value_this_layer /= max_value
                    if prune_ratio2(p.name)>0.0:
                       values = np.concatenate((values, value_this_layer))        
    #value_this_layer2 = np.mean(weight_copy, axis=(0, 2, 3))
    #pruning_perc = float(channeltoprune/len(value_this_layer2))*100
    threshold = np.percentile(values, pruning_perc)  
    ind = 0
    for p in model.modules():
            if pruning_perc>1:
                if type(p).__name__ == 'Conv2d':
                    #if len(p.weight.data.size()) == 4: # nasty way of selecting conv layer
                    p_np = p.weight.data.cpu().numpy()
                    weight_copy = p.weight.data.abs().clone().cpu().numpy()
                    value_this_layer2=[]
                    for i in range(p_np.shape[0]):
                        weight_copy2 = weight_copy[i]
                        value_this_layer = np.sum(weight_copy2, axis=(1, 2))
                        max_value = np.max(value_this_layer)
                        value_this_layer /= max_value
                        value_this_layer2 = np.concatenate((value_this_layer2, value_this_layer)) 
                    if ("downsample" not in p.name) and ("conv_logits" not in p.name):
                        lc = 1.0-prune_ratio2(p.name)
                        ths=threshold
                        thsmax = np.percentile(value_this_layer2, (1-lc)*100)
                        if (threshold>thsmax):
                           ths = thsmax   
                        mas = [value_this_layer2 >= ths]
                        masks[ind]= np.repeat(mas,p_np.shape[2]*p_np.shape[3]).astype('float32').reshape((p_np.shape))
                    ind += 1
    masks = [torch.from_numpy(mask) for mask in masks]
    return masks 
            
def quick_linear_prune(model,layer,pruning_perc=0,previous_indice=None, metric='l1',alpha=32):
    '''
    Prune pruning_perc% nodes one layer
    '''
    if pruning_perc>50: pruning_perc=50
    if previous_indice : desierd_nodes = int((len(previous_indice)/alpha)+1)*alpha 

    for p in model.modules():
        if type(p).__name__ == 'Linear':
           if layer in p.name:
                p_np = p.weight.data.cpu().numpy()
                weight_copy = p.weight.data.abs().clone().cpu().numpy()
                shape= p.weight.data.shape
                score = p.weight.abs().mean(dim=[1]) 
                value_this_layer2 = np.mean(weight_copy, axis=(1))
                if pruning_perc==0:
                    pruning_perc = float(desierd_nodes/len(value_this_layer2))*100
                threshold = np.percentile(value_this_layer2, pruning_perc)   
                indices = [i for i, elem in enumerate(value_this_layer2) if (elem<threshold)]
                keep_indices = [i for i, elem in enumerate(value_this_layer2) if (elem>=threshold)]
                mas = [value_this_layer2 >= threshold]
                masks = np.repeat(mas,p_np.shape[1]).astype('float32').reshape((p_np.shape))
                p.register_buffer('mask', torch.from_numpy(masks)) 
                p.mask.requires_grad = False
                p.mask=p.mask.cuda()
                p.weight=nn.Parameter(p.weight*p.mask)
                return indices
def prune_ratio(layer_name):
    stage=0
    stages_pr=[0,0,0,0,0,0,0,0]    
    stages_pr=[0,0.125 ,0.25 ,0.5,0.75, 0.2, 0.5, 0.5]    
    #stages_pr=[0,0.5,0.75,0.87,0.93,0.5,0.75,0.75]
    #stages_pr=[0,0.2,0.5,0.87,0.93,0.2,0.75,0.75]
    #stages_pr=[0,0.3,0.3,0.3,0.14, 0.3, 0.3, 0.14]
    if 'backbone' in layer_name:
        RTname =layer_name[16:]          
        if RTname=="conv1":
            stage = 0
        elif "conv3" not in RTname:
             stage  = int(RTname.split(".")[0][-1])
        if "conv3" in RTname:
            stage = 0
            #stage = int(RTname.split(".")[0][-1])
        if "0" in layer_name:
            stage = 0
    #if 'neck' in layer_name:
    #    #stage  = 5
    #    stage =0
    if 'rpn_conv' in layer_name:
         #if 'rpn_conv' in layer_name:
         stage = 6
    if 'roi_head' in layer_name:
         stage = 7
         if 'convs.3' in layer_name:
             stage = 0
    return stages_pr[stage]
def prune_ratio2(layer_name):
    stage=0
    #stages_pr=[0,0,0,0,0,0,0,0]    
    stages_pr=[0,0.5,0.75,0.87,0.93,0.5,0.75,0.75]
    #stages_pr=[0,0.3,0.3,0.3,0.14, 0.3, 0.3, 0.14]
    if 'backbone' in layer_name:
        RTname =layer_name[16:]          
        if RTname=="conv1":
            stage = 0
        elif "conv3" not in RTname:
             stage  = int(RTname.split(".")[0][-1])
        if "conv3" in RTname:
            stage = int(RTname.split(".")[0][-1])
    if 'neck' in layer_name:
        stage  = 5
    #    stage =0
    if 'rpn_conv' in layer_name:
         #if 'rpn_conv' in layer_name:
         stage = 6
    if 'roi_head' in layer_name:
         stage = 7
         if 'convs.3' in layer_name:
             stage = 0
    return stages_pr[stage]
def _pick_pruned( w_abs, pr, mode="min"):
        if pr == 0:
            return []
        w_abs_list = w_abs.flatten()
        n_wg = len(w_abs_list)
        n_pruned = min(ceil(pr * n_wg), n_wg - 1) # do not prune all
        if mode == "rand":
            out = np.random.permutation(n_wg)[:n_pruned]
        elif mode == "min":
            out = w_abs_list.sort()[1][:n_pruned]
        elif mode == "max":
            out = w_abs_list.sort()[1][-n_pruned:]
        return out        
def get_filter_codebook( weight_torch, compress_rate, length):
        codebook = np.ones(length).astype('float32')
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0.

            #print("filter codebook done")
        else:
            pass
        return codebook 

def create_mask(model):
    mask = {}
    n_mask_dims = 0
    for name, param in model.named_parameters():
        if 'weight' in name and (('conv' in name) or ('fpn' in name) \
            or ('fcn' in name) or ('fc' in name) or ('fc2') in name ):
            if 'conv_logits' in name:
                 continue

            #if self.check_modules(name):
            print('Pruning: ',name)
            mask[name] = torch.ones_like(param)
            n_mask_dims += param.numel()

    return mask,n_mask_dims

def quick_prune_lth(model,keep_percentage,mask=None,n_mask_dims=None):
    if mask:
        new_mask = mask
    else:
        new_mask, n_mask_dims = create_mask(model)
    mask_vec = torch.zeros(n_mask_dims).to(list(new_mask.values())[0])
    start_ind = 0
    for name, param in model.named_parameters():
        if name in new_mask:
            mask_vec[start_ind:start_ind+param.numel()] = torch.abs(param).reshape(-1)
            start_ind += param.numel()

    cur_keep_percentage = keep_percentage
    thresh = torch.topk(mask_vec,int(cur_keep_percentage*n_mask_dims), sorted=True)[0][-1]
    n_zeros = 0
    new_state_dict = model.state_dict()
    for name in model.state_dict():
        if name in new_mask:
                param = model.state_dict()[name]
                n_zeros += torch.sum(torch.abs(param)==0.0).item()
                mask_val = new_mask[name]
                mask_val[torch.abs(param)<thresh] = 0
                new_mask[name] = mask_val
                new_state_dict[name] = param*mask_val

    mask = new_mask
    model.load_state_dict(new_state_dict)
    
    return mask ,n_mask_dims
def apply_mask(model,mask):
    #apply mask on the state dictionatory in th elth techninqes
    new_state_dict = model.state_dict()
    for name in model.state_dict():
        if name in mask:
            new_state_dict[name] = model.state_dict()[name]*mask[name]
    model.load_state_dict(new_state_dict) 
    #print('applied mask')       
 
def quick_sfp_filter_prune(model, pruning_perc):
    '''
    Prune pruning_perc% filters globally
    '''
    masks = []

    values = []                 
    ind = 0
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()
            masks.append(np.ones(p_np.shape).astype('float32'))
            weight_copy = p.data.abs().clone().cpu().numpy()
            cp_rate=1-pruning_perc
            #if (p_np.shape[2]==1):
            #    cp_rate=1
            msk = get_filter_codebook(p.data, cp_rate,int(np.prod(weight_copy.shape)))
            msk=np.reshape(msk,(p_np.shape))
            masks[ind] = msk
            ind += 1
      
    masks = [torch.FloatTensor(torch.from_numpy(mask)) for mask in masks]
    return masks


    
    
