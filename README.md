# mmdetection-prune
This is the code for pruning the mmdetection model with filter pruning and channel pruning.
## Get Started
#### 1. Creat a basic environment with pytorch 1.3.0  and mmcv-full

#### Due to the frequent changes of the autograd interface, we only guarantee the code works well in `pytorch==1.3.0`.


1. Creat the environment
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```
2. Install PyTorch 1.3.0 and corresponding torchvision.
```shell
conda install pytorch=1.3.0 cudatoolkit=10.0 torchvision=0.2.2 -c pytorch
```
3.  Build the ``mmcv-full`` from source with pytorch 1.3.0 and cuda 10.0 : mmcv-version 1.3.16
#### Please use gcc-5.4 and nvcc 10.0
```shell
 git clone https://github.com/open-mmlab/mmcv.git
 cd mmcv
 MMCV_WITH_OPS=1 pip install -e .
```

#### 2. Install the corresponding codebase in [OpenMMLab](https://github.com/open-mmlab).

e.g. [MMdetection](https://github.com/open-mmlab/mmdetection)

```shell
pip install mmdet==2.18.0
```

#### 3. Pruning the model and Fine tuning.

e.g. Detection

Modify the `load_from` as the path to the baseline model in  of `xxxx_pruning.py`

```shell
# for slurm train
sh tools/slurm_train.sh PATITION_NAME JOB_NAME configs/retina/retina_pruning.py work_dir
# for slurm_test
sh tools/slurm_test.sh PATITION_NAME JOB_NAME configs/retina/retina_pruning.py PATH_CKPT --eval bbox
# for torch.dist
# sh tools/dist_train.sh configs/retina/retina_pruning.py 8
```
Built on top of these amazing libraries:
[MMdetection](https://github.com/open-mmlab/mmdetection) 
[MMCV](https://github.com/open-mmlab/mmcv.git)
[torch-pruning](https://github.com/VainF/Torch-Pruning)
[FisherPruning](https://github.com/jshilong/FisherPruning)