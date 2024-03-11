_base_ = ['../decoupled_solo_r50_fpn_3x_coco.py'
]
optimizer = dict(lr=0.002)
custom_hooks = [
    dict(
        type='BrushPruningHook',
        # In pruning process, you need set priority
        # as 'LOWEST' to insure the pruning_hook is excused
        # after optimizer_hook, in fintune process, you
        # should set it as 'HIGHEST' to insure it excused
        # before checkpoint_hook
        pruning=True,
        batch_size=2,
        interval=10,
        pr_type='lth',
        priority='LOWEST',
    )
]
#
model = dict(backbone=dict(frozen_stages=-1, ))
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_r50_fpn_1x_coco/decoupled_solo_r50_fpn_1x_coco_20210820_233348-6337c589.pth'  # noqa: E501
