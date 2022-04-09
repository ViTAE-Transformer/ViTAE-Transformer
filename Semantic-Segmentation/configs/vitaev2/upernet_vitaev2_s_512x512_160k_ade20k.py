_base_ = [
    '../_base_/models/upernet_vitaev2.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        in_chans=3,
        RC_tokens_type=['window', 'window', 'transformer', 'transformer'], 
        NC_tokens_type=['window', 'window', 'transformer', 'transformer'], 
        embed_dims=[64, 64, 128, 256], 
        token_dims=[64, 128, 256, 512], 
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2], 
        NC_heads=[1, 2, 4, 8], 
        RC_heads=[1, 1, 2, 4], 
        mlp_ratio=4., 
        NC_group=[1, 32, 64, 128], 
        RC_group=[1, 16, 32, 64],
        use_checkpoint=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        window_size=7,
    ),
    # neck=dict(in_channels=[64, 128, 256, 512])
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
find_unused_parameters=True