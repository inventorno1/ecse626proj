from diffusers import UNet2DConditionModel

cdpm = UNet2DConditionModel(
    sample_size=128,
    in_channels=128,
    out_channels=128, #initialise to maximum and compute loss only for indexed slices
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    mid_block_type=None,
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    # not sure about what the lower arguments do
    block_out_channels=(128, 256),
    layers_per_block=2,
    cross_attention_dim=256, # concat one-hot encodings of condition and target slices
    attention_head_dim=8,
    norm_num_groups=32,
    use_linear_projection=True
)

print(cdpm)