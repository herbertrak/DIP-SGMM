import torch
import torch.nn as nn
from .common import *

def skip_gmm(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True,
        last_activation="sigmoid"  # "softmax" | "sigmoid" | "none"
    ):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
        last_activation (string): 'softmax' (over channel dim=1), 'sigmoid', or 'none'
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, (list, tuple))):
        upsample_mode = [upsample_mode]*n_scales
    if not (isinstance(downsample_mode, (list, tuple))):
        downsample_mode = [downsample_mode]*n_scales
    if not (isinstance(filter_size_down, (list, tuple))):
        filter_size_down = [filter_size_down]*n_scales
    if not (isinstance(filter_size_up, (list, tuple))):
        filter_size_up = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_branch = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_branch, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip_branch.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_branch.add(bn(num_channels_skip[i]))
            skip_branch.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))

    # ---- Last activation selection ----
    if isinstance(last_activation, str):
        la = last_activation.lower()
        if la == "sigmoid":
            model.add(nn.Sigmoid())
        elif la == "softmax":
            # Softmax over channel dimension → per-pixel categorical probabilities
            model.add(nn.Softmax(dim=1))
        elif la == "none":
            pass
        else:
            raise ValueError(f"Unknown last_activation: {last_activation}")
    else:
        # If a nn.Module was given
        model.add(last_activation)

    return model
