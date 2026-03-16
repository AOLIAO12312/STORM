# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number
import logging
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from torch import nn

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)

import selective_scan_cuda

def patch_selective_scan_fwd_print_once():
    if getattr(selective_scan_cuda, "_patched_print_shapes", False):
        return
    selective_scan_cuda._patched_print_shapes = True

    orig_fwd = selective_scan_cuda.fwd
    printed = {"done": False}

    def fwd_hook(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        if not printed["done"]:
            printed["done"] = True

            def shp(t):
                try:
                    return None if t is None else tuple(t.shape)
                except Exception:
                    return str(type(t))

            print(
                "[selective_scan_cuda.fwd] shapes:",
                "u", shp(u),
                "delta", shp(delta),
                "A", shp(A),
                "B", shp(B),
                "C", shp(C),
                "D", shp(D),
                "z", shp(z),
                "delta_bias", shp(delta_bias),
                "delta_softplus", delta_softplus,
                flush=True
            )
        return orig_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    selective_scan_cuda.fwd = fwd_hook


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


from fvcore.nn.jit_handles import get_shape
from fvcore.nn import FlopCountAnalysis
def _to_int(x):
    # traced shape 里经常是 tensor(196) 这种
    try:
        return int(x)
    except Exception:
        try:
            return int(x.item())
        except Exception:
            return int(x)

def _numel(shape):
    n = 1
    for s in shape:
        n *= _to_int(s)
    return n

def eltwise_1_flop(inputs, outputs):
    # 对输出每个元素计 1 FLOP
    out_shape = get_shape(outputs[0])
    return _numel(out_shape)

def silu_flop(inputs, outputs):
    # silu(x)=x*sigmoid(x); sigmoid ~ exp + add + div
    # 粗略按每元素 12 FLOPs（口径可调）
    out_shape = get_shape(outputs[0])
    return 12 * _numel(out_shape)

def gelu_flop(inputs, outputs):
    # 常用近似 GELU 口径：每元素 ~ 8-10 FLOPs（口径可调）
    out_shape = get_shape(outputs[0])
    return 10 * _numel(out_shape)

def mean_flop(inputs, outputs):
    in_shape = get_shape(inputs[0])
    out_shape = get_shape(outputs[0])
    in_n = _numel(in_shape)
    out_n = _numel(out_shape)
    # 每个输出需要 (in_n/out_n - 1) 次加法，粗略：in_n - out_n
    return max(in_n - out_n, 0)

def softmax_flop(inputs, outputs):
    # 按最后一维做 softmax：exp + reduce_sum + div
    in_shape = get_shape(inputs[0])
    # 假设 softmax 在最后一维
    last = _to_int(in_shape[-1])
    outer = _numel(in_shape[:-1])
    # per element: exp(1) + add(sum)(~1) + div(1) + (可选减max)
    # 简化给 5 FLOPs/element
    return outer * last * 8

def selective_scan_flop_jit(inputs, outputs):
    """
    估算 prim::PythonOp.SelectiveScanFn 的 FLOPs
    inputs 大致对应: u, delta, A, B, C, D, z, delta_bias, delta_softplus
    你的实际 shape:
      u,delta: (B, D, L)
      A: (D, d_state)
    """
    u_shape = get_shape(inputs[0])   # (B, D, L)
    A_shape = get_shape(inputs[2])   # (D, d_state)

    Bsz = _to_int(u_shape[0])
    Dim = _to_int(u_shape[1])
    L   = _to_int(u_shape[2])
    d_state = _to_int(A_shape[1])

    # 常用近似口径：每个 (b,d,t) ~ (8*d_state + 1) FLOPs
    per_token = 8.5 * d_state + 1
    return Bsz * Dim * L * per_token

class MMClsInferWrapper(nn.Module):
    def __init__(self, mmcls_model):
        super().__init__()
        self.mmcls_model = mmcls_model

    def forward(self, img):
        # 强制走推理分支，避免 gt_label
        return self.mmcls_model(img, return_loss=False)

def calc_gflops_params_mmcls(model, input_size=224, device='cuda'):
    # 兼容 DataParallel / DDP
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval().to(device)

    wrapper = MMClsInferWrapper(raw_model).eval().to(device)
    x = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        flops = FlopCountAnalysis(wrapper, (x,))

        # 在你的 flops.set_op_handle 中一起注册
        flops.set_op_handle(**{
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
            "aten::add": eltwise_1_flop,
            "aten::mul": eltwise_1_flop,
            "aten::neg": eltwise_1_flop,
            "aten::exp": eltwise_1_flop,
            "aten::silu": silu_flop,
            "aten::gelu": gelu_flop,
            "aten::mean": mean_flop,
            "aten::softmax": softmax_flop,
        })

        total_flops = flops.total()

    gflops = total_flops / 1e9
    params = sum(p.numel() for p in raw_model.parameters())
    mparams = params / 1e6
    return gflops, mparams, flops

def main():
    # patch_selective_scan_fwd_print_once()
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    # # ====== 这里插入：计算 GFLOPs / Params ======
    # gflops, mparams, flops_obj = calc_gflops_params_mmcls(model, input_size=224, device=cfg.device)
    # print(f"[GFLOPs] {gflops:.3f} | [Params(M)] {mparams:.3f}")

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
        if cfg.device == 'ipu':
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper
            opts = cfg2options(cfg.runner.get('options_cfg', {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts['inference'])
        model.CLASSES = CLASSES
        show_kwargs = args.show_options or {}
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = wrap_distributed_model(
            model, device=cfg.device, broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=args.metrics,
                metric_options=args.metric_options,
                logger=logger)
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
