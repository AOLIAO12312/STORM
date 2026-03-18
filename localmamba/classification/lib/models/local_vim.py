import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights

import math

from .mamba.multi_mamba import MultiMamba

from .mamba.rope import *

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

import torch.nn.functional as F

from localmamba.classification.lib.models.token_reduction.ToMe1D import RMeeTo_Merge
from localmamba.classification.lib.models.token_reduction.RandomPrune import RandomPrune
from localmamba.classification.lib.models.token_reduction.InterpolatePrune import InterpolatePrune
from localmamba.classification.lib.models.token_reduction.ToMe2D import ToMe2D
from localmamba.classification.lib.models.token_reduction.ConvToMe2D import ConvToMe2D
class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
            block_idx=None,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.block_idx = block_idx

        schedual_0 = {}

        schedual_1 = {
            4: 169,
            10: 144,
            16: 100,
            22: 81
        }

        schedual_2 = {
            4: 144,
            10: 100,
            16: 64,
            22: 36
        }

        schedual_3 = {
            4: 100,
            10: 64,
            16: 36,
            22: 16
        }

        schedual_4 = {
            4: 100,
            10: 64,
            16: 25,
            22: 9
        }

        schedual_5 = {
            4: 64,
            10: 36,
            16: 16,
            22: 9
        }
        self.schedual = schedual_5
        self.prune_method = "conv_tome2d"  # tome/interpolate/none/random/tome2d
        self.merge = RMeeTo_Merge(
            class_token=False,
            num_prune=0,
            if_prune=False,  # merge/prune
            if_order=True,
            distance="cosine",
            metric='X',
            merge_mode="sum",
            choose="max"
        )
        self.merge2d = ToMe2D(if_order=True, distance='cosine', merge_mode='sum')
        self.interpolate_prune = InterpolatePrune(assume_square=True)
        self.random_prune = RandomPrune(share_across_batch=True)
        self.conv_tome_2d = ConvToMe2D(
            imp_mode="l2",
            alpha=1.0,
            beta=50.0,
        )
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    # 单层双向SSM
    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        def prune_and_downsample(hidden_states: torch.Tensor, num_prune: int) -> torch.Tensor:
            """
            对输入序列执行剪枝和下采样操作（针对 Vision-Mamba 类结构）.

            Args:
                hidden_states: torch.Tensor, shape [B, M, C], 其中包含 cls_token (位于中间 M//2).
                num_prune: int, 本层需裁剪的 token 数量 (不包括 cls_token).

            Returns:
                torch.Tensor: 更新后的 token 序列, shape [B, M_new, C].
            """
            if num_prune <= 0:
                return hidden_states  # 若不减则直接返回

            B, M, C = hidden_states.shape
            cls_pos = M // 2  # cls token 默认位置在中间

            # 1. 分离 cls token + 其余 token
            cls_token = hidden_states[:, cls_pos:cls_pos + 1, :]  # [B, 1, C]
            patch_tokens = torch.cat(
                [hidden_states[:, :cls_pos, :], hidden_states[:, cls_pos + 1:, :]], dim=1
            )  # [B, M-1, C]

            # 2. reshape 为 [B, C, H, W], 其中 H * W = M - 1 必须为平方数
            old_hw = M - 1
            H = W = int(math.sqrt(old_hw))
            assert H * W == old_hw, f"Old token count {old_hw} is not a perfect square."

            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)

            # 3. 计算新的序列长度，确保新的 token (不含 cls) 为平方数
            M_new = M - num_prune
            new_hw = M_new - 1
            H_new = W_new = int(math.sqrt(new_hw))
            assert H_new * W_new == new_hw, f"New token count {new_hw} is not a perfect square."

            # 4. 最近邻下采样为 [B, C, H_new, W_new]
            patch_tokens = F.interpolate(patch_tokens, size=(H_new, W_new), mode="area")  # bilinear/nearest/area

            # 5. reshape 回 [B, new_hw, C]
            patch_tokens = patch_tokens.reshape(B, C, new_hw).permute(0, 2, 1).contiguous()

            # 6. 插回 cls_token 至中间位置
            insert_pos = M_new // 2
            hidden_states = torch.cat(
                [patch_tokens[:, :insert_pos, :], cls_token, patch_tokens[:, insert_pos:, :]],
                dim=1
            )  # [B, M_new, C]

            return hidden_states
        B,L,D = hidden_states.shape
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                # 执行add+norm的操作（合并操作）
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        # 输出侧剪枝
        # No need to consider cls_token
        if self.prune_method != 'none':
            L_target = self.schedual.get(self.layer_idx, L)
            num_prune = L - L_target
            if self.prune_method == "interpolate":
                if num_prune != 0:
                    hidden_states = self.interpolate_prune(hidden_states, num_prune)
                    residual = self.interpolate_prune(residual, num_prune)
            elif self.prune_method == "tome":
                if num_prune != 0:
                    size = torch.ones(B, L, 1, device=hidden_states.device)
                    merge_fn = self.merge(hidden_states, token_position=None, num_prune=num_prune)
                    hidden_states, sizes, cls_new_pos = self.merge.merge_wavg(merge_fn, hidden_states,
                                                                                          size=size)
                    residual, _, _ = self.merge.merge_wavg(merge_fn, residual, size=size)
            elif self.prune_method == "random":
                prune_fn = self.random_prune(hidden_states, num_prune)
                hidden_states = prune_fn(hidden_states)
                residual = prune_fn(residual)
            elif self.prune_method == "tome2d":
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H, W = int(L**0.5),int(L**0.5)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
                # [B, L, D] -> [B, D, H, W]
                hidden_states_bchw = hidden_states.transpose(1, 2).reshape(B, D, H, W)
                residual_bchw = residual.transpose(1, 2).reshape(B, D, H, W)

                if hidden_states_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.merge2d(
                        residual_bchw,
                        num_prune_w=H - H_new,
                        num_prune_h=H - H_new,
                    )
                    hidden_states_bchw = prune_fn(hidden_states_bchw)
                    residual_bchw = prune_fn(residual_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                hidden_states = hidden_states_bchw.flatten(2).transpose(1, 2)
                residual = residual_bchw.flatten(2).transpose(1, 2)
            elif self.prune_method == "conv_tome2d":
                # L = H * W，这里也可以直接用 H_new = math.isqrt(L_target)
                H, W = int(L**0.5),int(L**0.5)
                H_new = math.isqrt(L - num_prune)  # 或 math.isqrt(L_target)
                # [B, L, D] -> [B, D, H, W]
                hidden_states_bchw = hidden_states.transpose(1, 2).reshape(B, D, H, W)
                residual_bchw = residual.transpose(1, 2).reshape(B, D, H, W)

                if hidden_states_bchw.shape[-2:] != (H_new, H_new):
                    prune_fn = self.conv_tome_2d(
                        residual_bchw,
                        target_hw=(H_new, H_new)
                    )
                    hidden_states_bchw = prune_fn(hidden_states_bchw)
                    residual_bchw = prune_fn(residual_bchw)

                # [B, D, H_new, H_new] -> [B, L_new, D]
                hidden_states = hidden_states_bchw.flatten(2).transpose(1, 2)
                residual = residual_bchw.flatten(2).transpose(1, 2)

        return hidden_states, residual


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
    directions=None,
    use_middle_cls_token=False,
    token_size=(14, 14),
    mamba_cls=None,
    block_idx=None
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(mamba_cls, layer_idx=layer_idx, bimamba_type=bimamba_type, directions=directions, token_size=token_size, use_middle_cls_token=use_middle_cls_token, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        block_idx=block_idx,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 use_middle_cls_token=False,
                 directions=None,
                 mamba_cls=MultiMamba,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.patch_size = patch_size

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim,
            # strict_img_size=False, dynamic_img_pad=True
        )
        num_patches = self.patch_embed.num_patches
        self.token_size = self.patch_embed.grid_size

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            if isinstance(img_size, (tuple, list)):
                hw_seq_len = img_size[0] // patch_size
            else:
                hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        if directions is None:
            directions = [None] * depth
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    directions=directions[i],
                    use_middle_cls_token=use_middle_cls_token,
                    token_size=self.token_size,
                    mamba_cls=mamba_cls,
                    block_idx=i,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, out_indices=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.use_middle_cls_token:
                token_position = x.shape[1] // 2
                # add cls token to the middle of sequence
                x = torch.cat([x[:, :token_position, :], cls_token, x[:, token_position:, :]], dim=1)
            else:
                # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            H, W = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
            for layer in self.layers:
                layer.mixer.multi_scan.token_size = (H, W)
            if H != self.token_size[0] or W != self.token_size[1]:
                # downstream tasks such as det and seg may have various input resolutions
                pos_embed = Backbone_LocalVisionMamba.resize_pos_embed(self.pos_embed, (H, W), self.token_size, 'bicubic')
                if self.if_rope:
                    freqs_cos = Backbone_LocalVisionMamba.resize_pos_embed(self.rope.freqs_cos.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
                    freqs_sin = Backbone_LocalVisionMamba.resize_pos_embed(self.rope.freqs_sin.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
            else:
                pos_embed = self.pos_embed
                freqs_cos = None
                freqs_sin = None
            x = x + pos_embed
            x = self.pos_drop(x)

        outs = []

        # mamba impl
        residual = None
        hidden_states = x
        for layer_idx, layer in enumerate(self.layers):
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            if out_indices is not None and layer_idx in out_indices:
                outs.append(hidden_states)

        if out_indices is not None:
            assert len(outs) == len(out_indices)
            return outs, (H, W)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_middle_cls_token:
                return hidden_states[:, token_position, :]
            else:
                return hidden_states[:, 0, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)
        return x
        
    def flops(self, input_shape=(3, 224, 224)):
        flops = 0
        from classification.lib.utils.measure import get_flops
        flops += get_flops(self.patch_embed, input_shape)

        L = self.patch_embed.num_patches
        if self.if_cls_token:
            L += 1
        for layer in self.layers:
            # 1 in_proj
            flops += layer.mixer.in_proj.in_features * layer.mixer.in_proj.out_features * L
            # 2 MambaInnerFnNoOutProj
            # 2.1 causual conv1d
            flops += (L + layer.mixer.d_conv - 1) * layer.mixer.d_inner * layer.mixer.d_conv
            # 2.2 x_proj
            flops += L * layer.mixer.x_proj_0.in_features * layer.mixer.x_proj_0.out_features
            # 2.3 dt_proj
            flops += L * layer.mixer.dt_proj_0.in_features * layer.mixer.dt_proj_0.out_features
            # 2.4 selective scan
            """
            u: r(B D L)
            delta: r(B D L)
            A: r(D N)
            B: r(B N L)
            C: r(B N L)
            D: r(D)
            z: r(B D L)
            delta_bias: r(D), fp32
            """
            D = layer.mixer.d_inner
            N = layer.mixer.d_state
            for i in range(len(layer.mixer.multi_scan.choices)):
                # flops += 9 * L * D * N + 2 * D * L
                # A
                flops += D * L * N
                # B
                flops += D * L * N * 2
                # C
                flops += (D * N + D * N) * L
                # D
                flops += D * L
                # Z
                flops += D * L
            # merge
            attn = layer.mixer.attn
            flops += attn.global_reduce.in_features * attn.global_reduce.out_features
            # flops += attn.local_reduce.in_features * attn.local_reduce.out_features * L
            flops += attn.channel_select.in_features * attn.channel_select.out_features
            # flops += attn.spatial_select.in_features * attn.spatial_select.out_features * L
            # 2.5 out_proj
            flops += L * layer.mixer.out_proj.in_features * layer.mixer.out_proj.out_features
            # layer norm
            flops += L * layer.mixer.out_proj.out_features

        # head
        flops += self.embed_dim * 1000
        return flops


class Backbone_LocalVisionMamba(VisionMamba):
    def __init__(self, out_indices=[4, 9, 14, 19], pretrained_ckpt=None, **kwargs):
        super().__init__(**kwargs)
        del self.head
        del self.norm_f

        self.out_indices = out_indices
        for i in range(len(out_indices)):
            layer = nn.LayerNorm(self.embed_dim)
            layer_name = f'outnorm_{i}'
            self.add_module(layer_name, layer)

        self.load_pretrained(pretrained_ckpt)


    def load_pretrained(self, ckpt):
        if ckpt is None:
            return
        print(f'Load backbone state dict from {ckpt}')
        if ckpt.startswith('http'):
            from mmengine.utils.dl_utils import load_url
            state_dict = load_url(ckpt, map_location='cpu')['state_dict']
        else:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        if 'pos_embed' in state_dict:
            pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1]))
            state_dict['pos_embed'] = self.resize_pos_embed(
                state_dict['pos_embed'],
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )
        if 'rope.freqs_cos' in state_dict:
            pos_size = int(math.sqrt(state_dict['rope.freqs_cos'].shape[0]))
            state_dict['rope.freqs_cos'] = self.resize_pos_embed(
                state_dict['rope.freqs_cos'].unsqueeze(0),
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )[0]
        if 'rope.freqs_cos' in state_dict:
            pos_size = int(math.sqrt(state_dict['rope.freqs_sin'].shape[0]))
            state_dict['rope.freqs_sin'] = self.resize_pos_embed(
                state_dict['rope.freqs_sin'].unsqueeze(0),
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )[0]
        res = self.load_state_dict(state_dict, strict=False)
        print(res)
    
    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        from mmseg.models.utils import resize
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        pos_embed_weight = pos_embed
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return pos_embed_weight

    def forward(self, x):
        C = self.embed_dim
        outs, (H, W) = self.forward_features(x, out_indices=self.out_indices)
        outs = [getattr(self, f'outnorm_{i}')(o) for i, o in enumerate(outs)]
        outs = [o.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous() for o in outs]
        if len(self.out_indices) == 1:
            return outs[0]
        return outs


@register_model
def local_vim_tiny_search(pretrained=False, **kwargs):
    directions = None
    model = VisionMamba(
        patch_size=16, embed_dim=128, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean',
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def local_vim_tiny(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, mamba_cls=MultiMamba, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def local_vim_tiny_middle_cls_token(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    """
    Changes:
    1. disable rope
    2. add middle cls token
    """

    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", directions=directions, mamba_cls=MultiMamba,
        if_cls_token=True, use_middle_cls_token=True, **kwargs)
    # if_cls_token=True, if_devide_out=True, use_middle_cls_token=True
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def local_vim_small(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def local_vim_tiny_wo_search(pretrained=False, **kwargs):
    directions = (('h', 'h_flip', 'w2', 'w2_flip'),) * 20
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, mamba_cls=MultiMamba, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def local_vim_small(pretrained=False, **kwargs):
    directions = (('h', 'h_flip', 'w2', 'w2_flip'),) * 20
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model