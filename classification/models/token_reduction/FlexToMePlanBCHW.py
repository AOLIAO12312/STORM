import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize_feat(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def _similarity(a, b, distance='cosine'):
    if distance == 'cosine':
        return (_normalize_feat(a) * _normalize_feat(b)).sum(dim=-1)
    elif distance == 'l2':
        return -((a - b) ** 2).sum(dim=-1)
    else:
        raise NotImplementedError

def _make_frac_grid(H_in, W_in, H_out, W_out, align_corners=False, device=None):
    if align_corners:
        yy = torch.linspace(0, H_in - 1, H_out, device=device)
        xx = torch.linspace(0, W_in - 1, W_out, device=device)
    else:
        yy = (torch.arange(H_out, device=device) + 0.5) * (H_in / H_out) - 0.5
        xx = (torch.arange(W_out, device=device) + 0.5) * (W_in / W_out) - 0.5
    Yc = yy[:, None].expand(H_out, W_out)
    Xc = xx[None, :].expand(H_out, W_out)
    return Yc, Xc

def _neighbors_2x2(Yc, Xc):
    y0 = torch.floor(Yc).to(torch.long)
    x0 = torch.floor(Xc).to(torch.long)
    oy = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=Yc.device)
    ox = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=Yc.device)
    return y0, x0, oy, ox

def _neighbors_k(Yc, Xc, k):
    assert k % 2 == 1, "k 应为奇数"
    half = k // 2
    yc0 = torch.round(Yc).to(torch.long)
    xc0 = torch.round(Xc).to(torch.long)
    offs = torch.arange(-half, half + 1, device=Yc.device)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')
    oy = oy.reshape(-1).to(torch.long)
    ox = ox.reshape(-1).to(torch.long)
    return yc0, xc0, oy, ox

def _bilinear_base_weights(Yc, Xc, y0, x0):
    dy = (Yc - y0.float()).clamp(0, 1)
    dx = (Xc - x0.float()).clamp(0, 1)
    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx
    return torch.stack([w00, w01, w10, w11], dim=-1)  # [..., 4]

def _gaussian_base_weights(Yc, Xc, ys, xs, sigma=0.5):
    dy2 = (Yc[..., None] - ys.float()) ** 2
    dx2 = (Xc[..., None] - xs.float()) ** 2
    return torch.exp(-(dy2 + dx2) / (2 * (sigma ** 2)))  # [..., K]

class FlexiToMePlanBCHW(nn.Module):
    """
    任意尺度的“语义感知重采样/剪枝”规划器（BCHW）。
    forward(metric, target_hw, ...) 生成 prune_fn；prune_fn 可用于 token/残差/size 等。

    requires_grad_plan=True: 规划阶段可导（默认，便于端到端微调）
    requires_grad_plan=False: 规划阶段权重在闭包中 detach（省显存/更稳）
    """
    def __init__(self,
                 kernel=2,
                 distance='cosine',
                 weighting='softmax',
                 alpha=1.0,
                 beta=1.0,
                 pos_lambda=0.0,
                 gaussian_sigma=0.5,
                 gate_tau=None,
                 align_corners=False,
                 requires_grad_plan=True):
        super().__init__()
        self.kernel = kernel
        self.distance = distance
        self.weighting = weighting
        self.alpha = alpha
        self.beta = beta
        self.pos_lambda = pos_lambda
        self.gaussian_sigma = gaussian_sigma
        self.gate_tau = gate_tau
        self.align_corners = align_corners
        self.requires_grad_plan = requires_grad_plan

    def forward(self, metric: torch.Tensor, target_hw: tuple[int, int], pos: torch.Tensor = None):
        assert metric.dim() == 4, "metric 必须是 [B,C,H,W]"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        device = metric.device

        # 1) 输出网格在输入坐标系的连续坐标
        Yc, Xc = _make_frac_grid(H, W, H_out, W_out, self.align_corners, device=device)

        # 2) 邻域索引
        if self.kernel == 2:
            y0, x0, oy, ox = _neighbors_2x2(Yc, Xc)
        else:
            y0, x0, oy, ox = _neighbors_k(Yc, Xc, self.kernel)

        K = oy.numel()
        ys = (y0[..., None] + oy.view(1, 1, K)).clamp(0, H - 1)
        xs = (x0[..., None] + ox.view(1, 1, K)).clamp(0, W - 1)

        # 3) 几何基权重
        if self.kernel == 2:
            base_w = _bilinear_base_weights(Yc, Xc, y0, x0)       # [H_out,W_out,4]
        else:
            base_w = _gaussian_base_weights(Yc, Xc, ys, xs, sigma=self.gaussian_sigma)  # [H_out,W_out,K]
        base_w_b = (base_w.clamp_min(1e-12) ** self.beta)[None, ...]  # [1,H_out,W_out,K]

        # 4) 取邻域特征 & 锚点特征（几何插值）
        neigh_feats = []
        for k in range(K):
            fk = metric[:, :, ys[..., k], xs[..., k]]            # [B,C,H_out,W_out]
            neigh_feats.append(fk.permute(0, 2, 3, 1))           # [B,H_out,W_out,C]
        neigh_feats = torch.stack(neigh_feats, dim=3)             # [B,H_out,W_out,K,C]
        anchor_feat = (base_w_b[..., None] * neigh_feats).sum(dim=3)  # [B,H_out,W_out,C]

        # 5) 语义相似 + 位置正则
        sim = _similarity(neigh_feats, anchor_feat.unsqueeze(3).expand_as(neigh_feats),
                          distance=self.distance)                 # [B,H_out,W_out,K]
        if self.pos_lambda > 0 and pos is not None:
            # 用整数邻域与输出中心的几何距离作额外惩罚
            dist2 = (Yc[..., None] - ys.float()) ** 2 + (Xc[..., None] - xs.float()) ** 2  # [H_out,W_out,K]
            sim = sim - self.pos_lambda * dist2[None, ...]

        # 6) 语义权重
        if self.weighting == 'softmax':
            sem_w = torch.softmax(self.alpha * sim, dim=-1)       # [B,H_out,W_out,K]
        else:
            sem_w = torch.ones_like(sim) / float(K)

        # 7) 组合权重 & 门控 & 归一化
        w = sem_w * base_w_b                                      # [B,H_out,W_out,K]
        if self.gate_tau is not None:
            disp = sim.max(dim=-1).values - sim.mean(dim=-1)      # [B,H_out,W_out]
            gate = (disp >= self.gate_tau).float()[..., None]     # [B,H_out,W_out,1]
            pure_base = base_w_b.expand_as(w)
            w = gate * w + (1 - gate) * pure_base

        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)       # 归一化

        # 若仅想让规划不参与反传（省显存/更稳），在闭包内使用 detached 权重
        ys_b, xs_b = ys, xs
        def prune_fn(x: torch.Tensor,
                     mode: str = 'wmean',
                     ext_weight: torch.Tensor | None = None) -> torch.Tensor:
            assert x.dim() == 4 and x.shape[0] == B and x.shape[2:] == (H, W)
            Cx = x.shape[1]

            # 取邻域值
            neigh_vals = []
            for k in range(K):
                vk = x[:, :, ys_b[..., k], xs_b[..., k]]         # [B,C,H_out,W_out]
                neigh_vals.append(vk.permute(0, 2, 3, 1))        # [B,H_out,W_out,C]
            neigh_vals = torch.stack(neigh_vals, dim=3)           # [B,H_out,W_out,K,C]

            # 选择权重
            if mode == 'pick':
                # 非可导选择；只对选中的邻域分支对 x 回传梯度
                pick_idx = w.detach().argmax(dim=-1) if not self.requires_grad_plan else w.argmax(dim=-1)
                out = neigh_vals[torch.arange(B)[:,None,None],
                                  torch.arange(H_out)[None,:,None],
                                  torch.arange(W_out)[None,None,:],
                                  pick_idx]                        # [B,H_out,W_out,C]
            else:
                if mode in ('mean', 'sum'):
                    w_use = base_w_b.expand_as(w)                 # 仅几何
                    w_use = w_use if self.requires_grad_plan else w_use.detach()
                elif mode in ('wmean', 'wsum'):
                    w_use = w if self.requires_grad_plan else w.detach()
                else:
                    raise ValueError(f"未知 mode: {mode}")

                # 叠加外部权重（可导到 ext_weight）
                if ext_weight is not None:
                    assert ext_weight.shape == (B, 1, H, W)
                    ew = []
                    for k in range(K):
                        ewk = ext_weight[:, :, ys_b[..., k], xs_b[..., k]]  # [B,1,H_out,W_out]
                        ew.append(ewk.squeeze(1))
                    ew = torch.stack(ew, dim=-1)                              # [B,H_out,W_out,K]
                    w_use = w_use * ew

                if mode in ('mean', 'wmean'):
                    w_use = w_use / w_use.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                out = (w_use[..., None] * neigh_vals).sum(dim=3)   # [B,H_out,W_out,C]

            return out.permute(0, 3, 1, 2).contiguous()            # [B,C,H_out,W_out]

        prune_fn.info = {
            "H_in": H, "W_in": W, "H_out": H_out, "W_out": W_out,
            "kernel": self.kernel, "K": K,
            "distance": self.distance, "weighting": self.weighting,
            "alpha": self.alpha, "beta": self.beta,
            "pos_lambda": self.pos_lambda, "gaussian_sigma": self.gaussian_sigma,
            "gate_tau": self.gate_tau, "align_corners": self.align_corners,
            "requires_grad_plan": self.requires_grad_plan,
        }
        return prune_fn

class MPBDownPlanBCHW(nn.Module):
    """
    Manifold Projected Bilinear Downsampling (MPB-Down) 规划器（BCHW）。

    思路：
      1) 用 metric 做 bilinear 下采样，得到 teacher 特征 T (46x46)；
      2) 对每个输出位置，在相应 2x2 邻域中，从 4 个原生 token 里
         挑一个最接近 T 且几何合理的 token 作为输出；
      3) 输出 token 始终来自原生 patch embedding（不造新 token），
         分布接近 nearest，但结构上更对齐 bilinear。

    forward(metric, target_hw) -> prune_fn
    prune_fn(x, mode='pick'):
        - mode='pick'：MPB-Down（推荐，用于 training-free）
        - mode='nearest'：几何最近邻（2x2 block 中 base_w 最大的那个）
        - mode='bilinear'：标准 bilinear 插值（用 base_w 做加权平均）
    """
    def __init__(self,
                 distance: str = 'cosine',
                 alpha: float = 1.0,       # 语义相似度权重
                 gamma: float = 0.3,       # 几何 bias (base_logits) 权重
                 gate_tau: float | None = None,  # 语义不可靠时退化为纯几何
                 align_corners: bool = False,
                 requires_grad_plan: bool = False):
        super().__init__()
        self.distance = distance
        self.alpha = alpha
        self.gamma = gamma
        self.gate_tau = gate_tau
        self.align_corners = align_corners
        self.requires_grad_plan = requires_grad_plan

    def forward(self, metric: torch.Tensor, target_hw: tuple[int, int]):
        """
        metric: [B,C,H,W] 规划用特征（通常直接用待剪枝特征即可）
        target_hw: (H_out, W_out)，例如 (46,46)
        """
        assert metric.dim() == 4, "metric 必须是 [B,C,H,W]"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        device = metric.device

        # 1) teacher: metric 的 bilinear 下采样结果 [B,C,H_out,W_out]
        teacher = F.interpolate(
            metric,
            size=(H_out, W_out),
            mode='bilinear',
            align_corners=self.align_corners
        )
        teacher = teacher.permute(0, 2, 3, 1).unsqueeze(3)          # [B,H_out,W_out,1,C]

        # 2) 输出网格 & 2x2 邻域
        Yc, Xc = _make_frac_grid(H, W, H_out, W_out,
                                 self.align_corners, device=device)
        y0, x0, oy, ox = _neighbors_2x2(Yc, Xc)                      # 仅支持 2x2（K=4）

        K = oy.numel()  # 4
        ys = (y0[..., None] + oy.view(1, 1, K)).clamp(0, H - 1)     # [H_out,W_out,4]
        xs = (x0[..., None] + ox.view(1, 1, K)).clamp(0, W - 1)     # [H_out,W_out,4]

        # 将 (ys,xs) 编成一维索引用于 gather
        flat_idx = (ys * W + xs).view(-1)                            # [H_out*W_out*4]

        # 3) 几何基权重（bilinear）
        base_w = _bilinear_base_weights(Yc, Xc, y0, x0)              # [H_out,W_out,4]
        base_w = base_w.clamp_min(1e-12)
        base_logits = base_w.log()                                   # [H_out,W_out,4]

        # 4) 邻域特征（用 metric 计算相似度）
        metric_flat = metric.view(B, C, H * W)                       # [B,C,HW]
        neigh_feats = metric_flat[:, :, flat_idx]                    # [B,C,H_out*W_out*4]
        neigh_feats = neigh_feats.view(B, C, H_out, W_out, K).permute(0, 2, 3, 4, 1)
        # neigh_feats: [B,H_out,W_out,4,C]

        # 5) 语义相似度 (cosine / L2) + alpha
        sim = _similarity(
            neigh_feats,
            teacher.expand_as(neigh_feats),
            distance=self.distance
        )                                                            # [B,H_out,W_out,4]
        sim = self.alpha * sim

        # 6) gate: 语义不可靠时退化为纯几何
        gate = None
        if self.gate_tau is not None:
            disp = sim.max(dim=-1).values - sim.mean(dim=-1)         # [B,H_out,W_out]
            gate = (disp >= self.gate_tau).float()[..., None]        # [B,H_out,W_out,1]

        # 7) 组合 logits：semantic + geometric bias
        base_logits_b = base_logits.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H_out,W_out,4]
        logits = sim + self.gamma * base_logits_b                        # [B,H_out,W_out,4]

        if gate is not None:
            # gate=0 时只用几何（pure nearest/bilinear）
            logits = gate * logits + (1.0 - gate) * base_logits_b

        # 是否允许规划反传
        logits_for_pick = logits if self.requires_grad_plan else logits.detach()
        base_logits_for_geo = base_logits_b if self.requires_grad_plan else base_logits_b.detach()
        base_w_for_bilinear = base_w if self.requires_grad_plan else base_w.detach()

        H_in, W_in = H, W  # 关闭变量

        def prune_fn(x: torch.Tensor,
                     mode: str = 'pick',
                     ext_weight: torch.Tensor | None = None) -> torch.Tensor:
            """
            x: [B,C,H_in,W_in] 要被剪枝/重采样的特征（可与 metric 不同但 H,W 必须相同）

            mode:
              - 'pick'     : MPB-Down（推荐，用于 training-free）
              - 'nearest'  : 2x2 基于 base_w 的几何最近邻
              - 'bilinear' : 纯 bilinear 插值（用 base_w 作加权平均）
            """
            assert x.dim() == 4 and x.shape[0] == B \
                and x.shape[2:] == (H_in, W_in), \
                f"x 应为 [B,C,{H_in},{W_in}]"

            Bx, Cx, _, _ = x.shape
            x_flat = x.view(Bx, Cx, H_in * W_in)                    # [B,C,HW]
            neigh_vals = x_flat[:, :, flat_idx]                     # [B,C,H_out*W_out*4]
            neigh_vals = neigh_vals.view(Bx, Cx, H_out, W_out, K).permute(0, 2, 3, 4, 1)
            # neigh_vals: [B,H_out,W_out,4,Cx]

            if mode == 'pick':
                # MPB-Down: teacher + semantic + geometric
                pick_idx = logits_for_pick.argmax(dim=-1)           # [B,H_out,W_out]

                b_idx = torch.arange(Bx, device=x.device)[:, None, None]
                y_idx = torch.arange(H_out, device=x.device)[None, :, None]
                x_idx = torch.arange(W_out, device=x.device)[None, None, :]

                out = neigh_vals[b_idx, y_idx, x_idx, pick_idx]     # [B,H_out,W_out,Cx]
                return out.permute(0, 3, 1, 2).contiguous()         # [B,C,H_out,W_out]

            elif mode == 'nearest':
                # 纯几何：2x2 内 base_w 最大的那个（相当于 nearest）
                geo_idx = base_logits_for_geo.argmax(dim=-1)        # [B,H_out,W_out]

                b_idx = torch.arange(Bx, device=x.device)[:, None, None]
                y_idx = torch.arange(H_out, device=x.device)[None, :, None]
                x_idx = torch.arange(W_out, device=x.device)[None, None, :]

                out = neigh_vals[b_idx, y_idx, x_idx, geo_idx]      # [B,H_out,W_out,Cx]
                return out.permute(0, 3, 1, 2).contiguous()

            elif mode == 'bilinear':
                # 用 base_w 做加权平均（标准 bilinear）
                w = base_w_for_bilinear.unsqueeze(0)                # [1,H_out,W_out,4]
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                out = (w[..., None] * neigh_vals).sum(dim=3)        # [B,H_out,W_out,Cx]
                return out.permute(0, 3, 1, 2).contiguous()

            else:
                raise ValueError(f"未知 mode: {mode}")

        prune_fn.info = {
            "H_in": H_in, "W_in": W_in,
            "H_out": H_out, "W_out": W_out,
            "kernel": 2, "K": 4,
            "distance": self.distance,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "gate_tau": self.gate_tau,
            "align_corners": self.align_corners,
            "requires_grad_plan": self.requires_grad_plan,
            "type": "MPBDownPlanBCHW",
        }
        return prune_fn


if __name__ == "__main__":
    B, C, H, W = 1, 96, 56, 56
    x = torch.randn(B, C, H, W)
    print(x.shape)
    planner = FlexiToMePlanBCHW(
        kernel=2,  # 2x2 邻域（最快），也可换 3/5 做更平滑的聚合
        distance='cosine',
        weighting='softmax',  # 用语义softmax；想更稳可改 'mean'（退化到插值）
        alpha=1.0,  # 语义相似强度
        beta=1.0,  # 基权重强度（对齐插值）
        pos_lambda=0.0,  # 如需更邻近的局部性可 >0
        gate_tau=None,  # 相似度分散低则退化到基权重（插值）
    )

    prune_fn = planner(metric=x,target_hw=(54,54))
    out = prune_fn(x, mode='pick')
    print(out.shape)