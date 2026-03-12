import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 工具函数（BCHW）
# =========================
def _pad_to_window_bchw(x: torch.Tensor, win: int):
    """
    x: [B, C, H, W]
    -> x_pad: [B, C, H', W']  (H',W' 为 win 的整数倍), 以及 (H', W', pad_h, pad_w)
    """
    B, C, H, W = x.shape
    pad_h = (win - H % win) % win
    pad_w = (win - W % win) % win
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    return x, (H, W, pad_h, pad_w)

def _unfold_windows_bchw(x: torch.Tensor, win: int):
    """
    x: [B, C, H, W]，H/W 已是 win 的倍数
    -> xw: [B, nH, nW, win, win, C], nH, nW
    """
    B, C, H, W = x.shape
    assert H % win == 0 and W % win == 0
    nH, nW = H // win, W // win
    xw = x.view(B, C, nH, win, nW, win).permute(0, 2, 4, 3, 5, 1).contiguous()
    return xw, nH, nW

def _unfold_pos_bchw(pos: torch.Tensor, win: int):
    """
    pos: [B, 2, H, W] -> [B, nH, nW, win, win, 2]
    """
    B, P, H, W = pos.shape
    assert P == 2
    nH, nW = H // win, W // win
    pw = pos.view(B, 2, nH, win, nW, win).permute(0, 2, 4, 3, 5, 1).contiguous()
    return pw

def _pairwise_sim_to_anchor(x_flat, anchor_flat, distance='cosine', eps=1e-6):
    """
    x_flat:      [B, nH, nW, N, C]
    anchor_flat: [B, nH, nW, 1, C]
    -> sim:      [B, nH, nW, N]
    """
    if distance == 'cosine':
        x_norm = x_flat / (x_flat.norm(dim=-1, keepdim=True).clamp_min(eps))
        a_norm = anchor_flat / (anchor_flat.norm(dim=-1, keepdim=True).clamp_min(eps))
        sim = (x_norm * a_norm).sum(dim=-1)
    elif distance == 'l2':
        sim = -((x_flat - anchor_flat) ** 2).sum(dim=-1)
    else:
        raise NotImplementedError(f"distance {distance} not supported")
    return sim

# =========================
# 规划-应用：语义感知 ToMe 下采样（BCHW）
# =========================
class GridToMePlanBCHW(nn.Module):
    """
    阶段1：forward(metric, pos) -> 返回一个 prune_fn（闭包），只描述如何剪枝/合并；
    阶段2：prune_fn(x, ...)    -> 对任意 [B,C,H,W] 张量执行同样的剪枝（token、残差、size 都可）。

    - 局部窗口大小 win×win（默认 2×2），窗口内按与“锚点”（左上角）相似度生成语义权重；
    - 输出分辨率为 (H//win, W//win)，几何上严格对齐，不破坏二维语义与卷积/SSM步进。

    参数：
      win: int，窗口大小（建议 2）
      distance: 'cosine' | 'l2'
      pos_lambda: float，位置正则权重 λ；>0 时需要传 pos=[B,2,H,W] 参与规划
      weighting: 'mean' | 'softmax'
      anchor_mode: 目前支持 'topleft'（左上角作为锚点）
    """
    def __init__(self, win=2, distance='cosine', pos_lambda=0.0, weighting='mean', anchor_mode='topleft'):
        super().__init__()
        self.win = win
        self.distance = distance
        self.pos_lambda = pos_lambda
        self.weighting = weighting
        self.anchor_mode = anchor_mode

    def forward(self, metric: torch.Tensor, pos: torch.Tensor = None):
        """
        metric: [B, C, H, W] —— 用于规划（不会被改动/剪枝）
        pos:    [B, 2, H, W] —— 可选，加入位置正则

        返回：
          prune_fn: 可调用函数，签名如下：
            prune_fn(x: torch.Tensor,
                     mode: str = 'wmean',
                     ext_weight: torch.Tensor | None = None) -> torch.Tensor

            - x:         [B, C, H, W]，可为 token、residual、size 等
            - mode:
                'pick'  : 仅取每窗锚点，纯剪枝（不做合并）
                'mean'  : 每窗均值
                'sum'   : 每窗求和
                'wmean' : 用规划时的语义权重做加权平均（默认）
                'wsum'  : 用语义权重做加权求和
            - ext_weight: [B, 1, H, W]，可选的外部权重（如 size/置信度），将乘到聚合权重上
            - 返回：剪枝后的张量 [B, C, H//win, W//win]
        """
        assert metric.dim() == 4, "metric 必须是 [B, C, H, W]"
        B, C, H0, W0 = metric.shape
        win = self.win

        # ---- 规划阶段：只计算窗口划分 & 语义权重，不动实际数据 ----
        metric_pad, (H, W, pad_h, pad_w) = _pad_to_window_bchw(metric, win)
        mw, nH, nW = _unfold_windows_bchw(metric_pad, win)       # [B, nH, nW, win, win, C]
        N = win * win

        # 锚点选择（当前固定为左上角）
        if self.anchor_mode != 'topleft':
            raise NotImplementedError("当前仅支持 anchor_mode='topleft'")
        anchor_feat = mw[..., 0, 0, :]                           # [B, nH, nW, C]

        # 相似度（到锚点）
        m_flat = mw.view(B, nH, nW, N, C)                        # [B, nH, nW, N, C]
        anchor_rep = anchor_feat.unsqueeze(3)                    # [B, nH, nW, 1, C]
        sim = _pairwise_sim_to_anchor(m_flat, anchor_rep, distance=self.distance)  # [B, nH, nW, N]

        # 位置正则
        if self.pos_lambda > 0 and pos is not None:
            pos_pad, _ = _pad_to_window_bchw(pos, win) if pos.shape[-2:] != (H, W) else (pos, (H, W, 0, 0))
            pw = _unfold_pos_bchw(pos_pad, win)                  # [B, nH, nW, win, win, 2]
            pos_flat = pw.view(B, nH, nW, N, 2)                  # [B, nH, nW, N, 2]
            anchor_pos = pw[..., 0, 0, :].unsqueeze(3)           # [B, nH, nW, 1, 2]
            dist2 = ((pos_flat - anchor_pos) ** 2).sum(dim=-1)   # [B, nH, nW, N]
            sim = sim - self.pos_lambda * dist2

        # 规划的基础权重（与 metric/pos 绑定）
        if self.weighting == 'softmax':
            plan_w = torch.softmax(sim, dim=-1)                  # [B, nH, nW, N]
        else:
            plan_w = torch.ones_like(sim)                        # 均匀权重（可与 ext_weight 结合）

        # 把规划信息放入闭包，返回“剪枝函数”
        def prune_fn(x: torch.Tensor,
                     mode: str = 'wmean',
                     ext_weight: torch.Tensor | None = None) -> torch.Tensor:
            """
            对任意 [B,C,H,W] 张量执行与规划一致的剪枝/合并。
            """
            assert x.dim() == 4, "x 必须是 [B, C, H, W]"
            Bx, Cx, Hx, Wx = x.shape
            # 形状必须与规划阶段一致（除 C 外）
            if (Hx, Wx) != (H0, W0):
                raise ValueError(f"x 的空间尺寸 {(Hx, Wx)} 与规划时 {(H0, W0)} 不一致")

            # 1) padding & unfold
            x_pad, _ = _pad_to_window_bchw(x, win)               # 到 [B, C, H, W]
            xw, nH2, nW2 = _unfold_windows_bchw(x_pad, win)      # [B, nH, nW, win, win, C]
            assert nH2 == nH and nW2 == nW
            x_flat = xw.view(Bx, nH, nW, N, Cx)                  # [B, nH, nW, N, C]

            # 2) 权重组合
            if mode == 'pick':
                # 只取锚点，纯剪枝（最近邻）
                out = xw[..., 0, 0, :]                           # [B, nH, nW, C]
                out = out.permute(0, 3, 1, 2).contiguous()       # [B, C, nH, nW]
                return out

            # 基础权重：由规划决定（plan_w）或均匀
            if mode in ('wmean', 'wsum'):
                w = plan_w                                       # [B, nH, nW, N]
            elif mode in ('mean', 'sum'):
                w = torch.ones_like(plan_w)
            else:
                raise ValueError(f"未知 mode: {mode}")

            # 叠加外部权重（如 size/置信度），保持广播一致
            if ext_weight is not None:
                assert ext_weight.shape[0] == Bx and ext_weight.shape[2:] == (H0, W0) and ext_weight.shape[1] == 1, \
                    "ext_weight 需要是 [B,1,H,W] 且 H,W 与输入一致"
                ew_pad, _ = _pad_to_window_bchw(ext_weight, win)
                ew = ew_pad.view(Bx, 1, nH, win, nW, win).permute(0, 2, 4, 3, 5, 1).contiguous()  # [B,nH,nW,win,win,1]
                ew = ew.view(Bx, nH, nW, N)                                                          # [B,nH,nW,N]
                w = w * ew

            # 3) 聚合（num/den）
            num = (w.unsqueeze(-1) * x_flat).sum(dim=3)          # [B, nH, nW, C]
            if mode in ('wmean', 'mean'):
                den = w.sum(dim=3, keepdim=True).clamp_min(1e-6) # [B, nH, nW, 1]
                out = num / den
            elif mode in ('wsum', 'sum'):
                out = num
            else:
                raise AssertionError("mode 分支已覆盖")

            # 4) 回到 [B, C, nH, nW]
            out = out.permute(0, 3, 1, 2).contiguous()
            return out

        # 为调试/日志附加一些信息
        prune_fn.info = {
            "win": win,
            "H_in": H0, "W_in": W0,
            "H_pad": H, "W_pad": W,
            "pad_h": pad_h, "pad_w": pad_w,
            "nH": nH, "nW": nW,
            "distance": self.distance,
            "pos_lambda": self.pos_lambda,
            "weighting": self.weighting,
            "anchor_mode": self.anchor_mode,
        }

        return prune_fn

if __name__ == '__main__':
    B, C, H, W = 1, 96, 56, 56
    x = torch.randn(B, C, H, W)
    planner = GridToMePlanBCHW(win=2, distance='cosine', pos_lambda=0.1, weighting='softmax')

    # 1) 用 token（或某层特征）做“规划”
    prune = planner(metric=x)  # 返回函数，不做实际剪枝

    # 2) 将同一个 prune 函数应用到 token 分支
    x_pruned = prune(x, mode='wmean')  # [B, C, H/2, W/2]

    print(x_pruned.shape)