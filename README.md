
<div align="center">
<h1>STORM </h1>
<h3>STORM: Spatial-Aware Token Reduction Framework for VSSM</h3>

[//]: # ()
[//]: # ([Yue Liu]&#40;https://github.com/MzeroMiko&#41;<sup>1</sup>,[Yunjie Tian]&#40;https://sunsmarterjie.github.io/&#41;<sup>1</sup>,[Yuzhong Zhao]&#40;https://scholar.google.com.hk/citations?user=tStQNm4AAAAJ&hl=zh-CN&oi=ao&#41;<sup>1</sup>, [Hongtian Yu]&#40;https://github.com/yuhongtian17&#41;<sup>1</sup>, [Lingxi Xie]&#40;https://scholar.google.com.hk/citations?user=EEMm7hwAAAAJ&hl=zh-CN&oi=ao&#41;<sup>2</sup>, [Yaowei Wang]&#40;https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN&oi=ao&#41;<sup>3</sup>, [Qixiang Ye]&#40;https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao&#41;<sup>1</sup>, [Yunfan Liu]&#40;https://scholar.google.com.hk/citations?user=YPL33G0AAAAJ&hl=zh-CN&oi=ao&#41;<sup>1</sup>)

[//]: # ()
[//]: # (<sup>1</sup>  University of Chinese Academy of Sciences, <sup>2</sup>  HUAWEI Inc.,  <sup>3</sup> PengCheng Lab.)

**Paper:** ([arXiv]()) 
<br>
**Project Page:** [https://spatial-aware-reduction-framework.github.io/](https://spatial-aware-reduction-framework.github.io/)

</div>

---

### 📋 Table of Contents
- [Abstract](#abstract)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage) <!-- 如果包含代码，这是标准部分 -->
- [Citation](#citation)

---

### 🔍 Abstract

<div align="justify">

Mamba demonstrates strong efficiency in modeling long visual sequences. However, when token reduction is applied to structurally enhanced Mamba variants, these models exhibit a severe performance collapse. We attribute this degradation to the spatially agnostic nature of existing reduction methods, which violate the two-dimensional structural premise required by the selective scanning mechanism. 

In this work, we propose STORM, a spatial-aware token reduction framework designed to maintain structural integrity throughout the compression process. STORM reformulates reduction into a structured operation on spatial units, enforcing localized constraints to maintain both grid topology and neighborhood coherence. As a plug-and-play module, STORM equips existing reduction pipelines with explicit spatial awareness without any training. Empirical results demonstrate that STORM achieves state-of-the-art pruning accuracy across diverse vision Mamba backbones under training-free settings. Notably, STORM delivers a substantial accuracy recovery on VMamba, outperforming prior methods by up to 63.3% in top-1 accuracy. Meanwhile, STORM incurs only a 1.0% accuracy drop on PlainMamba, achieving performance comparable to ViT.

</div>

---

### 🧠 Methodology

<div align="center">

![Main Framework Figure](assets/main_framework.png "STORM Framework")
<p><em>Overview of STORM. The framework performs spatially structured token reduction in two decoupled stages: row-wise and then column-wise reduction within localized windows, preserving the 2D grid layout required for selective scanning.</em></p>

</div>

<div align="justify">

The STORM framework proposes a lightweight solution that refactors token reduction into a spatially structured process, as illustrated in Figure 3 above. The framework comprises three core features:

1.  **Dimensional Decoupling:** Instead of globally flattening tokens, STORM refactors the reduction process into two successive stages—row-wise and column-wise. This preserves a regular 2D grid topology, ensuring seamless compatibility with the 2D Selective Scan (SS2D) mechanism in Mamba.
2.  **Localized Window:** To prevent semantic distortion from long-range interference, STORM partitions the feature map into non-overlapping local windows. Reduction operations are strictly confined within these contiguous neighborhoods to protect fine-grained details and local coherence.
3.  **Faithful Scanning Restoration:** By maintaining a structured layout, STORM ensures that the causal propagation paths of the four-way scanning are not disrupted. This allows the model to retain accurate spatial awareness and performance during inference without requiring any re-training.

For detailed algorithmic descriptions and ablation studies, please refer to Section 3 of our [paper](link-to-paper).

</div>

---

### 📊 Classification on ImageNet-1K

<div align="center">

| Method              | GFlops   | Params (M) | Acc1 (%) | Δ (%)    |
|---------------------|----------|------------|----------|----------|
| **VMamba-B (Base)** |          |            |          |          |
| VMamba-B            | 15.36    | 89         | 83.9     | -        |
| +EViT               | 9.33     | 89         | 24.4     | 59.5↓    |
| +ToMe               | 9.69     | 89         | 35.7     | 48.2↓    |
| **+STORM (EViT)**   | **9.33** | **89**     | **82.2** | **1.7↓** |
| **+STORM (ToMe)**   | **9.33** | **89**     | **82.6** | **1.3↓** |
| **PlainMamba-L3**   |          |            |          |          |
| PlainMamba-L3       | 14.44    | 51         | 82.2     | -        |
| +EViT               | 9.74     | 51         | 75.2     | 7.0↓     |
| +ToMe               | 9.75     | 51         | 76.1     | 6.1↓     |
| **+STORM (EViT)**   | **9.33** | **51**     | **82.2** | **1.7↓** |
| **+STORM (ToMe)**   | **9.74** | **51**     | **80.9** | **1.3↓** |
| **LocalMamba-S**    |          |            |          |          |
| LocalMamba-S        | 11.37    | 50         | 83.7     | -        |
| +EViT               | 6.70     | 50         | 19.3     | 64.4↓    |
| +ToMe               | 6.96     | 50         | 28.7     | 55.0↓    |
| **+STORM (EViT)**   | **6.70** | **50**     | **78.5** | **5.2↓** |
| **+STORM (ToMe)**   | **6.70** | **50**      | **79.6** | **4.1↓** |
<p><em>Performance comparison highlighting the efficiency gains of STORM on ImageNet-1K.</em></p>

</div>

<div align="justify">


![Performance Comparison: Accuracy vs. Reduction Ratio for EViT and STORM (EViT)](assets/EViT.png "Reduction Ratio vs. Accuracy & Throughput")

**Figure 1:** Accuracy and throughput comparison between EViT and STORM (EViT) under different reduction ratios. STORM maintains high accuracy while significantly improving inference speed.

![Performance Comparison: Accuracy vs. Reduction Ratio for ToMe and STORM (ToMe)](assets/ToMe.png "Speed vs. Accuracy & Throughput")

**Figure 2:** Accuracy and throughput comparison between ToMe and STORM (ToMe) across varying reduction ratios. STORM achieves better performance retention with higher throughput.

</div>

---
### Visualization



![Visualization](assets/Visualization.png "Token Reduciton Visualization")

**Figure 3:** Visualization of token reduction at varying compression ratios. ToMe produces fragmented and spatially inconsistent representations. Structured spatial reduction (without windowing) restores layout regularity but sacrifices fine-grained details. In contrast, the full STORM framework consistently preserves both structural integrity and semantic coherence across all pruning levels.

![Visualization](assets/Semantic_Grouping.png "Semantic Grouping Visualization(95% Pruned)")

**Figure 3:** Even with a 95% reduction ratio, STORM avoids the "checkerboard noise" of conventional methods. By constraining reduction within local windows and preserving grid topology, patches are grouped into semantically coherent objects, which is vital for the SS2D mechanism to maintain faithful inference.

---

### 🚀 Usage

<!-- 如果您的项目包含代码，请在此处详细说明如何安装依赖、准备数据集、运行训练或推理脚本。 -->

#### Environment Setup

```bash
# Example commands to set up the environment
conda create -n storm python=3.9
conda activate storm
pip install torch torchvision
# ... add other dependencies
```

#### Training & Evaluation

```bash
# Example command to run training
python train.py --config configs/storm_config.yaml

# Example command to run evaluation
python eval.py --model_path path/to/your/model.pth --data_dir /path/to/dataset
```

---

### 📝 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{liu2025storm,
  title={STORM: Spatial-Aware Token Reduction Framework for VSSM},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
