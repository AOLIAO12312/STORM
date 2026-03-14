
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

Mamba demonstrates strong efficiency in modeling long visual sequences. However, when token reduction is applied to structurally enhanced Mamba variants, these models exhibit a severe performance collapse. We attribute this degradation to the spatially agnostic nature of existing reduction methods, which violate the two-dimensional structural premise required by the selective scanning mechanism. In this work, we propose STORM, a spatial-aware token reduction framework designed to maintain structural integrity throughout the compression process. STORM reformulates reduction into a structured operation on spatial units, enforcing localized constraints to maintain both grid topology and neighborhood coherence. As a plug-and-play module, STORM equips existing reduction pipelines with explicit spatial awareness without any training. Empirical results demonstrate that STORM achieves state-of-the-art pruning accuracy across diverse vision Mamba backbones under training-free settings. Notably, STORM delivers a substantial accuracy recovery on VMamba, outperforming prior methods by up to 63.3% in top-1 accuracy. Meanwhile, STORM incurs only a 1.0% accuracy drop on PlainMamba, achieving performance comparable to ViT.

</div>

---

[//]: # (TODO: Start with this)

### 🧠 Methodology

<div align="center">

![Main Framework Figure](assets/main_framework.png "STORM Framework") <!-- 将 main_framework.png 替换为您实际的图片文件名 -->
<p><em>Overview of the STORM framework.</em></p>

</div>

<div align="justify">

The core idea of STORM is to leverage spatial information within the feature maps to guide the token reduction process. This is achieved through a multi-step mechanism:

1.  **Spatial Prior Generation:** A lightweight module analyzes the spatial distribution of features to generate a saliency map or importance score for each spatial location.
2.  **Token Selection:** Based on the generated priors, less important tokens are identified and marked for reduction.
3.  **Efficient Processing:** The reduced set of tokens is then fed into the subsequent layers of the VSSM, significantly decreasing the computational load without substantial loss in performance.

For detailed algorithmic descriptions and ablation studies, please refer to Section 3 of our [paper](link-to-paper).

</div>

---

### 📊 Results

<div align="center">

| Model | Dataset | Accuracy (%) | Speed-up vs. Baseline |
| :---: | :-----: | :----------: | :-------------------: |
| Baseline VSSM | ImageNet-1K | 82.1 | - |
| STORM | ImageNet-1K | 81.9 | 2.3x |

<p><em>Performance comparison highlighting the efficiency gains of STORM.</em></p>

</div>

<div align="justify">

Our experiments demonstrate that STORM achieves significant speed-ups (e.g., 2.3x on ImageNet-1K) compared to standard VSSM implementations, with minimal degradation in accuracy. Qualitative results also show that the preserved tokens effectively capture key semantic information.

<!-- 在此处插入展示性能提升或可视化结果的图片 -->
![Performance Comparison](assets/performance_comparison.png "Speed vs. Accuracy")
![Visualization](assets/token_visualization.png "Selected Tokens Visualization")

</div>

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
