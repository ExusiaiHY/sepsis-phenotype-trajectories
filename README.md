# 基于自监督患者轨迹表征的 ICU 脓毒症动态亚型发现与跨中心泛化验证研究

## 项目简介

本项目面向 ICU 脓毒症患者，基于电子病历（EHR）时序数据，构建标准化患者轨迹数据处理流程，通过统计特征提取与自监督表征学习获得患者级别的特征表示，并结合多种聚类算法实现脓毒症动态亚型发现。系统支持跨数据集泛化验证接口，并提供完整的评估指标体系和可视化分析工具。

## 项目特色

- **研究导向**：不是简单的分类器，而是面向亚型发现的无监督/自监督分析框架
- **临床合理**：模拟数据基于真实 ICU 脓毒症表型（α/β/γ/δ），保留时序性、多变量性和缺失模式
- **模块化设计**：数据加载、预处理、特征工程、表征学习、聚类、评估、可视化完全解耦
- **可扩展**：MVP 使用统计特征+聚类，V2 可无缝升级为自监督 Transformer 编码器
- **跨中心验证就绪**：预留 MIMIC-IV 和 eICU 数据加载接口

## 目录结构

```
project/
├── config/
│   └── config.yaml          # 全局配置文件
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据
│   └── demo/                 # 演示数据
├── docs/
│   ├── 项目计划书.md
│   ├── 中期进展评估.md
│   ├── 软件效果评估.md
│   └── 软件使用说明书.md
├── outputs/
│   ├── figures/              # 可视化图形
│   ├── models/               # 保存的模型
│   └── reports/              # 评估报告
├── src/
│   ├── main.py               # 主入口程序
│   ├── data_loader.py        # 数据读取与模拟数据生成
│   ├── preprocess.py         # 预处理（缺失填充、异常值、标准化）
│   ├── feature_engineering.py # 统计特征提取
│   ├── representation_model.py # 表征学习（MVP: PCA / V2: 自监督）
│   ├── clustering.py         # 聚类与最优 K 搜索
│   ├── evaluation.py         # 评估（内部指标 + 生存分层 + 外部验证）
│   ├── visualization.py      # 可视化（散点图、热力图、K-M 曲线等）
│   └── utils.py              # 公共工具（日志、配置、种子、计时器）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 建议使用 Python 3.10+
pip install -r requirements.txt
```

### 2. 运行（使用模拟数据）

```bash
cd project/src
python main.py
```

### 3. 自定义参数

```bash
# 生成 1000 名患者，使用 GMM 聚类
python main.py --n-patients 1000 --method gmm

# 指定 K=4，使用 t-SNE 降维
python main.py --k 4 --reduction tsne

# 运行多方法聚类对比
python main.py --compare-methods
```

### 4. 查看结果

- 图形输出：`outputs/figures/`
- 评估报告：`outputs/reports/`
- 处理后数据：`data/processed/`

## 技术栈

| 层级 | 工具 |
|------|------|
| 数据处理 | pandas, numpy, scipy |
| 机器学习 | scikit-learn |
| 降维可视化 | umap-learn, matplotlib |
| 生存分析 | lifelines |
| 配置管理 | PyYAML |
| 深度学习（V2） | PyTorch |

## 评估指标

- **聚类质量**：轮廓系数、Calinski-Harabasz 指数、Davies-Bouldin 指数
- **外部验证**：ARI、NMI（有真实标签时）
- **生存分层**：Kaplan-Meier 曲线、log-rank 检验、各亚型死亡率
- **临床画像**：各亚型的人口学、器官功能、治疗模式对比

## 版本规划

- **MVP（当前）**：模拟数据 → 统计特征 → PCA → K-Means/GMM → 评估可视化
- **V2**：自监督 Transformer 预训练 → 结局约束聚类 → 跨中心验证
- **V3**：多器官交互图 → 图聚类 → SHAP 可解释性分析
