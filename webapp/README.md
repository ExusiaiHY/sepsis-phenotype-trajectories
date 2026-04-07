# ICU Sepsis Phenotype Visualization Dashboard

ICU 脓毒症表型可视化端口 - 基于 Web 的交互式数据探索工具

## Features 功能特性

### 📊 Overview 概览
- 关键指标卡片（总患者数、稳定/转换患者比例）
- 各表型死亡率柱状图
- 方法比较雷达图
- 表型定义说明

### 🔬 Method Comparison 方法比较
- PCA vs S1 (Masked) vs S1.5 (Contrastive) 对比
- Silhouette Score 和 Mortality Range 柱状图
- 详细对比表格（K=2 和 K=4）

### 📈 Trajectories 轨迹分析
- 滚动窗口表型占比堆叠图
- 最常见的轨迹模式分布
- Top 15 轨迹模式展示

### 🔄 Transitions 转换分析
- 转换概率热力图（Plotly）
- Top 非自我转换路径
- 转换统计（总事件数、自我/非自我转换比例）

### 🌌 Spatio-Temporal 时空可视化
- **Embedding Space Projection**: 128维嵌入空间的2D PCA投影（2500个时空点）
- **Embedding Movement**: 患者在连续窗口间的嵌入位移统计
- **Embedding Norm by Phenotype**: 各表型在表示空间中的模长分布
- **Individual Trajectory**: 单个患者的3D轨迹可视化（窗口间的嵌入变化）
- **Window Evolution Stats**: 时间维度上的嵌入演化统计

### 👤 Patient Browser 患者浏览器
- 按轨迹类型筛选（稳定/单次转换/多次转换）
- 患者轨迹可视化
- 各表型停留时间统计

## Quick Start 快速开始

### 1. 安装依赖

```bash
cd /Users/exusiaihy/Desktop/Python高阶程序设计/project
source .venv/bin/activate
pip install flask flask-cors
```

### 2. 启动服务器

```bash
python webapp/app.py
```

或使用启动脚本：

```bash
./webapp/launch.sh
```

### 3. 访问应用

浏览器打开：http://localhost:5050

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | 主页面 |
| `GET /api/summary` | 摘要统计 |
| `GET /api/comparison` | 方法比较数据 |
| `GET /api/trajectory/stats` | 轨迹统计 |
| `GET /api/trajectory/transitions` | 转换矩阵 |
| `GET /api/patients/sample` | 患者样本 |
| `GET /api/patient/<id>` | 单个患者详情 |
| `GET /api/spatiotemporal/projection` | 嵌入空间2D投影（PCA） |
| `GET /api/spatiotemporal/window-evolution` | 窗口演化统计 |
| `GET /api/spatiotemporal/embedding-trajectory/<id>` | 患者嵌入轨迹 |

## Tech Stack 技术栈

- **Backend**: Python Flask + scikit-learn (PCA)
- **Frontend**: HTML5 + Tailwind CSS + Vanilla JavaScript
- **Charts**: Chart.js + Plotly.js (2D/3D)
- **Data**: 直接使用项目的 NumPy/JSON 数据文件

## Data Sources 数据源

- `data/s15/comparison_report.json` - 方法比较
- `data/s2/trajectory_stats.json` - 轨迹统计
- `data/s2/transition_matrix.json` - 转换矩阵
- `data/s2/window_labels.npy` - 患者窗口标签
- `data/s15/embeddings_s15.npy` - S1.5 嵌入（采样）

## Screenshots 截图预览

### 概览页
- 顶部四个指标卡片展示关键统计
- 死亡率柱状图展示 4 个表型的临床分离度
- 方法比较雷达图对比不同表示学习方法

### 轨迹页
- 桑基图风格的表型转换流
- 时间窗口上的表型占比变化
- 常见轨迹模式列表

### 患者浏览器
- 下拉选择不同轨迹类型的患者
- 可视化展示患者在 5 个时间窗口的表型变化
- 各表型停留时间统计

### 🌌 Spatio-Temporal 时空可视化
- **嵌入空间投影**: 500患者×5窗口=2500个时空点的PCA投影，颜色区分表型
- **嵌入位移**: 展示患者在连续窗口间在128维空间中的移动距离
- **表型模长**: 不同表型在嵌入空间中的"强度"分布
- **3D个体轨迹**: 单个患者的5个窗口形成的三维轨迹曲线
- **演化统计表格**: 量化的窗口间变化指标

## Development 开发

### 添加新的可视化

1. 在 `app.py` 中添加新的 API endpoint
2. 在 `index.html` 中添加图表容器
3. 在 `dashboard.js` 中添加图表初始化函数

### 修改样式

- 主样式使用 Tailwind CSS（CDN 引入）
- 自定义样式在 `<style>` 标签中

## License

同主项目
