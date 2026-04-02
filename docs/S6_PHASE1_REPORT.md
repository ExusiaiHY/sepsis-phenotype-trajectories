# S6 Phase 1 报告：增强 MIMIC-IV 数据提取 + 代理亚型标签 + 现有管道融合

**Date:** 2026-04-03  
**Cohort:** MIMIC-IV 3.1 (94,458 ICU stays, 41,295 Sepsis-3)  
**Output Directory:** `data/processed_mimic_enhanced/`

---

## 1. 执行内容概述

本阶段完成了从「单一死亡率预测」向「病因亚型诊断 + 治疗推荐」转型的**数据基础与标签工程**：

1. **扩展 MIMIC-IV 数据提取管道**
   - 在原有 31 个时序特征基础上，新增 12 个临床维度，总计 **43 个时序特征**。
   - 新增数据包括：血细胞分类（淋巴细胞等）、炎症标志物（CRP）、肝酶（ALT/AST/胆红素）、微生物培养结果、机械通气状态、以及从 raw labevents 提取的铁蛋白、D-二聚体、纤维蛋白原。

2. **构建代理亚型标签（Proxy Subtype Labels）**
   - 基于现有可观测指标，为每位患者生成三类代理标签：
     - **免疫内型**（immune_subtype）: MAS-like / EIL-like / Unclassified
     - **器官主导型**（organ_subtype）: alpha-like / beta-like / gamma-like / delta-like / Unclassified
     - **液体获益分型**（fluid_benefit_proxy）: high_benefit / low_benefit / Unclassified

3. **与原有 S1.5/S5 管道融合验证**
   - 增强数据成功通过 `src/data_loader.py` 加载。
   - 预处理和特征提取模块（`preprocess_pipeline`、`extract_features`）在 5,000 患者子集上验证通过。
   - 现有 `src/main.py` 可通过 `--processed-dir data/processed_mimic_enhanced` 直接读取增强数据。

---

## 2. 生成文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `patient_static_enhanced.parquet` | ~8.6 MB | 增强版患者静态信息表（55 列） |
| `patient_timeseries_enhanced.parquet` | ~97 MB | 增强版 48h 时序数据表（47 列） |
| `patient_static_with_subtypes.parquet` | ~9 MB | 含代理亚型标签的静态表（65 列） |
| `time_series_enhanced.npy` | ~778 MB | 3D 张量 `(94,458, 48, 43)`，可直接输入 S1.5 编码器 |
| `patient_info_enhanced.csv` | ~12 MB | 与 3D 张量对齐的 patient_info DataFrame |
| `enhanced_manifest.json` | ~2 KB | 特征名清单与元数据 |

---

## 3. 新增时序特征列表（12 项）

| 特征类别 | 具体特征 | 临床意义 | 缺失率* |
|---------|---------|---------|--------|
| 血细胞分类 | `lymphocytes_abs`, `lymphocytes_pct`, `monocytes_abs`, `neutrophils_abs` | 免疫状态、EIL 判断 | ~92.6% |
| 炎症标志物 | `crp` | 炎症强度 | ~99.2% |
| 肝酶/胆红素 | `alt`, `ast`, `bilirubin_total` | 肝损伤、MAS 判断 | ~82% |
| 凝血/铁代谢 | `ferritin`, `ddimer`, `fibrinogen` | 高凝状态、MAS 判断 | ~93-99.5% |
| 器官支持 | `mech_vent` | 机械通气状态 | ~0% |

*缺失率反映的是 48h 窗口内「小时级插值」后的缺失比例；这些实验室指标本身检测频率较低（每 4-24h 一次），高缺失率是预期现象。

---

## 4. 代理亚型标签分布与临床分层

### 4.1 免疫内型（immune_subtype）

| 亚型 | 人数 | 占比 | 28 天死亡率 | 特征摘要 |
|------|------|------|------------|---------|
| **Unclassified** | 92,116 | 97.5% | 39.8% | 不符合 MAS/EIL 的明确代理标准 |
| **MAS-like** | 2,165 | 2.3% | **56.3%** | 高 CRP/铁蛋白 + 肝损伤/凝血障碍 |
| **EIL-like** | 177 | 0.2% | **54.8%** | 淋巴细胞减少 + 培养阳性 + 炎症不极端 |

**观察**：MAS-like 和 EIL-like 虽然占比小，但死亡率显著高于普通未分类组（+16.5pp 和 +15.0pp），提示代理标签成功捕捉到了高风险的免疫极端表型。

### 4.2 器官主导型（organ_subtype）

| 亚型 | 人数 | 占比 | 28 天死亡率 | Seymour 对应 |
|------|------|------|------------|-------------|
| **Unclassified** | 60,148 | 63.7% | 30.4% | 轻症或器官受累不明确 |
| **alpha-like** | 15,299 | 16.2% | **58.0%** | 肝肾主导型 |
| **beta-like** | 8,611 | 9.1% | **47.2%** | 心肾主导型 |
| **delta-like** | 7,777 | 8.2% | **69.4%** | 多器官衰竭型 |
| **gamma-like** | 2,623 | 2.8% | **52.5%** | 呼吸主导型 |

**观察**：死亡率排序 `delta (69.4%) > alpha (58.0%) > gamma (52.5%) > beta (47.2%) > Unclassified (30.4%)`，与 Seymour 2019 文献中的严重程度梯度方向一致。

### 4.3 液体获益分型（fluid_benefit_proxy）

| 亚型 | 人数 | 占比 | 28 天死亡率 |
|------|------|------|------------|
| **Unclassified** | 78,691 | 83.3% | 37.5% |
| **high_benefit** | 9,517 | 10.1% | 39.7% |
| **low_benefit** | 6,250 | 6.6% | **74.2%** |

**观察**：`low_benefit` 组（心肾双重障碍或严重肺损伤）死亡率极高（74.2%），与「限制液体策略可能获益」的临床假设方向一致。

### 4.4 免疫 × 器官交叉表（死亡率）

| immune_subtype | Unclassified | alpha-like | beta-like | delta-like | gamma-like |
|----------------|-------------|------------|-----------|------------|------------|
| EIL-like       | 0.505       | 0.565      | 0.400     | **0.857**  | 0.333      |
| MAS-like       | 0.430       | 0.533      | 0.603     | 0.674      | 0.591      |
| Unclassified   | 0.302       | 0.582      | 0.469     | 0.695      | 0.524      |

**关键发现**：`EIL-like + delta-like` 的交叉死亡率最高（85.7%），提示免疫抑制叠加多器官衰竭是极端预后不良的组合。

---

## 5. 与现有管道融合验证

### 5.1 数据加载层（data_loader.py）
- 已修改 `load_mimic_data()`，**优先识别增强布局**（`time_series_enhanced.npy` + `patient_info_enhanced.csv`）。
- 若增强文件不存在，自动回退到旧版 `patient_static.parquet` + `patient_timeseries.parquet`。
- 新增亚型标签列（`immune_subtype`、`organ_subtype` 等）在加载时自动保留。

### 5.2 特征工程层
- 在 5,000 患者子集上验证：
  - `preprocess_pipeline` 成功处理 43 维时序数据（缺失率 50.7%，经前向填充+均值填充后降为 0%）。
  - `extract_features` 成功提取 **776 维统计特征**（含子窗口 12h/24h/48h 的 mean/std/min/max/trend/last）。

### 5.3 统一入口命令
```bash
python src/main.py \
  --source mimic \
  --processed-dir data/processed_mimic_enhanced \
  --tag mimic_enhanced \
  --skip-vis \
  --k 4
```

---

## 6. 局限性与诚实声明

1. **代理标签非金标准**
   - 我们没有真实的 mHLA-DR、IL-6、IL-1、TNF-α、PCT 等免疫标志物。
   - MAS/EIL 的判断是基于血小板、肝酶、CRP、铁蛋白、淋巴细胞等**可观测指标的近似规则**。
   - 这些标签的临床验证需要前瞻性队列或流式细胞术数据，目前仅用于模型训练和概念验证。

2. **高缺失率的增强特征**
   - 铁蛋白、D-二聚体、CRP 的小时级缺失率高达 93-99%，意味着时序模型需要在**高度稀疏**的数据中学习。
   - 这既是挑战也是机会：S1.5 的 mask-aware 编码器天然适合处理这种缺失模式。

3. **无影像/文本数据**
   - 当前数据库未导入 MIMIC-IV Note 模块，缺少放射学报告文本。
   - 肺部影像特征（如「肺炎」「ARDS」「肺水肿」）暂时无法纳入。

---

## 7. 下一步建议（Phase 2）

### 7.1 训练多任务亚型诊断模型（优先级：高）
改造现有 S5 实时学生架构，从单一二分类死亡率模型扩展为**多任务模型**：
- 共享编码器：复用 S1.5 的 mask-aware Transformer（输入维度扩展为 43）。
- 多任务输出头：
  - `免疫头`：3-class softmax（MAS-like / EIL-like / Unclassified）
  - `器官头`：5-class softmax（alpha / beta / gamma / delta / Unclassified）
  - `液体头`：3-class softmax（high / low / Unclassified）
  - `死亡头`：1-class sigmoid（保留原有死亡率预测能力）

### 7.2 构建治疗推荐引擎（优先级：高）
基于用户提供的最新文献，建立结构化治疗规则库：
- 输入：模型输出的亚型概率组合 + 患者当前关键指标。
- 输出：证据级别的治疗推荐（如 MAS-like → 考虑 rhIFN-γ；low_benefit → 限制性液体策略）。
- 技术路线：优先使用结构化 JSON 规则引擎保证 100% 可解释性；上层可用 LLM/RAG 生成自然语言总结。

### 7.3 可解释性验证（优先级：中）
- 利用 S1.5 的注意力权重，分析模型在判断 MAS-like 时最关注哪些时间步和哪些特征。
- 对比「模型注意力」与「临床规则中的关键指标」（如 ferritin、lymphocytes）的一致性。

---

## 8. 核心代码变更清单

| 文件 | 变更 |
|------|------|
| `src/build_enhanced_analysis_table.py` | **新增**：扩展 DuckDB SQL，提取微生物、血细胞分类、炎症、肝酶、铁蛋白/D-二聚体 |
| `src/subtype_label_engine.py` | **新增**：基于可观测指标生成 MAS/EIL/器官型/液体获益代理标签 |
| `scripts/s6_build_enhanced_mimic.py` | **新增**：端到端脚本，一键完成提取 → 标签 → 3D 张量 |
| `src/data_loader.py` | **修改**：`load_mimic_data()` 优先加载增强布局；`get_feature_names()` 读取增强 manifest |

---

## 9. 一句话总结

> **第一阶段已完成：我们在 94,458 例 MIMIC-IV 患者上构建了包含 43 维时序特征和 3 类代理亚型标签的增强数据集，并成功验证其与现有 S1.5/S5 数据管道的兼容性。代理标签展现出清晰的死亡率分层（delta-like 69.4% vs. Unclassified 30.4%），为下一步训练「多任务病因诊断模型」奠定了数据基础。**
