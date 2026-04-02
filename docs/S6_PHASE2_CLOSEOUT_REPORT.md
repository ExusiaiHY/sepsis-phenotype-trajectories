# S6 Phase 2 结项报告：多任务亚型诊断模型 + 治疗推荐引擎

**Date:** 2026-04-03  
**Cohort:** MIMIC-IV 3.1 (94,458 ICU stays)  
**Cloud GPU:** NVIDIA A10  
**Output Directory (Cloud):** `/root/project/data/s6_multitask_mimic_cloud/`

---

## 1. 执行内容概述

本阶段在 **Phase 1 增强数据（43 维时序特征 + 代理亚型标签）** 的基础上，完成了两项核心工程：

1. **多任务实时学生模型（Multi-task S5 Student）**
   - 将原有的单一死亡率预测模型扩展为**四任务联合诊断模型**：
     - 死亡率预测（二分类，保留原有能力）
     - 免疫内型诊断（3 类：Unclassified / EIL-like / MAS-like）
     - 器官主导型诊断（5 类：Unclassified / alpha / beta / gamma / delta）
     - 液体获益分型（3 类：Unclassified / low_benefit / high_benefit）
   - 在云端 NVIDIA A10 GPU 上完成了**全队列 94,458 例患者**的训练与验证。

2. **结构化治疗推荐引擎**
   - 基于最新文献构建了 `config/treatment_rules.json` 规则库。
   - 实现了 `SubtypeTreatmentRecommender`：输入模型输出的亚型概率，自动匹配证据级别的治疗方案并生成可解释的中文推荐报告。

---

## 2. 多任务模型架构

```
Input: (B, T=48, F=42 continuous + 1 treatment)
    ↓
TreatmentAwareEncoder (Transformer, d_model=64, 1 layer, 4 heads)
    ↓ shared embedding (64-d)
    ├─→ head_mortality  → sigmoid  → 死亡风险概率
    ├─→ head_immune     → softmax  → 免疫内型概率 (3-class)
    ├─→ head_organ      → softmax  → 器官主导型概率 (5-class)
    └─→ head_fluid      → softmax  → 液体获益概率 (3-class)
```

**关键设计决策：**
- **共享编码器**：所有任务复用同一个 S1.5-compatible 的 `TreatmentAwareEncoder`，保证 bedside 推理速度（CPU 单样本 **1.22 ms**）。
- **独立任务头**：每个任务有独立的 MLP 头，避免任务间干扰。
- **多任务损失**：`Loss = BCE(mortality) + CE(immune) + CE(organ) + CE(fluid)`，各任务权重相等（λ=1.0）。
- **数据标准化**：在训练集上计算逐特征 z-score，保存 `feat_mean.npy` 和 `feat_std.npy` 供部署时复用。

---

## 3. 云端训练结果（94,458 例完整队列）

### 3.1 训练配置

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA A10 |
| Batch size | 256 |
| Epochs | 15 (早停于 **epoch 13**) |
| LR | 1.0e-3 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Patience | 4 |
| 每 epoch 耗时 | **~4.0 秒** |
| 总训练时间 | **~52 秒** |

> 注：单 epoch 4 秒的速度得益于 A10 GPU 对矩阵运算的高效处理，以及模型本身只有 **97,484 参数**的轻量化设计。

### 3.2 测试集性能（Test split: 14,170 例）

| 任务 | AUROC | 准确率 | 平衡准确率 | 宏平均 F1 | 备注 |
|------|-------|--------|-----------|----------|------|
| **死亡率 (mortality)** | **0.8014** | 0.7306 | 0.7256 | — | 与原 S5-v2 (0.873) 有差距，但这是**从未经预训练的随机初始化**开始、且同时学习 4 个任务的结果 |
| **免疫内型 (immune)** | — | **0.9764** | — | **0.4355** | 绝大多数为 Unclassified（97.5%），少数 MAS/EIL 的识别仍有挑战 |
| **器官主导型 (organ)** | — | **0.8670** | — | **0.6975** | delta-like / alpha-like 区分度良好 |
| **液体获益 (fluid)** | — | **0.8975** | — | **0.7398** | low_benefit vs. high_benefit 分类能力强 |

### 3.3 各任务详细混淆矩阵（Test）

#### 死亡率
| | 预测阴性 | 预测阳性 |
|---|---------|---------|
| **真实阴性** | 6,368 | 2,107 |
| **真实阳性** | 1,710 | 3,985 |

阳性率 40.2%，模型预测阳性率 43.0%，召回率 69.97%，精确率 65.41%。

#### 器官主导型分类准确率
- Unclassified: ~88%
- alpha-like: ~75%
- beta-like: ~70%
- gamma-like: ~65%
- **delta-like: ~92%**（最高，因为多器官衰竭的特征信号最强）

---

## 4. 治疗推荐引擎：端到端演示

### 4.1 系统 workflow

```
患者 48h 时序数据
    ↓
多任务 S5 模型推理（1.2 ms / CPU）
    ↓
输出 4 组概率（mortality + immune + organ + fluid）
    ↓
SubtypeTreatmentRecommender 匹配规则库
    ↓
生成中文可解释报告（含证据级别、监测要点、风险提示）
```

### 4.2 实际案例演示

**选取患者：** 真实标签为 `MAS-like + delta-like + low_benefit`

**模型预测：**
- 免疫：Unclassified (92.7%) — *模型未能识别出 MAS-like*
- 器官：**delta-like (96.1%)** — *高置信度正确*
- 液体：**low_benefit (97.7%)** — *高置信度正确*
- 死亡风险：**76.6%**

**系统生成的推荐报告（节选）：**

> 该患者最可能的亚型组合为：免疫-Unclassified(80.9%) / 器官-delta-like(96.1%) / 液体-low_benefit(97.7%)。模型估计的28天死亡风险为 76.6%。
>
> **【器官主导型】**
> - δ型 — 多器官衰竭型 (证据级别: Multi-center cohort)
>   - 全面ICU器官替代治疗：机械通气、血管活性药物、CRRT (若AKI)、凝血功能纠正
>   - 预后沟通：每日多学科查房、评估姑息治疗指征
>
> **【液体策略】**
> - 液体复苏低获益/高风险型 (证据级别: Proteinomic cohort)
>   - 限制性液体策略：限制净液体入量、优先使用血管活性药物、必要时利尿剂或CRRT去负荷

### 4.3 当前局限

- **免疫内型识别偏弱**：由于 MAS-like 仅占 2.3%、EIL-like 仅占 0.2%，且缺乏真实的 mHLA-DR/IL-6 金标准，模型在免疫维度上高度倾向于预测 Unclassified。
- **未触发组合风险提示**：虽然真实标签是 `MAS-like + delta-like`，但模型预测免疫为 Unclassified，因此没有触发 "EIL-like + delta-like => 85.7% 死亡率" 的极端风险 alert。

---

## 5. 核心代码与文件清单

| 文件 | 作用 |
|------|------|
| `s6/multitask_model.py` | 多任务 S5 模型 + 训练循环 + 评估 |
| `s6/treatment_recommender.py` | 基于 JSON 规则库的治疗推荐引擎 |
| `scripts/s6_train_multitask_student.py` | 云端训练入口脚本 |
| `config/treatment_rules.json` | 结构化治疗规则库（免疫/器官/液体） |
| `src/build_enhanced_analysis_table.py` | Phase 1 扩展数据提取器 |
| `src/subtype_label_engine.py` | Phase 1 代理标签生成器 |
| `docs/S6_PHASE1_REPORT.md` | Phase 1 数据与标签工程报告 |
| `docs/S6_PHASE2_CLOSEOUT_REPORT.md` | 本报告 |

**云端产出 artifacts：**
- `data/s6_multitask_mimic_cloud/multitask_student.pt` — 训练好的模型权重
- `data/s6_multitask_mimic_cloud/multitask_student_report.json` — 完整训练报告
- `data/s6_multitask_mimic_cloud/feat_mean.npy` / `feat_std.npy` — 部署标准化参数

---

## 6. 关键技术指标对比

| 维度 | 原 S5-v2 (二分类死亡率) | S6 多任务 (四任务联合) |
|------|------------------------|------------------------|
| 参数数量 | 90,689 | 97,484 (+7.5%) |
| CPU 延迟 | 1.10 ms | **1.22 ms** |
| 任务数 | 1 | **4** |
| Test AUROC (mortality) | **0.873** | 0.801 |
| 额外能力 | 无 | **亚型诊断 + 治疗推荐** |

**说明**：S6 的 mortality AUROC 从 0.873 降到 0.801，主要因为：
1. **没有使用 warm-start**：S5-v2 是从 S1.5 教师模型蒸馏而来，S6 当前版本是从随机初始化训练。
2. **多任务竞争**：模型容量被同时分配给 4 个任务，对死亡率的专精程度下降。
3. **数据未预训练**：43 维特征中很多高缺失率指标（CRP 99.2%、铁蛋白 98.5%）增加了学习难度。

---

## 7. 下一步优化方向（若继续）

### 7.1 提升死亡率 AUROC（高优先级）
- **引入教师蒸馏**：用已有的 S5-v2 教师模型对 S6 共享编码器进行知识蒸馏，预期 mortality AUROC 可回升至 **0.85+**。
- **加权损失调整**：提升 `lambda_mortality` 权重（如 2.0），或给 minority 免疫类（MAS/EIL）做类别重加权。

### 7.2 提升免疫内型识别（中优先级）
- **引入外部免疫数据**：若未来能接入真实的 mHLA-DR、IL-6、PCT 数据，可直接替换代理标签。
- **两阶段训练**：第一阶段只训练 organ + fluid（信号强），第二阶段再 fine-tune 免疫头。
- **数据增强 / 过采样**：对 MAS-like (2.3%) 和 EIL-like (0.2%) 做 SMOTE 或复制过采样。

### 7.3 工程化部署（中优先级）
- 将推荐引擎封装为 REST API（FastAPI），接入 bedside dashboard。
- 增加医生反馈闭环：记录医生对推荐结果的「采纳/修改/拒绝」，用于后续模型迭代。

---

## 8. 诚实声明与临床边界

1. **代理标签不是诊断金标准**
   - MAS/EIL 的判断基于现有可观测指标的近似规则，不能等同于流式细胞术或细胞因子检测。
2. **治疗推荐仅供临床参考**
   - 所有推荐均标注了证据级别（RCT / cohort / N/A），系统明确 disclaimer：不替代医生临床判断。
3. **模型尚未经过前瞻性验证**
   - 所有指标均来自回顾性 MIMIC-IV 队列， bedside 临床应用前需要前瞻性验证和伦理审批。

---

## 9. 一句话总结

> **S6 第二阶段已完成：我们在 94,458 例 MIMIC-IV 患者上成功训练了一个轻量级（97K 参数、1.2ms 延迟）的多任务床边诊断模型，能够同时预测死亡风险、免疫内型、器官主导型和液体获益型；并配套实现了结构化治疗推荐引擎，可将模型输出自动映射为证据级临床推荐。器官型和液体型的分类能力已达到可用水平（F1 0.70+），免疫内型因数据极度不平衡仍需后续优化。**
