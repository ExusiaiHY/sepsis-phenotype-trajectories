# ICU脓毒症自监督时间表型轨迹分析后续研究成果

## 1. 成果边界

本轮工作围绕原论文之后的两个核心延伸模块完成了**可执行研究框架**的实现：

1. **模块3：治疗感知因果时间表型分析**
2. **模块4：轻量级实时在线表型轨迹分析**

已完成内容以**代码、脚本、配置、测试、部署原型、临床落地文档**的形式落入仓库，可直接复现和扩展。

未在本地直接完成的内容：

1. 1~2家三甲医院 ICU 的真实前瞻性临床验证
2. 医疗器械注册与院内部署审批
3. 跨院多中心正式统计学结题分析

这些工作需要 IRB、医院数据接口、临床合作团队，不可能在单机仓库内伪造完成，因此本次成果明确区分为：

1. **仓库内已落地的研究资产**
2. **需要外部条件推进的转化步骤**

## 2. 模块3：治疗感知因果时间表型分析

### 2.1 新增代码资产

- `s4/treatment_features.py`
  统一构建 MIMIC-IV / eICU 的小时级治疗特征张量
- `s4/treatment_aware_model.py`
  在原 S1.5 基础上增加 treatment adapter 分支，保留原生理自监督主干
- `s4/causal_analysis.py`
  实现 PSM、因果森林风格 DML、RDD 三类因果分析
- `scripts/s4_prepare_treatments.py`
  生成治疗时序特征包
- `scripts/s4_train_treatment_aware.py`
  训练治疗感知 S1.5 扩展模型
- `scripts/s4_run_causal_analysis.py`
  构建因果队列并导出因果分析报告
- `config/s4_config.yaml`
  模块3统一配置入口

### 2.2 治疗时序特征定义

统一输出 7 个小时级治疗变量：

1. `vasopressor_on`
2. `vasopressor_rate`
3. `antibiotic_on`
4. `crystalloid_fluid_ml`
5. `fluid_bolus_ml`
6. `mechanical_vent_on`
7. `rrt_on`

数据来源策略：

1. **MIMIC-IV**
   从 `patient_timeseries` 读取 `norepi_rate`，并结合 `inputevents`、`prescriptions`、`procedureevents`、`d_items` 抽取真实治疗暴露
2. **eICU**
   复用已准备的 `time_series_eicu_demo.npy` 中治疗信号，并从 `medication`、`intakeOutput` 补充抗生素和液体复苏暴露

输出格式：

1. `treatments.npy`
2. `masks_treatments.npy`
3. `cohort_static.csv`
4. `patient_level_summary.csv`
5. `treatment_feature_names.json`
6. `treatment_report.json`

### 2.3 模型设计

遵循“最小化重构原模型 + 临床优先”原则，当前采用如下结构：

1. 原 `ICUTransformerEncoder` 继续作为**生理时序主干**
2. 新增 **treatment adapter branch**
   输入为 `concat([treatments, treatment_mask, optional_note_embeddings])`
3. 通过轻量 Transformer treatment encoder 获得治疗序列表征
4. 通过 learned gate 将治疗序列表征与原 S1.5 sequence state 融合
5. 保留 observation-density-weighted pooling 的设计思想
6. 下游头部使用 attention pooling + classifier 做监督适配

这个设计的优点：

1. 原 S1.5 预训练参数可以直接加载
2. 生理表征学习主路径不被破坏
3. 治疗信息作为附加分支进入，便于后续逐步加文本、多模态和蒸馏
4. 更符合论文后续工作里“保留核心架构”的要求

### 2.4 因果推断框架

当前仓库内实现了三条互补链路：

1. **PSM**
   逻辑回归估计倾向得分 + 最近邻匹配，输出 ATE
2. **因果森林风格 DML**
   倾向模型 + 结果模型 + doubly robust pseudo-outcome + RandomForestRegressor，输出 CATE / phenotype-wise HTE
3. **RDD**
   围绕临床阈值的局部线性断点回归

默认临床阈值模板：

1. `vasopressor_on_any_6h` 以 `map_mean_6h = 65 mmHg` 为断点
2. `rrt_on_any_24h` 以 `creatinine_mean_6h = 2.0 mg/dL` 为断点
3. `mechanical_vent_on_any_6h` 以 `spo2_min_6h = 92%` 为断点

建议解读规则已经内置在 `generate_precision_treatment_recommendations(...)` 中：

1. 至少两种方法方向一致才给出候选建议
2. 明确将结论标记为 observational / hypothesis-generating，而不是临床处方

### 2.5 推荐执行顺序

```bash
./.venv/bin/python scripts/s4_prepare_treatments.py --config config/s4_config.yaml
./.venv/bin/python scripts/s4_train_treatment_aware.py --config config/s4_config.yaml
./.venv/bin/python scripts/s4_run_causal_analysis.py --config config/s4_config.yaml --embeddings data/external_temporal/mimic/s15/embeddings_s15.npy
```

### 2.6 建议正式实验

在 MIMIC-IV Sepsis-3 队列上执行：

1. 原始 frozen S1.5 vs treatment-aware S1.5
2. 不同 treatment horizon
   6h / 12h / 24h
3. 不同 phenotype 定义
   first-window / dominant / stable-only
4. 因果方法一致性分析
   PSM vs DML vs RDD

主要报告指标：

1. AUROC
2. Recall
3. Balanced accuracy
4. Brier
5. ECE
6. phenotype-wise CATE
7. overlap / positivity 诊断

## 3. 模块4：轻量级实时在线表型轨迹分析

### 3.1 新增代码资产

- `s5/realtime_model.py`
  轻量 student、动态量化、CPU latency 评估、流式 buffer、在线监控器
- `s5/text_features.py`
  结构化轻量文本嵌入
  `HashingVectorizer -> TruncatedSVD -> hourly tensor`
- `s5/dashboard.py`
  床旁 HTML 界面生成器
- `scripts/s5_build_note_embeddings.py`
  生成 eICU note embedding tensor
- `scripts/s5_distill_realtime.py`
  蒸馏训练 Stage 5 student
- `scripts/s5_build_dashboard.py`
  从 snapshot JSON 导出 bedside dashboard
- `config/s5_config.yaml`
  模块4统一配置入口

### 3.2 轻量 student 设计

轻量化策略采用三步：

1. **结构缩小**
   `d_model` 从 128 缩减到 64，可进一步调到 48/32
2. **知识蒸馏**
   student 对 teacher embedding 做 MSE distillation，同时保留监督 BCE
3. **动态量化**
   优先对 Linear 层做 dynamic quantization；若运行时后端不支持，则自动回退到 float CPU 路径并继续产出 latency 报告

这比直接硬剪枝更稳妥，因为：

1. 保留了原输入语义
2. 更容易控制性能退化
3. 对 ICU 低算力 CPU 终端更现实

### 3.3 文本融合

当前实现没有引入大型本地语言模型，而是采用可部署的轻量路线：

1. eICU `note.csv*` 中聚合 `notetype + notepath + notevalue + notetext`
2. 构造成小时级 note rows
3. 使用 hashing + SVD 生成固定维度向量
4. 作为 `TreatmentAwareEncoder` 的 adapter 输入之一参与融合

这条路线的意义在于：

1. 明确验证“文本 + 结构化时序”是否增益
2. 不引入大模型推理依赖
3. 保持床旁部署成本可控

### 3.4 在线推理链路

`RealtimePatientBuffer` + `RealtimePhenotypeMonitor` 已实现以下流程：

1. 每小时接收一次 vitals / labs / treatments / optional notes 更新
2. 维护固定长度 rolling buffer
3. 缓冲区满后触发 student inference
4. 输出：
   `risk_probability`
   `risk_alert`
   `phenotype`
   `hours_seen`

这个接口可直接作为 ICU-CIS / EMR 集成的服务层原型。

### 3.5 床旁交互界面

`render_clinical_dashboard_html(...)` 生成单文件 HTML，设计目标不是科研图，而是**临床工作流内嵌**：

1. 当前风险与 phenotype 状态卡片
2. rolling risk trajectory
3. rounds / order review / handoff 三段式 workflow card
4. 结构化模型元数据与可追溯信息

建议集成位点：

1. 查房前综述页
2. 医嘱审核侧栏
3. 护士交班/医生交班摘要页

### 3.6 推荐执行顺序

```bash
./.venv/bin/python scripts/s5_build_note_embeddings.py --config config/s5_config.yaml
./.venv/bin/python scripts/s5_distill_realtime.py --config config/s5_config.yaml
./.venv/bin/python scripts/s5_build_dashboard.py --config config/s5_config.yaml --snapshots-json path/to/snapshots.json
```

## 4. 真实临床验证方案

### 4.1 多中心验证建议

建议分三步推进：

1. **离线回顾性验证**
   1 家医院、6~12 个月 ICU 历史队列
2. **静默前瞻性验证**
   模型在线跑但不显示给医生
3. **受控显示验证**
   固定部分床位/班次显示界面并收集反馈

### 4.2 主要观察指标

模型层：

1. AUROC
2. Recall
3. Balanced accuracy
4. Brier / ECE
5. phenotype stability consistency

工作流层：

1. 每次查房额外耗时
2. 告警接受率
3. 医生主观可解释性评分
4. 医嘱前查看率

安全层：

1. 高风险漏检率
2. 错误触发后的临床复核可追溯率
3. 数据接口中断恢复能力

## 5. 监管与合规要点

建议将落地合规拆成 5 条主线：

1. **数据合规**
   脱敏、访问控制、日志审计、最小权限
2. **模型可追溯**
   记录模型版本、训练时间、配置、阈值、输入可用性
3. **临床安全**
   明确 advisory-only，禁止无人工复核自动执行医嘱
4. **注册准备**
   从软件功能边界、预期用途、风险等级开始梳理
5. **漂移监控**
   按季度复核性能、缺失率、治疗模式变化

仓库内当前已经具备的配套条件：

1. 配置化脚本入口
2. JSON 报告输出
3. HTML 界面原型
4. latency / calibration 报告能力
5. 数据与模型分层目录

## 6. 本轮仓库内验证结果

本轮完成了本地 smoke-level 验证：

1. `tests/test_s4_treatment_causal.py`
   验证 eICU 治疗特征抽取、Stage 4 模型训练、因果分析链路
2. `tests/test_s5_realtime.py`
   验证 Stage 5 student 蒸馏、实时 buffer/monitor、HTML dashboard 导出

验证方式：

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python tests/test_s4_treatment_causal.py
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python tests/test_s5_realtime.py
```

## 7. 建议的下一步

优先级建议如下：

1. 先在 MIMIC-IV Sepsis-3 全量队列跑 `S4` 治疗感知 + 因果分析
2. 固定一个 recall 优先阈值，蒸馏 `S5` student 并导出 latency / calibration 报告
3. 在 eICU 上补做 note-fusion ablation
4. 选定 1 家 ICU 做静默前瞻性验证
5. 将 dashboard 嵌入院内现有 EMR/ICU-CIS sandbox

## 8. 对原论文主线的延续关系

本轮实现不是另起炉灶，而是严格沿着原论文路线延伸：

1. **S1.5 自监督表征主干保留**
2. **治疗变量作为并联 adapter 融合**
3. **轨迹分析仍然是 rolling-window 语义**
4. **新增因果分析与 bedside student 作为转化层**

因此，当前仓库已经从：

`静态表型 -> 自监督时间表征 -> 描述性时间轨迹`

扩展为：

`静态表型 -> 自监督时间表征 -> 描述性时间轨迹 -> 治疗感知因果分析 -> 轻量实时部署`
