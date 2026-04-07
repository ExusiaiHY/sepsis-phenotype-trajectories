# S6 Subtype API

`webapp_v2` 已暴露新的主线亚型接口，前端直接对接以下路径即可：

- `GET /api/sepsis-subtypes/metadata`
- `POST /api/sepsis-subtypes/predict`
- `POST /api/sepsis-subtypes/recommend`

## 重要语义

- `clinical_phenotype` 家族是 `alpha / beta / gamma / delta`
  - 这是临床器官功能障碍表型家族
- `trajectory_phenotype` 家族是 `Trajectory A / B / C / D`
  - 这是 ICU 早期生命体征轨迹家族
- 这两个家族必须在前端分开展示，不能混成同一组标签

## Metadata

`GET /api/sepsis-subtypes/metadata`

返回：

- `contract_version`
- `model`
- `families`
- `classification_tasks`
- `regression_tasks`
- `legacy_aliases`
- `frontend_notes`

前端建议先拉一次这个接口，再按 `family_id` 和 `classes` 渲染 subtype 说明。

## Predict

`POST /api/sepsis-subtypes/predict`

请求体：

```json
{
  "time_series": [[[0.0, 0.0, 0.0]]],
  "mortality_threshold": 0.4
}
```

说明：

- `time_series` 形状是 `(B, T, F)`
- 当前主线模型 `F = 43`
  - `42` 个连续特征
  - `1` 个治疗特征

返回核心字段：

- `model`
- `predictions`

每个 `prediction` 包含：

- `mortality`
- `classification_tasks`
- `regression_tasks`
- `families`
- `legacy_outputs`

前端优先读 `families`，因为这里已经按家族归好类：

- `gold_standard`
- `immune_state`
- `clinical_phenotype`
- `trajectory_phenotype`
- `fluid_strategy`
- `scores`

## Recommend

`POST /api/sepsis-subtypes/recommend`

请求体：

```json
{
  "time_series": [[[0.0, 0.0, 0.0]]],
  "mortality_threshold": 0.4,
  "probability_threshold": 0.3,
  "top_k": 2
}
```

返回在 `predict` 基础上新增：

- `recommendations`

每个 recommendation 包含：

- `family_recommendations`
- `action_plan`
- `monitoring_plan`
- `alerts`
- `summary`
- `disclaimer`

## 前端建议

- 主卡片按 `families.clinical_phenotype` 和 `families.trajectory_phenotype` 分两块显示
- `predicted_label` 用于程序判断
- `predicted_display_name_zh` 用于中文展示
- `probabilities` 可用于显示 top-2 或概率条
- `regression_tasks` 或 `families.scores` 可用于次级解释，不建议替代主标签

