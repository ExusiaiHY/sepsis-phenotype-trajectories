# 云端服务器信息

## 基本信息

| 项目 | 内容 |
|------|------|
| 公网 IP | 8.145.60.211 |
| 账号 | root |
| 操作系统 | Ubuntu Linux 6.8.0 (x86_64) |
| CPU/内存 | 184GB RAM |
| 磁盘 | 295GB（上传时可用约268GB） |

## SSH 连接

```bash
ssh root@8.145.60.211
# 密码：见项目内部记录
```

## 项目路径

- 服务器端项目目录：`/root/project/`

## 首次上传内容（2026-04-06）

上传由 Claude Code 完成，包含以下内容：

### 代码与配置（~12MB，312个文件）
- `s0/`, `s1/`, `s15/`, `s2light/`, `s4/`, `s5/`, `s6/`, `s6_optimization/`
- `scripts/`, `src/`, `tests/`, `config/`
- `webapp/`, `webapp_v2/`, `webapp_s6/`
- `docs/`, `CLAUDE.md`, `requirements.txt`, `README.md`

### 核心处理数据（~330MB，排除 .npy 大数组）
- `data/processed/` — 主处理数据
- `data/processed_eicu_real/` — EICU 处理数据
- `data/processed_mimic_enhanced/` — MIMIC 增强处理数据
- `data/s0/`, `data/s4/`, `data/s5/` — 各阶段中间数据

### 数据库（18GB）
- `db/mimic4_real.db` — MIMIC-IV DuckDB 数据库

### 未上传（原始数据，服务器可从 PhysioNet 重新下载）
- `mimic-iv-3.1/`（9.9GB 原始MIMIC数据）
- `EICU 2.0数据/`（5.1GB 原始EICU数据）
- `archive/`（旧版实验文件）
- `upload_bundles/`（打包文件）
- `.npy` 大数组文件（可由代码重新生成）
- `.pt` 模型检查点（在 `archive/` 中，未上传）

## 环境配置（在服务器上运行）

```bash
# 进入项目目录
cd /root/project

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行时设置（必须）
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

## 注意事项

- `.npy` 文件未上传（属于大型数组，可由各阶段脚本重新生成）
- 原始数据未上传，如需重跑 S0 数据预处理，需另行下载 MIMIC-IV / EICU 数据集
- 服务器无 swap，大内存操作请注意限制并发进程数
