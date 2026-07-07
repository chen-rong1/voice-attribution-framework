# voice-attribution-framework

## 这是什么

`voice-attribution-framework` 是一个面向真实业务场景的自研声纹归属框架。

它现在重点解决的不是“开放集大模型训练”，而是下面这条业务链路：

- 员工先录注册音频
- 外场长录音先做说话人分离
- 分离后的音频片段进入框架
- 框架判断每段属于哪个已注册员工，或者判成 `UNKNOWN`

白话理解：

- `voice-attribution-framework` 是我们要真正做成业务底座的新框架

## 现在做到哪了

目前已经打通了“离线业务闭环”，也就是：

- 注册音频 -> 员工画像
- 分离片段 -> embedding 提取
- embedding -> 打分归属
- 归不到任何注册员工 -> `UNKNOWN`
- 业务集评测 -> 中文报告输出

当前已经验证过两类数据：

- 真实业务集
  - `辽宁0222_前5分钟_diarization_thr1.0_merge_gap1.2`
- 标准严格集
  - `voice-benchmark-strict/processed/enroll/eval`
  - `voice-benchmark-strict/processed/attribution/eval`

当前阶段结论：

- 业务集最优结果：`30/31 = 96.77%`
- 标准严格集结果：`8/8 = 100.00%`

这说明当前框架已经不是“只能跑 demo”，而是已经有了可复现、可验证的业务方案基线。

## 适合什么业务

当前最适合的业务形态是：

- 已知一批员工或目标说话人
- 每个目标说话人可以提供少量注册音频
- 上游已经能把长录音切成单说话人片段
- 业务目标是：
  - 识别这段是不是某个已注册员工
  - 如果不是，就拒识成 `UNKNOWN`

典型例子就是你现在要做的“加油站员工归属”：

- 每个员工先注册
- 加油区录音先做说话人分离
- 分离后片段传给框架
- 框架输出：`员工A / 员工B / UNKNOWN`

## 现在还没做完的部分

当前已经有“核心识别引擎”，但还没完全做成生产系统。

还没完全收口的部分主要是：

- 员工注册库后台管理
- HTTP/API 服务化入口
- diarization 上游自动对接
- 结果入库和业务系统集成
- 更完整的脏样本治理和批量任务调度

所以准确说法是：

- 核心业务闭环已经能跑
- 生产化接入层还需要继续补

## 目录说明

当前这个项目目录里主要放四类东西：

- 设计文档
- 核心代码
- 配置文件
- benchmark 和实验产物

目前重点文件包括：

- `README.md`
- `01_总体方案与实施计划.md`
- `02_目录结构设计.md`
- `03_数据库与画像库设计.md`
- `04_Embedding后端抽象设计.md`
- `05_打分与拒识策略设计.md`
- `06_Benchmark与报告规范.md`
- `07_开发日志.md`

## 当前底座说明

当前底层 backbone 先统一围绕 `ECAPA-TDNN`。

第一版稳定可跑底座是：

- `WeSpeaker ECAPA1024_LM ONNX`

但这里要特别强调：

- 这不代表我们要继续依赖 `WeSpeaker` 原项目整套逻辑
- 也不代表我们只是“接第 11 个第三方项目”

真正的定位是：

- 把 `WeSpeaker ONNX` 当成一个可控的 embedding 引擎
- 在这个引擎上自研：
  - 前处理
  - 画像构建
  - 打分与拒识
  - benchmark
  - 业务接入层

白话理解：

- `WeSpeaker ONNX` 只是发动机毛坯
- 这个仓库要做的是整车

## 当前模型资产

为了保证这个项目以后能独立上传和运行，当前已经把模型资产本地化到项目内：

- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx.data`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/config.yaml`

当前原则是：

- 只保留运行 ONNX 推理真正需要的文件
- 不把旧项目里无关残留一股脑复制进来

## 当前核心能力

目前已经落地的核心能力包括：

- 音频标准化加载
- 轻量质量分估计
- ECAPA-TDNN ONNX embedding 提取
- 质量感知画像构建
- 多种打分策略
  - `center`
  - `max`
  - `top_k_mean`
  - `quality_weighted_center`
- `UNKNOWN` 拒识
- 业务集加载
- 标准严格集加载
- 阈值扫描
- 注册样本推荐
- 注册样本组合搜索
- 官方方案固化与一键复跑

## 数据输入方式

当前已经支持四种输入形态：

- 目录模式
  - 注册目录：`enrollments/<speaker_id>/*.wav`
  - 测试目录：`testset/<label>/*.wav`
- 平铺标准严格集目录
  - 例如：
    - `enroll_ES2005_A_1.wav`
    - `clip_ES2005_A_pos_1.wav`
- manifest 模式
  - 适配类似 `voice-benchmark-strict/meta/eval_clip_manifest_2026-07-02.csv`
- 业务平铺目录模式
  - 例如：
    - 一批平铺 `.wav`
    - `merged_truth.tsv`
    - `pure_test_files.txt`
    - `segments.json`

## 主脚本

当前主要脚本有这几个：

- `scripts/run_benchmark.py`
  - 跑单次 benchmark
- `scripts/scan_business_thresholds.py`
  - 扫业务集阈值
- `scripts/recommend_business_enrollment_pack.py`
  - 推荐注册样本
- `scripts/search_best_enrollment_combination.py`
  - 搜索最优注册组合
- `scripts/run_official_liaoning0222_solution.py`
  - 一键复跑当前官方方案

## 快速开始

### 目录模式

```bash
.venv/bin/python scripts/run_benchmark.py \
  --enroll-dir /path/to/enrollments \
  --test-dir /path/to/testset \
  --output-dir outputs/benchmark/demo \
  --run-name demo_run \
  --dataset-name demo_dataset
```

### manifest 模式

```bash
.venv/bin/python scripts/run_benchmark.py \
  --output-dir outputs/benchmark/strict_eval \
  --run-name strict_eval \
  --dataset-name voice-benchmark-strict \
  --dataset-version 2026-07-02 \
  --manifest-path /path/to/meta/eval_clip_manifest.csv \
  --dataset-root /path/to/dataset_root
```

### 业务平铺目录模式

```bash
.venv/bin/python scripts/run_benchmark.py \
  --business-dataset-dir /path/to/business_dataset \
  --enroll-speaker xiaoli \
  --enroll-list /path/to/enroll_list.txt \
  --scoring-config /path/to/scoring.yaml \
  --output-dir outputs/benchmark/business_run \
  --run-name business_run \
  --dataset-name business_dataset
```

当前脚本会自动输出：

- 中文 `TSV` 总表
- 中文 `Markdown` 正式报告
- `JSON` 摘要

## 当前官方方案

目前已经固化出一套“辽宁0222 / 小丽”官方验证方案。

它的目标很明确：

- 业务集结果稳定复现
- 标准严格集不翻车
- 不再依赖手工复制一长串 `--enroll-file`

当前官方方案由两部分组成：

- 打分配置：
  - `configs/scoring/business_best_verified.yaml`
- 注册清单：
  - `configs/enrollment_packs/liaoning0222_xiaoli_best_verified.txt`

这套方案当前验证结果为：

- 业务集：`30/31 = 96.77%`
- 标准严格集：`8/8 = 100.00%`

如果只想跑业务集，可直接使用：

```bash
.venv/bin/python scripts/run_benchmark.py \
  --business-dataset-dir /Users/工作/声纹识别/voice-benchmark-strict/processed/attribution/辽宁0222_前5分钟_diarization_thr1.0_merge_gap1.2 \
  --enroll-speaker xiaoli \
  --enroll-list configs/enrollment_packs/liaoning0222_xiaoli_best_verified.txt \
  --scoring-config configs/scoring/business_best_verified.yaml \
  --output-dir outputs/benchmark/liaoning0222_official_business \
  --run-name liaoning0222_official_business \
  --dataset-name liaoning0222_business
```

如果想一键同时复跑业务集和标准严格集，可直接使用：

```bash
.venv/bin/python scripts/run_official_liaoning0222_solution.py
```

这个脚本会自动输出：

- `outputs/benchmark/official_liaoning0222_solution/business/`
- `outputs/benchmark/official_liaoning0222_solution/strict/`
- `outputs/benchmark/official_liaoning0222_solution/官方方案汇总.json`
- `outputs/benchmark/official_liaoning0222_solution/官方方案汇总.md`

## 当前结论

当前方向已经很明确：

- 不从 0 训练一个全新大模型
- 先自研适配真实业务场景的 ECAPA 声纹归属框架
- 底层 backbone 先统一围绕 `ECAPA-TDNN`
- 先把注册画像、归属打分、拒识、业务接入和 benchmark 体系做扎实

## 后续方向

接下来更重要的工作不是再堆更多模型，而是继续补业务化接入层：

- 员工注册库管理
- diarization 输出自动接入
- 批量任务处理
- 结果落库
- API 服务化

到那一步，这个框架才算真正从“离线验证框架”走向“业务系统底座”。
