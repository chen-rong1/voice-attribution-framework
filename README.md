# voice-attribution-framework

`voice-attribution-framework` 是一个面向已知说话人集合的离线声纹归属框架，重点解决“注册一批目标说话人后，对分离出的语音片段进行归属或拒识”的问题。

它不是一个通用语音大模型训练仓库，也不是一个完整的在线服务系统；当前定位是一个可复用、可扩展、可评测的离线基座。

## 一眼看懂

- 目标问题：给定一批已注册说话人，判断测试片段属于谁，或输出 `UNKNOWN`
- 当前形态：离线框架 + CLI + benchmark 体系
- 适用前提：上游已完成 diarization 或已经拿到单说话人片段
- 默认后端：基于 ONNX Runtime 的 `ECAPA-TDNN`
- 仓库定位：基座项目，不承载一次性的业务专项实验脚本

## 为什么做这个项目

许多现成说话人识别项目更偏向模型能力展示或通用验证，不直接面向“注册样本有限、需要归属与拒识、还要可复现评测”的实际流程。

这个项目关注的是一条更工程化的链路：

- 注册样本管理
- 说话人画像构建
- 归属打分
- `UNKNOWN` 拒识
- benchmark 与报告导出

该项目的重点不在于重新实现训练框架，而在于将注册、归属、拒识和评测流程整理为稳定的可复用底座。

## 适用场景

这个项目适合下面这类任务：

- 已知一批目标说话人
- 每个说话人能提供少量注册音频
- 上游已经完成长录音切分，输入是单说话人片段
- 目标是判断片段属于哪位已注册说话人，或拒识为 `UNKNOWN`

典型流程如下：

1. 准备注册音频并构建说话人画像
2. 对测试片段提取 embedding
3. 使用打分策略完成归属判断
4. 当最高分不满足条件时输出 `UNKNOWN`

## 不适合什么

下面这些并不是这个仓库当前的目标：

- 从 0 训练一个新的大规模声纹模型
- 替代完整的 diarization 系统
- 提供开箱即用的在线 API 服务
- 在完全开放集、完全无注册样本的场景直接做身份识别
- 承载一次性的业务实验脚本、专题报告和临时结果文件

## 当前状态

当前仓库已经具备完整的离线闭环能力：

- 注册音频加载与标准化
- embedding 提取
- 说话人画像构建
- 多种打分与拒识策略
- benchmark 执行
- TSV / Markdown / JSON 结果导出

当前还不是完整的生产系统。下面这些能力仍属于后续工作：

- 注册库管理
- API / 服务化接入
- 与 diarization 上游的自动化衔接
- 批量任务调度
- 结果入库和业务系统集成

## 核心能力

目前仓库内提供的核心能力包括：

- 音频标准化加载
- 轻量质量分估计
- 基于 ONNX Runtime 的 ECAPA-TDNN embedding 提取
- 画像构建
- 多种打分策略
- `UNKNOWN` 拒识
- 多种 benchmark 数据加载方式
- 标准化报告导出
- `external_known` / `external_unknown` 混合 holdout 评测

当前已实现的画像聚合 / 打分相关策略包括：

- `center`
- `max`
- `top_k_mean`
- `quality_weighted_center`

## 项目亮点

- 基座清晰：核心代码、默认配置、CLI、测试、文档分层明确
- 可评测：除识别结果外，还提供 benchmark 和标准化导出
- 可拒识：默认将 `UNKNOWN` 作为正式输出，而不是只做强制分类
- 可扩展：embedding backend、画像构建、打分策略都做了模块化抽象
- 可落地：更贴近“注册一批人后进行片段归属”的业务流程

## 模型与实现

当前默认 embedding backend 基于 `ECAPA-TDNN`，以本地 ONNX 模型作为推理后端。

仓库内保留了运行所需的最小模型资产：

- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx.data`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/config.yaml`

这里的目标不是集成某个第三方项目的整套逻辑，而是将可控的 embedding 引擎作为底层能力，并在其上提供：

- 前处理
- 画像构建
- 打分与拒识
- benchmark 体系
- 业务接入层

## 仓库结构

当前仓库重点包含以下内容：

- `app/`: 核心实现
- `configs/`: 通用默认配置
- `models/`: 本地模型资产
- `scripts/`: 正式 CLI 入口
- `tests/`: 单元测试与集成测试

设计文档和内部研发记录建议保留在本地工作区或私有仓库中，不作为公开发布内容的一部分。

目前保留在基座仓库中的正式入口是：

- `scripts/run_benchmark.py`

## 环境要求

- Python `3.12+`
- macOS / Linux 优先
- 推荐使用 `uv`
- 需要本地可用的 ONNX Runtime 运行环境

## 数据输入方式

`run_benchmark.py` 当前支持以下几种输入形态：

- 目录模式
  - 注册目录：`enrollments/<speaker_id>/*.wav`
  - 测试目录：`testset/<label>/*.wav`
- manifest 模式
  - 通过 CSV manifest 指定注册与测试样本
- 平铺业务目录模式
  - 使用平铺 `.wav` 目录配合真值表或纯净样本清单

## 安装

项目使用 Python `3.12+`。

如果你使用 `uv`：

```bash
uv sync
```

如果你使用 `pip`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 快速开始

### 目录模式

```bash
python scripts/run_benchmark.py \
  --enroll-dir /path/to/enrollments \
  --test-dir /path/to/testset \
  --output-dir ./outputs/demo \
  --run-name demo_run \
  --dataset-name demo_dataset
```

### manifest 模式

```bash
python scripts/run_benchmark.py \
  --manifest-path /path/to/eval_manifest.csv \
  --dataset-root /path/to/dataset_root \
  --output-dir ./outputs/manifest_demo \
  --run-name manifest_demo \
  --dataset-name manifest_dataset
```

### 平铺业务目录模式

```bash
python scripts/run_benchmark.py \
  --business-dataset-dir /path/to/business_dataset \
  --enroll-speaker speaker_a \
  --enroll-list /path/to/enroll_list.txt \
  --scoring-config /path/to/scoring.yaml \
  --output-dir ./outputs/business_demo \
  --run-name business_demo \
  --dataset-name business_dataset
```

脚本默认输出：

- 中文 `TSV` 总表
- 中文 `Markdown` 报告
- `JSON` 摘要

当前摘要与报告会固定输出以下关键信息：

- 总体准确率
- `UNKNOWN` 拒识率
- `external_known` Top1 准确率
- `external_unknown` 拒识率
- `REVIEW` 数量
- 平均时延与最大时延
- 标准化 `decision_reason`
- 结构化 `decision_evidence`

其中：

- `decision_reason` 用于表达最终判决理由，适合作为聚合统计键
- `decision_evidence` 用于表达结构化证据，当前采用分层 schema：
- `summary`
- `score_evidence`
- `gate_evidence`
- `candidate_evidence`
- `profile_evidence`
- `query_evidence`

### Holdout 评测建议

如果要用这个基座做更标准的开放集评测，推荐使用 manifest 模式，并在同一份清单中同时包含：

- enrollment 样本
- `external_known` positive query
- `external_unknown` unknown query

manifest 推荐显式提供 `trial_role` 字段来表达试验语义：

- `external_known_query`
- `external_unknown_query`

为了兼容旧数据，当前 loader 仍接受 `trial_label`，并自动将以下旧值映射到标准语义：

- `pos` / `positive` / `known` / `external_known`
- `unknown`

推荐的 manifest 列至少包括：

- `kind`
- `output_rel_path`
- `agent`
- `trial_role`

manifest loader 当前会对以下内容做显式校验，并在 CLI 中直接报错：

- 缺少必填列：`kind` / `output_rel_path` / `agent`
- 行内缺少关键字段值
- 不支持的 `kind`
- attribution 样本缺少 `trial_role`
- 非法 `trial_role`

CLI 对 manifest 校验失败会输出稳定的结构化行格式，包含：

- `code`
- `row`
- `column`
- `message`

同时建议在 CLI 中显式传入：

```bash
python scripts/run_benchmark.py \
  --manifest-path /path/to/holdout_manifest.csv \
  --dataset-root /path/to/dataset_root \
  --dataset-role external_holdout \
  --output-dir ./outputs/holdout_demo \
  --run-name holdout_demo \
  --dataset-name holdout_dataset
```

一个更推荐的 holdout manifest 片段示例如下：

```csv
kind,output_rel_path,source_file,start_sec,end_sec,agent,trial_role,trial_label,note
enroll,processed/enroll/eval/enroll_A_1.wav,raw/a.wav,0,1.5,A,,enroll,demo
attribution,processed/attribution/eval/clip_A.wav,raw/a.wav,0,1.5,A,external_known_query,pos,demo
attribution,processed/attribution/eval/clip_C.wav,raw/c.wav,0,1.5,C,external_unknown_query,unknown,demo
```

## 最小上手路径

首次使用时，建议按下面顺序上手：

1. 先准备一小批注册音频和测试片段
2. 使用目录模式跑通一次 `run_benchmark.py`
3. 查看输出的 TSV / Markdown / JSON
4. 再根据需要切换到 manifest 模式或业务平铺目录模式
5. 最后再考虑自定义 scoring 配置或替换 backend

## 配置

基座仓库目前只保留通用默认配置：

- `configs/audio/default.yaml`
- `configs/benchmark/default.yaml`
- `configs/models/default.yaml`
- `configs/scoring/default.yaml`

业务专项配置建议放在独立的实验层仓库中，而不是直接放入基座仓库。

## 测试

运行测试：

```bash
pytest
```

如果只想跑某一类测试：

```bash
pytest tests/unit
pytest tests/integration
```

## 限制说明

在使用这个项目之前，建议注意下面几点：

- 输入音频质量会显著影响结果
- 如果注册样本过短、过少或存在串音，画像稳定性会下降
- 当前仓库默认面向离线流程，不等同于现成在线服务
- 上游 diarization 质量会直接影响最终归属效果
- 当前默认模型与配置不保证适合所有语言、所有场景、所有设备录音条件

## 模型来源与合规

仓库当前包含运行所需的本地模型文件。公开发布或对外分发前，建议明确检查：

- 模型文件的来源
- 模型权重的再分发许可
- 第三方依赖的许可证要求
- 是否需要在 README 中补充致谢或来源说明

对于合规要求较高的场景，建议在公开仓库或对外分发前先补齐这部分信息。

当前仓库已经提供模型来源说明文档：

- `models/MODEL_PROVENANCE.md`

其中包含：

- 当前打包模型的来源口径
- 可确认的配置特征
- 发布时的许可说明
- 文件完整性校验值

## FAQ

### 这个项目能直接处理长录音吗

不能直接替代 diarization。当前更推荐把它放在说话人分离之后，处理单说话人片段。

### 这个项目能直接拿来上线吗

当前版本更适合离线验证、策略研发和业务接入前的基座建设，不建议直接视为完整生产系统。

### 只有一条注册样本能不能用

可以跑，但风险会明显变高。注册样本过少、过短或不干净时，误认和漏认都更容易发生。

## 社区与协作

如果你希望参与协作或提交问题，可优先阅读以下文件：

- `CONTRIBUTING.md`
- `SECURITY.md`
- `CODE_OF_CONDUCT.md`

仓库已提供：

- Bug 反馈模板
- 功能建议模板
- Pull Request 模板

## 后续方向

后续更重要的工作主要有：

- 注册库管理
- diarization 自动接入
- API 服务化
- 批量任务处理
- 结果落库与业务集成

## Roadmap

- 完善注册样本质量约束
- 增强拒识与边界保护策略
- 提供更清晰的服务化接入层
- 补充更标准的公开示例与许可证信息

## License

仓库代码当前采用 `MIT` 许可证，见 `LICENSE`。

需要额外注意的是：

- 仓库代码使用 `MIT`
- 仓库内模型文件是否允许再分发，仍需单独核查
- 第三方依赖与模型来源的许可证要求，也应在正式公开前确认清楚
