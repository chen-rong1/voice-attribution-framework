# 模型资产说明

本目录用于存放 `voice-attribution-framework` 运行所需的本地模型资产。

## 目录用途

`voice-attribution-framework` 当前使用本地 ONNX 模型完成 embedding 提取。为保证仓库在离线环境下具备最小可运行能力，默认模型文件随仓库一同提供。

本目录只保留运行当前默认后端所需的必要文件，不包含训练检查点、压缩包或与推理无关的中间产物。

## 当前包含的资产

当前默认模型目录如下：

- `ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx`
- `ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx.data`
- `ecapa_tdnn/wespeaker_ecapa1024_lm/config.yaml`

这些文件对应仓库默认使用的 `ECAPA-TDNN` ONNX 推理后端。

## 默认用途

当前仓库将这组模型资产用于：

- 提取说话人 embedding
- 支撑注册画像构建
- 支撑归属打分与 benchmark 流程

这些文件是框架默认运行路径的一部分，并不代表本仓库提供完整的模型训练能力。

## 来源与许可

模型资产的来源、公开分发口径和完整性校验说明见：

- `models/MODEL_PROVENANCE.md`

需要特别注意的是：

- 仓库代码许可证与模型资产许可证不是同一件事
- 使用或再分发模型文件前，应确认上游项目和对应数据集的许可要求
- 如需在更严格的合规场景中使用，建议先完成来源和许可证复核

## 为什么保留这组模型

当前默认模型资产采用 `WeSpeaker` 风格的 `ECAPA-TDNN` runtime 口径，主要原因是：

- 能够满足当前框架的本地推理需求
- 适合作为默认 embedding backend 集成到框架中
- 与当前仓库的离线 benchmark 和归属流程兼容

本仓库对这些资产的定位是“默认推理后端”，而不是第三方项目的完整镜像或训练仓库。

## 维护说明

如果后续替换默认模型资产，建议同步更新以下内容：

- `configs/models/default.yaml`
- `models/MODEL_PROVENANCE.md`
- 相关测试中的默认 backend 名称或模型目录

如果发布版本中继续包含模型文件，也建议保留完整性校验信息，便于版本核对和问题排查。
