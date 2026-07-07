# 模型来源说明

本文件用于记录 `voice-attribution-framework` 仓库内随代码一同分发的模型资产来源、可确认信息和分发说明。

## 适用范围

当前说明适用于以下目录中的模型资产：

- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx.data`
- `models/ecapa_tdnn/wespeaker_ecapa1024_lm/config.yaml`

## 当前资产说明

当前默认使用的 embedding backend 为：

- `wespeaker-ecapa1024-lm-onnx`

对应默认模型目录为：

- `models/ecapa_tdnn/wespeaker_ecapa1024_lm`

从目录命名、配置文件内容以及项目接入方式来看，这组资产对应于 `WeSpeaker` 生态下的 `ECAPA-TDNN` ONNX runtime 模型口径。

## 当前可确认的技术信息

根据随仓库分发的 `config.yaml`，当前可确认的信息包括：

- 模型名称：`ECAPA_TDNN_GLOB_c1024`
- embedding 维度：`192`
- 特征维度：`80` mel bins
- 池化方式：`ASTP`
- `LM` 口径：目录命名使用 `ECAPA1024_LM`
- 训练数据字段：`train_data: data/vox2_dev/shard.list`

从这些字段可以看出，当前资产至少具备以下特征：

- 属于 `ECAPA-TDNN` 家族
- 是面向运行时推理的 ONNX 资产
- 配置中存在 `VoxCeleb2` 风格训练数据路径标记

## 上游参考来源

这组模型资产的公开说明可同时引用以下上游信息：

- 上游项目：`https://github.com/wenet-e2e/wespeaker`
- 上游项目许可证口径：`Apache-2.0`
- 上游预训练模型说明：`https://wenet-e2e.github.io/wespeaker/pretrained.html`

上游文档中有两点说明与当前资产直接相关：

1. `onnx` 文件属于从 checkpoint 导出的 runtime model
2. 预训练模型的许可证通常跟随其对应数据集

## 许可与分发说明

需要特别区分下面两件事：

- 仓库代码许可证
- 模型资产许可证

当前仓库代码采用 `MIT`，但这 **不自动等同于** 目录内模型资产也按 `MIT` 分发。

对于这组模型资产，当前文档采用如下分发口径：

- 代码仓库许可证：`MIT`
- 模型资产来源：`WeSpeaker` 生态的 runtime 模型资产
- 模型资产许可：应同时遵守上游项目说明以及对应训练数据集的许可证要求

如果上游模型说明指向 `VoxCeleb` 体系，则发布者还应额外关注数据集许可证是否对再分发、署名或用途有附加要求。

## 发布者责任

如果继续保留这组模型文件并公开发布当前仓库，建议在发布说明中明确以下几点：

- 这组模型文件并非由本仓库重新训练得到
- 本仓库仅对其进行集成、封装与业务流程接入
- 使用者应同时阅读上游项目和对应数据集的许可证说明
- 如上游许可证发生变化，应以最新公开说明为准

## 完整性校验

当前仓库中这 3 个文件的 `SHA-256` 如下：

- `avg_model.onnx`
  - `855791bde0afaa92b98b1ef4a32160932477c1d3a8b3495d73fae77e05177c3e`
- `avg_model.onnx.data`
  - `748b7c847f59aa8cc27006308712bde5d5c6470d2a4c794456330228776d8567`
- `config.yaml`
  - `6ff025d80b51a906fdad7469e6292e35f759dfcf8b8a45963025afbb001e781b`

这些校验值用于：

- 确认发布资产未被意外改动
- 方便后续替换或升级模型时做版本核对
