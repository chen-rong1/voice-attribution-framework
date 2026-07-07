# 模型目录说明

这个目录用于存放 `voice-attribution-framework` 自己使用的本地模型资产。

## 当前规则

- 只保留当前框架运行真正需要的模型文件
- 不把旧项目里无关的训练残留、压缩包、缓存文件整坨复制过来

## 当前已复制资产

- `ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx`
- `ecapa_tdnn/wespeaker_ecapa1024_lm/avg_model.onnx.data`
- `ecapa_tdnn/wespeaker_ecapa1024_lm/config.yaml`

## 为什么这里先保留 WeSpeaker ONNX

原因不是“它效果最好”，而是：

- 它已经在旧项目中验证过可稳定推理
- 它本质上是一个现成可用的 `ECAPA-TDNN` embedding 引擎
- 适合作为新框架第一阶段的底座

后面自研的重点仍然是：

- 音频前处理
- 特征层
- 画像构建
- 打分策略
- `UNKNOWN` 拒识
- benchmark 体系

也就是说：

- 这里借用的是一个底层引擎
- 不是继续把整个第三方项目当成业务系统来依赖
