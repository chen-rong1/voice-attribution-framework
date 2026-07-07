# Embedding 后端抽象设计

## 1. 这份文档解决什么问题

这份文档专门回答下面几个问题：

- 为什么要把不同 `ECAPA` 实现统一抽象
- 抽象层应该长什么样
- 不同后端怎么挂进同一套框架
- 后面接 `PyTorch`、`ONNX`、自训练模型时，接口怎么保持稳定

白话理解：

- 以后不想每接一个模型，就把上层业务代码改一遍
- 所以要先定一个“统一插座”

## 2. 当前问题

现在已有系统里，不同模型的接入方式差异很大。

有的后端是：

- 直接 Python 调包

有的后端是：

- 子进程调用

有的后端是：

- `PyTorch` 直接推理

有的后端是：

- `ONNX Runtime` 推理

表面上它们都能产出 embedding，但工程上有几个问题：

- 输入输出口径不统一
- 前处理可能不统一
- 资源初始化方式不统一
- 缓存方式不统一
- 上层业务不得不感知底层差异

这会导致后面越来越难维护。

## 3. 设计目标

后端抽象层的目标只有一句话：

- 上层业务只认“标准 embedding 接口”，不关心底层具体是谁

更具体一点，就是要做到：

- 不同后端统一输入
- 不同后端统一输出
- 不同后端统一异常风格
- 不同后端统一初始化方式
- 不同后端统一元数据描述
- benchmark 可以无缝切换后端

## 4. 抽象层边界

Embedding 后端抽象层只负责下面这些事：

- 加载模型
- 接收标准化音频或标准特征
- 产出 embedding
- 提供 backend 元数据

它不负责下面这些事：

- 拼片
- 说话人画像聚合
- cosine 打分
- unknown 拒识
- benchmark 报告输出

白话理解：

- 它只是“向量提取机”
- 不是“最终判决器”

## 5. 统一接口设计目标

后面所有 embedding backend 都建议遵守同一套接口约定。

建议至少包含下面这些能力：

- `backend_name`
- `backend_version`
- `feature_version`
- `embedding_dim`
- `load()`
- `is_loaded()`
- `extract_embedding()`
- `extract_embeddings()`
- `unload()`

## 6. 推荐接口语义

### 6.1 `backend_name`

表示后端名称，例如：

- `speechbrain-ecapa`
- `paddlespeech-ecapa`
- `wespeaker-ecapa1024-lm-onnx`

要求：

- 稳定
- 唯一
- 能进数据库

### 6.2 `backend_version`

表示后端实现版本。

这个版本不是只看模型文件名，还应该反映：

- 当前代码实现版本
- 当前模型资产版本

作用：

- 后面 benchmark 能追溯

### 6.3 `feature_version`

表示这个 backend 默认依赖的特征口径版本。

例如：

- `fbank80_v1`
- `fbank80_cmn_v2`

作用：

- 防止不同后端虽然名字一样，但特征口径偷偷变了

### 6.4 `embedding_dim`

表示输出向量维度。

作用：

- 画像聚合和持久化时要用到

### 6.5 `load()`

负责加载模型和相关资源。

要求：

- 支持延迟加载
- 重复调用时不应重复初始化

### 6.6 `is_loaded()`

返回当前 backend 是否已经完成初始化。

作用：

- 服务层可以更容易控制资源状态

### 6.7 `extract_embedding()`

负责处理单条输入，产出单条 embedding。

建议统一输入形式为：

- 标准化后的单条音频

或者：

- 已经准备好的特征矩阵

但不要两者混着来，一定要定清楚。

### 6.8 `extract_embeddings()`

负责批量提 embedding。

作用：

- 提升 benchmark 和画像批处理效率

### 6.9 `unload()`

负责释放资源。

说明：

- 第一版可以先实现成轻量接口
- 后面如果模型越来越重，这个接口会很有用

## 7. 推荐输入输出格式

### 7.1 输入建议

建议后端统一吃下面两类输入之一：

#### 方案 A：吃标准化音频

输入：

- `float32`
- 单声道
- `16k`

优点：

- 上层调用简单

缺点：

- 每个 backend 可能重复做特征提取

#### 方案 B：吃统一特征

输入：

- 统一 `fbank`

优点：

- 特征层和 backend 层职责更清楚
- 更容易保证前处理一致

缺点：

- 有些第三方 backend 原生接口更偏向直接吃音频

### 7.2 当前建议

对你们这个项目，我建议最终走：

- `音频标准化层 -> 特征层 -> Embedding backend`

也就是说：

- 正式方向建议 backend 尽量统一吃特征

但为了平滑落地：

- 第一阶段允许一部分 backend 暂时吃标准化音频
- 后面再逐步收敛到统一特征输入

### 7.3 输出建议

建议统一输出为：

- `np.ndarray`
- `float32`
- 一维向量

不要让上层再处理：

- `torch.Tensor`
- Python list
- shape 不固定的二维数组

统一标准就是：

- 一条输入，出来就是一条一维 embedding

## 8. 推荐数据结构

后面代码里建议明确出两个概念对象。

### 8.1 `EmbeddingRequest`

描述一次后端提向量请求。

建议包含：

- 输入数据路径或内存对象
- 输入类型
- 是否已标准化
- 样本时长
- 样本标识

### 8.2 `EmbeddingResult`

描述一次提向量的结果。

建议包含：

- `backend_name`
- `backend_version`
- `feature_version`
- `embedding`
- `embedding_dim`
- `duration_sec`
- `quality_score`
- `extra_metadata`

作用：

- 后面不只是拿到向量
- 还可以顺手拿到质量信息和版本信息

## 9. backend 分类建议

后面建议从实现角度把 backend 分成 3 类。

### 9.1 原生 Python backend

例如：

- `SpeechBrain`
- `WeSpeaker PyTorch`

特点：

- 直接 import 使用
- 启动方便

### 9.2 子进程 backend

例如：

- 某些必须走独立环境的后端

特点：

- 隔离性强
- 但维护和调试更麻烦

### 9.3 ONNX backend

例如：

- `WeSpeaker ECAPA ONNX`

特点：

- 部署轻
- 启动快
- 更适合后期正式服务

## 10. 第一阶段支持清单

第一阶段建议重点统一下面几类：

- `speechbrain-ecapa`
- `paddlespeech-ecapa`
- `wespeaker-ecapa-pytorch`
- `wespeaker-ecapa-onnx`

原因：

- 它们都和 `ECAPA-TDNN` 路线强相关
- 很适合做第一版统一抽象

## 11. 后端注册机制

后面建议不要在业务代码里写一大串 `if/else`。

建议做成：

- backend registry

也就是：

- 一个名字
- 对应一个 backend 实现类

上层只需要：

- 传入名字
- 拿到实例

这样后面新增 backend 时：

- 只需要注册
- 不用到处改业务代码

## 12. 初始化策略

后面 backend 的初始化建议采用：

- 延迟加载

也就是：

- 服务启动时不全量把所有模型都加载进来
- 第一次真正用到时再加载

原因：

- 有些 backend 很重
- benchmark 虽然会全跑，但线上服务不一定每次都用全部模型

## 13. 缓存策略

后面建议分两层缓存。

### 13.1 模型级缓存

指：

- backend 实例缓存
- 避免反复加载模型

### 13.2 embedding 级缓存

指：

- 已注册样本 embedding 缓存

作用：

- benchmark 和多轮识别时，不用反复重算注册样本向量

## 14. 异常设计建议

后端抽象层需要统一错误风格。

建议至少区分：

- 模型文件不存在
- backend 初始化失败
- 输入音频非法
- 特征提取失败
- embedding 为空
- 维度不匹配

目的：

- 上层服务能够正确报错
- benchmark 能明确知道失败原因

## 15. 与画像层的关系

Embedding backend 和画像层一定要分清楚。

backend 只负责：

- 提供“每条样本的向量”

画像层负责：

- 把多条样本向量组织起来
- 聚合成说话人 profile

不要让 backend 直接做画像聚合。

否则后面：

- 很难替换聚合策略
- 很难做统一 benchmark

## 16. 与 benchmark 的关系

benchmark 不应该直接操作某个具体第三方模型。

benchmark 只应该操作：

- 标准 backend 接口

这样以后 benchmark 才能真正做到：

- 同口径比较
- 同输入比较
- 同输出格式比较

## 17. 第一阶段最小可用方案

如果现在马上开始搭代码，第一阶段的最小可用方案建议如下：

1. 先定义统一 backend 基类
2. 先接 `WeSpeaker ECAPA ONNX`
3. 再接 `SpeechBrain ECAPA`
4. 再接 `Paddle ECAPA`
5. 最后把当前现有调用方式逐步迁移到 registry

原因：

- `WeSpeaker ONNX` 已经在你们这里验证过
- 很适合作为“标准样板”

## 18. 当前决定

从现在开始，Embedding 后端设计正式按下面的原则执行：

- 上层只认统一接口
- backend 尽量只做向量提取
- 画像聚合不放进 backend
- 优先支持 `ECAPA` 家族统一接入
- 后期正式服务优先走 `ONNX` 路线

## 19. 下一步

这份文档定下来后，最该继续补的就是：

- `05_打分与拒识策略设计.md`

因为后端统一后，真正决定业务效果的下一层就是：

- 怎么比
- 怎么判
- 什么时候拒识
