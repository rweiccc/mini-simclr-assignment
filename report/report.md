# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- 论文名称：A Simple Framework for Contrastive Learning of Visual Representations
- 论文地址：https://arxiv.org/abs/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

## 2. 任务说明

本实验复现的是 SimCLR 自监督对比学习方法，用于学习图像表征。

预训练输入：无标签 CIFAR-10 图像  
预训练目标：同一图像的两种增强视图在特征空间中拉近，不同图像拉远  
评估方式：冻结 encoder + 训练 linear probe + 测试分类准确率

## 3. 数据集

- 数据集名称：CIFAR-10
- 数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- 实际使用预训练图像数：2250
- 实际使用 linear probe 训练图像数：
- 实际使用测试图像数：
- 使用设备：CPU
- 总训练耗时：

## 4. 数据增强

请说明自己使用的增强策略：

增强方法	               参数设置
RandomResizedCrop	      scale=(0.2, 1.0)
RandomHorizontalFlip	  p=0.5
ColorJitter	         brightness=0.4, contrast=0.4, saturation=0.4
RandomGrayscale           p=0.2
GaussianBlur	          kernel_size=3

请说明为什么这些增强适合 SimCLR：

```text
这些增强方式可以生成同一图像的不同“语义保持但像素变化较大”的视图。
SimCLR 的核心目标是让模型学习“语义不变性”，而不是像素一致性，因此这些增强可以有效提升模型对视觉变化的鲁棒性。
```

## 5. 模型结构

请说明自己的 Mini-SimCLR 结构：

```text
Image -> Two Augmented Views -> Shared Encoder -> Projection Head -> NT-Xent Loss
```

### 5.1 Encoder

encoder 类型：ResNet-18（或轻量 CNN）
输出特征维度：512
是否使用预训练权重：否（from scratch）

### 5.2 Projection Head

MLP 层数：2 层
hidden dimension：256
output dimension：128
使用 ReLU + Linear

### 5.3 Linear Probe

encoder 是否冻结：是
linear classifier 输入维度：512
类别数：10

## 6. Loss 实现

请说明 NT-Xent loss 的实现方式：

batch size：32 / 64
构造方式：每张图生成 2 个增强视图 → batch 变为 2N
正样本：同一图像的两个 view
temperature：0.5
logits shape： (2N, 2N)

可以粘贴关键代码片段或伪代码。

## 7. 训练设置

### 7.1 自监督预训练

配置	           数值
train images	   50000
epochs	           2
batch size	       64
optimizer	       Adam
learning rate	   1e-3
temperature        0.5
encoder	           CNN / ResNet-18
device	           CPU

### 7.2 Linear Probe

配置	         数值
train images	 50000
test images 	 10000
epochs	         10
batch size	     256
optimizer	     Adam
learning rate	 1e-3
device	         CPU

## 8. 训练过程

粘贴 contrastive loss 日志或 loss 曲线。

示例：

| Epoch | Contrastive Loss |
|---|---:|
| 1 |  |
| 2 |  |
| 3 |  |

请简要描述 loss 是否下降，以及训练是否稳定：

```text
（在这里填写）
```

## 9. Linear Probe 结果

| 指标 | 结果 |
|---|---:|
| test accuracy |  |
| random baseline | 10% |

请分析结果是否明显高于随机猜测：

```text
（在这里填写）
```

## 10. 预测结果展示

至少展示 3 个测试样例。

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 如果继续改进，可以从哪些方面入手，例如 batch size、epoch、temperature、projection head、数据增强等。

```text
（在这里填写）
```

## 12. AI 对话过程记录

- 录制工具：entire.io
- 对话链接：
- 使用的 AI 模型：gpt
- 累计对话时长 / 会话数：几个小时

简要说明 AI 在哪些环节提供帮助，以及哪些部分是自己独立完成或验证的：

```text
- SimCLR 代码调试
- DataLoader / Windows multiprocessing 问题解决
- 路径错误修复
- linear probe 与 test pipeline 修复
- 实验报告结构整理
```

## 13. Git 提交记录

- 仓库地址：https://github.com/rweiccc/mini-simclr-assignment.git
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```
