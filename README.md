# FedNSAM

这个仓库已经裁剪为一个只保留核心 `FedNSAM` 算法的最小实现。
在精简时尽量保留了旧版的训练意图，例如：

- 客户端数据量默认保持基本均衡
- 每轮先做全局 Nesterov 外推，再做局部 SAM
- 仍然使用旧版脚本风格的余弦学习率

同时修复了旧版里会改变训练语义的明显错误，例如服务器动量更新和数据切分中的重复采样问题。

## 现在保留的内容

- `main_FedNSAM.py`: 干净的训练入口
- `fednsam.py`: `FedNSAM` 核心训练循环
- `sam.py`: SAM 优化器
- `dirichlet_data.py`: Dirichlet 非 IID 客户端划分
- `models/resnet.py`: CIFAR 用的最小 `ResNet-18`
- `models/cnn.py`: EMNIST 用的最小 CNN

## 已移除的无关内容

- 其他联邦算法分支
- Ray 并行执行逻辑
- LoRA / PEFT
- ViT / Swin / DeiT / LLM 相关代码
- 多余优化器实现
- Tiny-ImageNet / GLUE / RoBERTa 相关代码
- 日志、绘图、示例权重和实验残留文件

## 运行

```bash
python main_FedNSAM.py \
  --algorithm fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.1
```

## 对比 FedAvg / FedSAM / FedNSAM

单独跑某一个算法：

```bash
python main_FedNSAM.py --algorithm fedavg  --dataset cifar100 --rounds 300 --num-clients 100 --client-fraction 0.1 --local-epochs 5 --local-steps 50 --batch-size 50 --lr 0.1 --alpha 0.1
python main_FedNSAM.py --algorithm fedsam  --dataset cifar100 --rounds 300 --num-clients 100 --client-fraction 0.1 --local-epochs 5 --local-steps 50 --batch-size 50 --lr 0.1 --rho 0.05 --alpha 0.1
python main_FedNSAM.py --algorithm fednsam --dataset cifar100 --rounds 300 --num-clients 100 --client-fraction 0.1 --local-epochs 5 --local-steps 50 --batch-size 50 --lr 0.1 --rho 0.05 --gamma 0.85 --alpha 0.1
```

一次性公平对比三个算法：

```bash
python main_FedNSAM.py \
  --compare fedavg fedsam fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.1 \
  --save-json results/cifar100_compare.json
```

对比模式下会复用：

- 同一份客户端划分
- 同一份初始模型
- 同一组每轮客户端采样顺序

这样 `FedAvg`、`FedSAM`、`FedNSAM` 的差异主要来自算法本身，而不是随机性。

## EMNIST 支持

现在 `--dataset` 也支持 `emnist`。

- 默认采用 `EMNIST(split='byclass')`
- 类别数固定为 `62`
- 默认模型使用 `DP-FedSAM` 风格的 `cnn_emnist`
- 训练和测试都使用 `DP-FedSAM` 风格的 EMNIST 变换：`RandomCrop(32, padding=4) + HorizontalFlip + VerticalFlip + RandomGrayscale + ToTensor`

如果你想跑一条尽量贴近 `DP-FedSAM` README 的 EMNIST 命令，可以直接用：

```bash
python main_FedNSAM.py \
  --algorithm fedsam \
  --dataset emnist \
  --dp \
  --dp-clip 0.2 \
  --sigma 0.95
```

如果你想让 `FedNSAM` 的服务器动量系数 `gamma` 在第 `75` 轮开始直接归零，可以加：

```bash
python main_FedNSAM.py \
  --algorithm fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --gamma-zero-round 75 \
  --restart-factor 1.5 \
  --lr-decay 0.998 \
  --alpha 0.1
```

这时前 `74` 轮保持原始 `gamma`，从第 `75` 轮起 `gamma = 0`。
第 `75` 轮学习率会重启为 `original_lr(75) * 1.5`，第 `76` 轮起不再使用 `original_lr(t) * 1.5`，而是以重启学习率为起点继续乘 `lr_decay` 做指数衰减。

这条命令在没有额外显式覆盖时，会自动解析成更贴近 `DP-FedSAM` 的 EMNIST 默认配置，例如：

- `rounds=200`
- `num_clients=500`
- `client_fraction=0.1`
- `local_epochs=30`
- `batch_size=32`
- `lr_decay=0.998`
- `momentum=0.5`
- `rho=0.5`
- `alpha=0.6`

如果你有两张 GPU，可以显式打开算法级并行 compare：

```bash
python main_FedNSAM.py \
  --compare fedavg fedsam fednsam \
  --compare-parallel \
  --devices cuda:0 cuda:1 \
  --dataset cifar10 \
  --rounds 100 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 1 \
  --local-steps 20 \
  --batch-size 64 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.5 \
  --device cuda \
  --fast-cuda \
  --amp auto \
  --save-json results/cifar10_compare_parallel.json
```

这一模式会把不同算法分配到不同设备上并行运行；如果只给了一个设备，或者启用了 `--ckpt-dir` / `--resume`，程序会自动回退到顺序 compare。

## 公平对比与随机种子

现在 `--seed` 会统一控制公平对比所需的整套随机上下文：

- 数据划分
- 初始模型权重
- 每轮客户端采样
- 每轮本地 DataLoader 打乱
- DP 噪声种子

这意味着即使你不使用 `--compare`，只要不同算法使用相同的 `--seed` 和相同的数据/训练配置，它们也会共享同一份公平对比上下文。

保存的 JSON 结果里会额外写入顶层 `_meta`，包含：

- `seed`
- `fairness_mode`
- `context_hash`
- `command`
- `log_path`

你可以直接对比不同实验文件里的 `_meta.context_hash`，快速判断它们是不是建立在同一份公平上下文上。

如果你传了 `--save-json results/xxx.json`，程序还会自动把终端命令和训练输出一起保存到同目录下的 `results/xxx.log`。命令会按多行续行格式写入，方便直接复制修改。

## Client-level DP

当前 `--dp` 路径已经按 `DP-FedSAM` 的服务器端客户端级 DP 语义重构：

- 每个客户端本地训练完成后，服务器计算客户端更新
- 对每个浮点参数张量分别做 `L2` 裁剪
- 对每个张量分别立刻加高斯噪声
- 再对这些“已裁剪且已加噪”的客户端更新按样本数加权聚合
- 隐私会计复用了 `DP-FedSAM/eps_computer.py` 的 RDP 实现
- 默认局部 SAM 使用命令行给定的 `--rho`，并固定普通 SAM 扰动，即 `adaptive=False`
- 如果显式启用 `--rho-mode dp_algorithm`，则 DP 下 `fedsam` 使用 `rho=0.5, adaptive=True`，`fednsam` 使用 `rho=0.1, adaptive=False`

DP 模式下如果你没有显式传入关键训练超参，会自动切到 `DP-FedSAM` 的论文默认值：

- `rounds=300`
- `num_clients=500`
- `client_fraction=0.1`
- `local_epochs=30`
- `batch_size=50`
- `rho=0.5`
- `momentum=0.5`
- `weight_decay=5e-4`
- `lr_decay=0.998`
- `alpha=0.6`

学习率调度现在也支持显式切换：

- 默认 `--lr-schedule auto`：非 DP 走余弦，DP 走 `DP-FedSAM` 风格指数衰减
- 如果你想让 DP 也用余弦学习率，可以显式传 `--lr-schedule cosine`

同时还有两条 DP 专属规则：

- `local_steps` 默认不生效，按完整 `local_epochs` 训练；只有显式传了 `--local-steps` 才会截断
- `grad-clip` 默认关闭；只有显式传了 `--grad-clip` 才会启用

如果你想让 DP 裁剪阈值按轮逐步变小，也可以开启公开、确定性的指数日程：

- `--dp-clip-decay` 控制衰减系数 `lambda`
- `--dp-clip-min` 控制裁剪阈值下界 `C_min`
- 实际使用的当轮阈值为 `C_t = max(C_min, C0 * lambda^t)`

这条 schedule 只会改变每轮的裁剪强度和绝对噪声大小；当前隐私会计仍然只依赖 `sigma / q / steps / delta`，所以在固定这些量时，`epsilon` 不会因为 `dp-clip-decay` 而改变。

## DP-aware Consistency-Gated FedNSAM

在 DP 噪声较强或训练后期，朴素固定 `gamma` 的 DP-FedNSAM 可能被 DP-FedSAM 反超。一个常见原因是历史 server momentum 与当前聚合更新方向不再一致，但固定 `gamma` 仍然继续做同等强度的 Nesterov 外推。

现在可以用 `--gamma-strategy cosine_gate` 启用 consistency-gated gamma。它会在每轮聚合出 `avg_delta` 后，计算当前聚合更新和本轮开始时 `global_momentum` 的 cosine similarity：

```text
cos_t = <avg_delta, global_momentum> / (||avg_delta|| * ||global_momentum|| + 1e-12)
gamma_next = max(gamma_min, gamma * max(0, cos_t))
```

如果历史 `global_momentum` 的范数为 `0`，说明还没有可用的一致性信号，此时 `gamma_next` 会保持为基础值 `gamma`，避免第一轮后直接把 `gamma` 关掉。

当前轮仍使用当前生效的 `gamma_t`；`gamma_next` 从下一轮开始生效。默认 `gamma_strategy=fixed`，所以不传新参数时行为不变。`cosine_gate` 模式下会忽略已有的 `gamma_zero_round` / restart 逻辑，建议不要和这两个实验开关混用。

可直接运行：

```bash
python main_FedNSAM.py \
  --algorithm fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --gamma-strategy cosine_gate \
  --gamma-min 0.0 \
  --alpha 0.1 \
  --dp \
  --dp-clip 0.2 \
  --sigma 0.95 \
  --delta 2e-5 \
  --save-json results/cifar100_fednsam_cosine_gate_dp.json
```

最接近 `DP-FedSAM` 原论文的 CIFAR-10 命令可以直接写成：

```bash
python main_FedNSAM.py \
  --algorithm fedsam \
  --dataset cifar10 \
  --dp \
  --dp-clip 0.2 \
  --sigma 0.95
```

上面这条命令没有额外显式传参时，会自动解析成接近论文的训练配置。

如果你想把所有关键项都写全，等价的 paper-aligned 命令是：

```bash
python main_FedNSAM.py \
  --algorithm fedsam \
  --dataset cifar10 \
  --rounds 300 \
  --num-clients 500 \
  --client-fraction 0.1 \
  --local-epochs 30 \
  --batch-size 50 \
  --lr 0.1 \
  --lr-decay 0.998 \
  --momentum 0.5 \
  --weight-decay 5e-4 \
  --rho 0.5 \
  --alpha 0.6 \
  --dp \
  --dp-clip 0.2 \
  --sigma 0.95
```

如果你希望在 DP 下改用余弦学习率，可以在上面的命令后追加：

```bash
  --lr-schedule cosine
```

如果你希望前期阈值大一点、后期阈值小一点，可以额外加上：

```bash
  --dp-clip-decay 0.998 \
  --dp-clip-min 0.05
```

第一版建议优先试：

- `--dp-clip-decay 0.995`
- `--dp-clip-decay 0.998`
- `--dp-clip-decay 0.999`
- `--dp-clip-min 0.05`
- `--dp-clip-min 0.1`
- `--dp-clip-min 0.2`

直接指定 DP 裁剪阈值和噪声倍率：

```bash
python main_FedNSAM.py \
  --compare fedavg fedsam fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.1 \
  --dp \
  --dp-clip 0.2 \
  --sigma 0.95 \
  --delta 2e-5
```

或者给定目标隐私预算，由程序在训练前反推 `sigma`：

```bash
python main_FedNSAM.py \
  --algorithm fednsam \
  --dataset cifar100 \
  --rounds 300 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 5 \
  --local-steps 50 \
  --batch-size 50 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.1 \
  --dp \
  --dp-clip 0.2 \
  --eps 8.0
```

启用 DP 后，会在每个评估点记录对应的 `epsilon`，并一起写入保存的 JSON 结果。
另外，DP 模式下的 `test_acc / test_loss` 与 `DP-FedSAM` 一样，使用“各客户端测试集指标的简单平均”，不是全局 test loader 的单次评估。

## 结果可视化

训练完成后，可以直接把保存的结果 JSON 画成准确率曲线：

```bash
python plot_results.py results/cifar100_compare.json
python plot_results.py results/cifar10_compare_dp.json --show
```

默认会把图片保存到 `figures/`，输出文件名形如 `figures/cifar100_compare_accuracy.png`。
如果传入多个 JSON，默认会把它们合并到同一张图：

```bash
python plot_results.py \
  results/cifar100_compare.json \
  results/cifar10_compare_dp.json \
  --output-name cifar_compare_all.png
```

如果你仍然想像旧行为一样每个 JSON 单独输出一张图，可以加 `--separate`。

## 单卡提速

如果你只有一张 GPU，可以先打开这两个开关：

```bash
python main_FedNSAM.py \
  --compare fedavg fedsam fednsam \
  --dataset cifar100 \
  --rounds 100 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 1 \
  --local-steps 5 \
  --batch-size 128 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.6 \
  --device cuda \
  --fast-cuda \
  --amp auto
```

- `--fast-cuda`: 打开 `cudnn.benchmark`、TF32，并让卷积走 `channels_last`
- `--amp auto`: CUDA 上优先用 `bf16`，否则自动回退到 `fp16`

## Checkpoint 与恢复

如果训练容易被中断，可以给实验加一个滚动 checkpoint 目录：

```bash
python main_FedNSAM.py \
  --compare fedavg fedsam fednsam \
  --dataset cifar10 \
  --rounds 100 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --local-epochs 1 \
  --local-steps 20 \
  --batch-size 64 \
  --lr 0.1 \
  --rho 0.05 \
  --gamma 0.85 \
  --alpha 0.5 \
  --device cuda \
  --fast-cuda \
  --amp auto \
  --ckpt-dir checkpoints/cifar10_compare \
  --save-json results/cifar10_compare.json
```

启用 `--ckpt-dir` 后：

- 每轮会覆盖保存 `latest.pt`
- 每个评估轮会原子刷新一次 `--save-json`
- 如果训练在某一轮中断，最多只会丢失这一轮

恢复时直接从 `latest.pt` 继续：

```bash
python main_FedNSAM.py \
  --resume checkpoints/cifar10_compare/latest.pt \
  --device cuda \
  --fast-cuda \
  --amp auto
```

恢复时会沿用 checkpoint 里的训练配置、数据划分、DP 配置和对比算法顺序；你只需要按需覆盖设备和输出相关参数。

## FedNSAM 主流程

1. 服务器维护全局模型 `w_t` 和全局动量 `m_t`
2. 每轮先做 Nesterov 外推：`w_t + gamma * m_t`
3. 客户端从外推点出发执行局部 SAM 更新
4. 服务器聚合客户端更新，得到平均增量
5. 用平均增量更新全局动量，再更新全局模型

这版实现默认只支持 `CIFAR10/CIFAR100`，目的是把算法本身暴露清楚，而不是继续保留大量实验分支。
