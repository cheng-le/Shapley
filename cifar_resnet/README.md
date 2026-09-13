# CIFAR10 + ResNet18 In-Run Data Shapley 实验

这个文件夹是在当前 MNIST+MLP 代码之外新增的一套 CIFAR10+ResNet18 版本，用来更接近 `Data Shapley in One Training Run` 论文附录里的小规模对比设置。

## 文件结构

- `cifar_resnet_common.py`：CIFAR10 读取器、CIFAR 版 ResNet18、CIFAR 专用参数/可视化和六种打分算法。通用 Shapley 工具来自根目录 `shapley_common.py`。
- `first_order/in_run_sgd_first_order.py`：一阶 SGD。
- `first_order/in_run_adam_first_order.py`：一阶 Adam-aware。
- `first_order/in_run_adam_graph_first_order.py`：一阶 Adam-aware + Graph。
- `second_order/in_run_sgd_second_order.py`：二阶 SGD。
- `second_order/in_run_adam_second_order.py`：二阶 Adam-aware。
- `second_order/in_run_adam_graph_second_order.py`：二阶 Adam-aware + Graph，并在一张图里对比 SGD、Adam-aware、Graph。

## 数据准备

代码不依赖 torchvision，直接读取官方 CIFAR10 python pickle 格式。数据应位于：

```text
data/CIFAR10/cifar-10-batches-py/
```

如果本地没有数据，可以运行脚本时加：

```bash
--download
```

它会从 Toronto 官网下载 `cifar-10-python.tar.gz` 并解压到 `data/CIFAR10/`。如果官网临时返回 502，可以手动下载同名压缩包并解压到上述目录。

也可以指定自定义下载地址，例如学校镜像或你自己放在服务器上的 tar.gz：

```bash
python cifar_resnet/first_order/in_run_sgd_first_order.py --download --cifar-url <你的 CIFAR10 tar.gz 地址>
```

注意：pip/conda 的清华源只影响 Python 包安装，不会影响这里的 CIFAR10 数据下载。

## 运行示例

论文对齐默认版，使用 1000 个训练样本和完整 CIFAR10 test set：

```bash
python cifar_resnet/second_order/in_run_adam_graph_second_order.py --download --train-limit 1000 --val-limit 10000 --hidden-dim 64 --batch-size 32
```

更轻量的快速检查：

```bash
python cifar_resnet/first_order/in_run_sgd_first_order.py --download --train-limit 128 --val-limit 128 --hidden-dim 16 --batch-size 8 --mc-permutations 2 --prune-epochs 1
```

## 与论文设置的对应关系

论文 `Data Shapley in One Training Run` 的主实验是 Pile + GPT2/Pythia；附录的小规模 data attribution benchmark 使用 CIFAR10 的 1000 样本子集和 ImageNet-pretrained ResNet18。这里当前实现为 CIFAR10 + torchvision ResNet18，默认加载 ImageNet 预训练权重；如需随机初始化，可加 --no-pretrained。Table 5 的 data selection 训练轮数为 50 epochs，因此 CIFAR 默认 --prune-epochs 也设为 50。

因此当前代码适合做公式和方法趋势对比；若要严格复现论文附录数值，需要确保 ImageNet-pretrained ResNet18 权重可加载，并按论文设置使用 Adam、lr=0.001、weight_decay=1e-4、label_smoothing=0.1、50 epochs 和对应的 corruption/data-selection protocol。

## 计时解释

Graph 图中的 `Online pass time` 包含 raw 一阶 per-sample 梯度、图/二阶项和训练更新；`Second-order component only` 对 SGD/Adam 是 HVP 二阶项，对 Graph 是拓扑邻居距离代理，因此更能体现 graph 二阶计算本身的效率。

## Common 拆分说明

现在的依赖关系是：

```text
shapley_common.py               # 数据集/模型无关的通用 Shapley 工具
mnist_mlp/mnist_mlp_common.py   # MNIST + MLP 专用
cifar_resnet/cifar_resnet_common.py # CIFAR10 + ResNet18 专用
```

CIFAR 代码只复用 `shapley_common.py` 中的通用函数，不会使用 MNIST/MLP 专用函数。

## ImageNet 预训练 ResNet18

当前默认使用 `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`，并将最后的 `fc` 层替换为 10 类 CIFAR10 分类头。第一次运行时 torchvision 会从 PyTorch 官方地址下载预训练权重到本机缓存。

如果想退回随机初始化，可加：

```bash
--no-pretrained
```

注意：`--hidden-dim` 现在只是兼容旧脚本的占位参数；torchvision 标准 ResNet18 使用固定宽度。

## 二阶 HVP 曲率项缩放

在 ImageNet-pretrained ResNet18 上，Adam 单样本方向和 HVP 二阶项可能非常大，直接使用未缩放二阶 Taylor 项会让曲率噪声压过一阶贡献，导致 MC fidelity 变差甚至负相关。代码默认使用：

```bash
--curvature-clip-ratio 0.25
```

含义是把二阶曲率项的 batch 内标准差限制到一阶项标准差的 0.25 倍。若要运行完全未缩放的原始二阶公式，可设置：

```bash
--curvature-clip-ratio -1
```

若要退化为一阶近似，可设置：

```bash
--curvature-clip-ratio 0
```
