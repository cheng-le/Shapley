# MNIST + MLP In-Run Data Shapley 实验

这个文件夹保存原来的 MNIST + 两层 MLP 复现实验代码和已生成结果。`mnist_mlp_common.py` 保存 MNIST 数据集、MLP 模型、MNIST 参数和图片显示；真正通用的 Shapley 公式、MC fidelity、drop/keep 和画图框架在根目录 `shapley_common.py`。

## 文件结构

- `first_order/`：只计算一阶近似的 SGD、Adam-aware、Adam-aware + Graph。
- `second_order/`：融合二阶计算/二阶代理的 SGD、Adam-aware、Adam-aware + Graph。
- `mnist_mlp_common.py`：MNIST 数据集 + MLP + MNIST 专用参数/可视化。
- `legacy/`：项目早期的单文件 MNIST 实现，保留用于对照。

## 数据路径

MNIST 数据按数据集整理在：

```text
data/MNIST/raw/
```

## 运行示例

```bash
python mnist_mlp/first_order/in_run_sgd_first_order.py
python mnist_mlp/second_order/in_run_adam_graph_second_order.py
```

默认输出也会保存到 `mnist_mlp/first_order/` 或 `mnist_mlp/second_order/` 下。
