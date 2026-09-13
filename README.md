# Graph Data Attribution Experiments

本仓库包含现有 Pile、CIFAR、In-Run Data Shapley 与 Graph 归因实验代码及论文草稿。各实验的协议与限制以对应目录的报告为准。

## Pile 10B token 四卡入口

```bash
WORK_DIR=/大容量磁盘/pile10b bash pile_10b/run.sh
```

目标为四张各 48GB GPU。需要 Linux、Python 3.10–3.12、可用 NVIDIA 驱动、C++ 编译器、Python venv 和网络。脚本安装独立环境，流式抽取精确 token 预算、构建全局近似图、运行可恢复的分布式归因并汇总结果。

默认沿用 GPT-2 最终 LayerNorm 训练口径，只运行 Graph 与 Adam-aware 一阶的共享轨迹归因。**10B token 不等于整个 Pile；本入口不包含全部基线、二阶 HVP 或子集重训，也不是全参数预训练。**

- [配置、运行、续跑及 GitHub 上传说明](pile_10b/README.md)
- [默认配置](pile_10b/config.json)
- [验证记录与待验证项](pile_10b/VALIDATION.md)
- [原项目迁移与实验进度](PROJECT_MIGRATION_PROMPT.md)

数据、模型权重、缓存和二进制结果不进入 GitHub 源码包，由部署入口在新服务器获取或生成。已有实验的完整数据归档需要单独保管。
