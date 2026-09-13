# 部署入口验证记录

日期：2026-09-13。没有在本机启动 10B 实验，也没有将合成测试的数值计入论文结果。

## 已通过

- `bash -n pile_10b/run.sh`：Shell 语法检查。
- `python -m unittest pile_10b.test_pipeline -v`：7 项通过，包括精确 token 预算、train/query/val 隔离、精确去重、抽取恢复、配置锁、完整覆盖的置换、Graph 与原在线公式对照、真实 HNSW 构图及损坏缓存拒绝。
- `python -m pile_10b.integration_test`：两个 CPU/Gloo 进程完成本地抽取、特征、图、校验、训练、验证与汇总。
- 集成测试通过运行预算保护主动保存检查点，再续跑；与从头连续运行的 Adam-aware/Graph 全部分数及最终 LayerNorm 参数逐位相同。
- 集成测试包含不足整批的末批次与补齐 token 块；数据预算仅 134 个训练 token，模型为随机初始化的单层微型 GPT-2。

最后一次完整测试产物位于原服务器 `/tmp/pile10b-integration-nztrqxme/`，此临时目录不进入 GitHub。测试可通过仓库命令重新生成。

## 尚未验证

- 当前服务器 GPU0 的 `nvidia-smi` 显示 Unknown Error，选择其他 GPU 编号及 UUID 仍无法初始化 CUDA。因此没有声称通过 CUDA/NCCL、四张 48GB GPU 的测试，也未重置 GPU 或修改驱动。
- 本机现有验证环境为 Python 3.8 / PyTorch 2.4.1；测试采用 `SKIP_INSTALL=1`。正式部署固定依赖的全新安装组合（Python 3.10–3.12 / PyTorch 2.6.0）仍需在目标服务器验证。
- 尚未执行远端 10B 全量抽取、10M 级图构建或完整训练；实际耗时、内存峰值、ANN recall 和方法质量必须以目标服务器日志为准。
- 没有创建 GitHub 远端或推送代码；这需要用户的目标仓库地址和已配置的 Git 身份认证。

## 目标服务器验收

先运行 README 中的 preflight，再用独立小预算配置验证四卡。正式目录不要混入测试数据；核对硬件、ANN recall、缓存审计和吞吐估计后再接受长任务。若超过运行预算，检查 `throughput.json` 后使用 `ALLOW_LONG_RUN=1` 续跑。
