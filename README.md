# AIR5058 RL1 · MuJoCo 赛车实验

用强化学习训练一辆四轮小车，在统一的车辆和赛道规则下尽快完成一圈。你需要实现奖励，训练一个可复现的基线，再用相同的环境交互预算验证一项改进。

## 安装

克隆仓库或解压课程 ZIP，在含有 `environment.yml` 的目录运行。

```bash
conda env create -f environment.yml
conda activate mujoco-racing
```

`--device auto` 会在可用时选择 CUDA，否则使用 CPU。若显式使用 `--device cuda`，先安装适合本机的 CUDA 版 PyTorch。MuJoCo 物理在 CPU 上运行，GPU 用于策略网络。 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)

## 跑通第一个实验

1. 先检查训练与保存能正常运行。下列命令跑一个 PPO 采样批次，使用起步奖励。

   ```bash
   python train.py --algo ppo --device cpu --envs 1 --steps 512 --eval-every 512 --out runs/smoke
   ```

2. 在 [`racing/reward.py`](racing/reward.py) 实现自己的奖励，再训练基线。起步奖励只在完赛时给分。

   ```bash
   python train.py --algo ppo --device auto --steps 204800 --seed 7 --out runs/ppo7
   ```

3. 用公开评测入口生成逐回合数据和回放。

   ```bash
   python evaluate.py --algo ppo --model runs/ppo7/best.zip --layout shanghai --output runs/ppo7/shanghai.json
   python evaluate.py --algo ppo --model runs/ppo7/best.zip --layout spa --output runs/ppo7/spa.json
   python evaluate.py --algo ppo --model runs/ppo7/best.zip --layout shanghai --camera chase --replay runs/ppo7/lap.gif
   ```

| 选项 | 用法 |
| --- | --- |
| PPO 或 SAC | 训练用 `--algo ppo` 或 `--algo sac`，评测传入相同的算法 |
| 上海或斯帕 | 上海用于训练和公开验证。用 `--layout spa` 检查跨赛道表现 |
| 并行环境 | 用 `--envs` 设置数量，加 `--parallel` 可在多个进程中运行物理 |
| 回放视角 | `--camera overview` 看全景，`--camera chase` 看跟车。GIF 需要可用的 OpenGL 环境 |

每次训练使用新的 `--out` 目录。目录内有 `config.json`、`runtime.json`、`validation.json`、训练回合日志、`best.zip` 和 `final.zip`。PPO 按整批采样，做等预算对照时以 `runtime.json` 记录的实际步数为准。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| [`train.py`](train.py) | PPO、SAC 训练与验证，保存最佳及最终权重 |
| [`evaluate.py`](evaluate.py) | 固定起点评测、圈速统计和 GIF 回放 |
| [`racing/reward.py`](racing/reward.py) | 作业的奖励入口和各信号的单位 |
| [`racing/env.py`](racing/env.py) | Gymnasium 环境、观测和比赛判定 |
| [`racing/vehicle.py`](racing/vehicle.py) 与 [`racing/car.xml`](racing/car.xml) | 车辆模型、转向、驱动和轮地接触 |
| [`racing/track.py`](racing/track.py) 与 [`racing/visuals.py`](racing/visuals.py) | 赛道几何、检查线和场景显示 |

## 比赛规则与接口

- 策略每 50 ms 输出一次动作 `[steering, speed]`，两个分量都在 `[-1, 1]`。正转向向左，速度指令线性映射到 0–5 m/s。车辆实际速度由执行器与轮地接触决定。
- 观测共 20 维。前 10 维是 5 个前视点在车体坐标系中的坐标除以 5 m。后 10 维依次是归一化横向误差、航向误差正弦和余弦、归一化纵向和横向速度、偏航角速度除以 5 rad/s、归一化左右前轮角、归一化虚拟转角和归一化目标速度。
- 车辆须按顺序正向通过 8 道检查线（含终点）。车体或车轮越界、持续超速、翻车或 48 秒内未完成均为失败。圈速统计使用成功完赛的回合。
- 你可以修改奖励、网络、算法和训练设置。正式评分会用课程原版的车辆、赛道、观测和判定规则重新加载策略。

赛道分别以上海国际赛车场和斯帕赛道为灵感。上海场景约 136.4 m，包含收紧首弯、内场 S 弯、长后直道和发卡弯。 [上海赛道参考](https://www.formula1.com/en/information/china-shanghai-interational-circuit-shanghai.7ESK3ZQ5DHqihx3DpPZKWL)

## 提交与评分

提交基线和改进版的配置、训练种子、实际交互步数、可加载权重、公开评测 JSON、训练曲线与简短分析。解释一项对照改动和一个具体失败回合。

| 项目 | 分值 | 判据 |
| --- | ---: | --- |
| 可复现性 | 20 | 权重可加载，配置与复现步骤齐全 |
| 控制表现 | 35 | 完赛率 20 分，平均最大合法进度 10 分。完赛率达到 95% 时，中位圈速从 44 秒到 29 秒线性换算 0–5 分 |
| 对照实验 | 30 | 训练预算相同，改动明确，结果支持结论 |
| 失败分析 | 15 | 能用逐回合数据或回放定位失败案例 |
