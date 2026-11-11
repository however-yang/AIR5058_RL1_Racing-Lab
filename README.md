# MuJoCo 赛车强化学习作业

训练一个连续控制策略，让四轮小车尽量稳定、快速地正向完成一圈。课程赛道约 136.4 米，以上海国际赛车场为灵感，串联收紧的首弯、内场 S 弯、长后直道和发卡弯。请在 `racing/reward.py` 中实现课堂上讨论的奖励，先建立你自己的基线，再在相同的环境交互步数下比较一项改进。

## 安装

在这个目录中执行。Conda 负责 Python 环境，其他依赖随配置安装。

```bash
conda env create -f environment.yml
conda activate mujoco-racing
```

NVIDIA GPU 用户按 PyTorch 安装页为自己的系统选择 CUDA 构建，然后用 `--device cuda` 训练。也可用 `--device cpu`；默认 `auto` 会优先使用可用的 CUDA。MuJoCo 的物理步进在 CPU 上运行，GPU 加速的是策略网络。小网络有时在 CPU 上更快。 [PyTorch 安装页](https://pytorch.org/get-started/locally/)

## 训练、评测和回放

```bash
python train.py --algo ppo --device auto --steps 204800 --seed 7 --out runs/ppo7
python evaluate.py --model runs/ppo7/best.zip --output runs/ppo7/public.json
python evaluate.py --model runs/ppo7/best.zip --camera chase --replay runs/ppo7/lap.gif
```

`reward.py` 中的占位函数只有完赛信号，便于验证命令能运行。正式实验前请自行实现奖励。训练可以用 `--algo sac` 切换算法，评测时传入对应的 `--algo`。输出目录必须是新目录。每次训练会写出配置、最佳验证权重、最后权重、每回合训练记录和验证曲线数据。PPO 会按整批采样结束，默认配置下 204800 步恰好是整批。

用 `--camera chase` 看跟车视角，或用 `--camera overview` 看全景。GIF 回放需要可用的 OpenGL 环境。训练和数值评测不需要打开窗口。

## 接口

- 控制频率 20 Hz。动作 `[steering, speed]` 两个分量都在 `[-1, 1]`。正转向向左，速度指令从 0 到 5 m/s 线性映射。执行器限制加减速和转向速率，车轮用 MuJoCo 接触力带动车身。
- 观测共 20 维。前 10 维是 5 个前视点的车体坐标 `(x, y)` 除以 5 m，其后依次是归一化横向误差、航向误差正弦与余弦、归一化纵向与横向速度、偏航角速度除以 5 rad/s、归一化左右前轮角、归一化已施加的虚拟转角，以及归一化目标速度。
- 车辆须按顺序正向经过 8 道检查线（含终点）。车体或车轮越界、持续超速、翻车或 48 秒内未完成均为失败。评测记录完赛率、平均最大合法进度、完赛车辆中位圈速及失败类型。
- 奖励函数会收到 `RewardSignals`，字段及单位在 `racing/reward.py` 内定义。可改奖励、网络、算法和训练设置。评测会在课程原版的车辆、赛道、观测和判定规则下重新加载策略。

## 提交

提交起步配置和改进配置、训练随机种子及预算、可加载权重、公开评测 JSON、训练曲线和简短分析。解释一项对照改动与一个具体失败回合。统一评分将从独立的固定起步条件重跑策略。

## 评分

| 项目 | 分值 | 判据 |
| --- | ---: | --- |
| 可复现性 | 20 | 权重可加载，配置和复现步骤齐全 |
| 控制表现 | 35 | 完赛率占 20 分，平均进度占 10 分，各自按比例计分。完赛率至少 95% 时，完赛回合中位圈速从 44 秒到 29 秒线性换算 0 到 5 分 |
| 对照实验 | 30 | 同预算对照、清楚说明改动、结果支持结论 |
| 失败分析 | 15 | 能用逐回合结果或回放定位一个失败案例 |

赛道参考 [F1 上海国际赛车场布局](https://www.formula1.com/en/information/china-shanghai-interational-circuit-shanghai.7ESK3ZQ5DHqihx3DpPZKWL)。

物理规则版的策略使用 20 维观测。上海用于训练，斯帕可通过评测的 `--layout spa` 选项复跑。正式评测由统一车辆和赛道配置执行。
