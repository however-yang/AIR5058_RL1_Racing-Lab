# TA 训练与复现实验

- [当前设计与规则](DESIGN.md) 说明物理、走线、奖励和控制方式。
- [版本对照与逐圈结果](RESULTS.md) 汇总不同批次及各批次可比较的结果。
- [开发历史与旧入口](HISTORY.md) 保存早期配置、验证命令与研究笔记。

## 代码职责

| 模块 | 职责 |
| --- | --- |
| [`reference.py`](reference.py)、[`train_racing_line.py`](train_racing_line.py) | 基础奖励、参考 PPO、驾驶示范与分段起步 |
| [`calibrate_vehicle.py`](calibrate_vehicle.py)、[`calibrate_pace.py`](calibrate_pace.py) | 车辆标定、实测纵向加减速和上海规划候选筛选 |
| [`time_line.py`](time_line.py)、[`planned_racer.py`](planned_racer.py) | 合法车道内的走线与速度规划、参考线跟踪和可选横向目标 |
| [`multi_line_potential.py`](multi_line_potential.py)、[`racecraft_reward.py`](racecraft_reward.py) | 可行候选线的剩余时间估计、弯后直道权重和轮胎风险项 |
| [`train_remaining_time.py`](train_remaining_time.py)、[`train_lateral_target.py`](train_lateral_target.py) | SAC 奖励对照、横向目标与固定线续训 |
| [`experiments/evaluation.py`](experiments/evaluation.py) | 共用完赛门槛、计时线、配对统计、冻结权重与赛道校验 |
| [`compare_measured_pace.py`](compare_measured_pace.py)、[`compare_multi_line_sac.py`](compare_multi_line_sac.py)、[`compare_lateral_target.py`](compare_lateral_target.py)、[`compare_racecraft.py`](compare_racecraft.py) | 各轮上海选模及斯帕复测 |
| [`replay_measured_pace.py`](replay_measured_pace.py) | 从冻结结果重放指定策略，核对圈速及固定赛段并输出图表 |

各轮 `results.json` 保存逐圈时间、完赛状态、分段和训练元数据，`frozen_selection.json` 记录上海选中的配置、两类起点和哈希值。评测先核对车辆、环境和赛道是否与学生 ZIP 一致。已有实验入口和数据路径可继续使用。

## 安装和重放

GitHub 版提供源码、赛道参考线、训练日志和评测记录。策略权重由本地训练生成。使用已有冻结记录重放时，需在记录指向的位置提供对应权重。

先按[学生作业说明](../student/README.md)创建 Conda 环境，再从项目根目录安装 TA 依赖并运行回放。

```bash
conda activate mujoco-racing
python -m pip install -r ta_private/requirements.txt
python ta_private/replay_measured_pace.py ta_private/racecraft_comparison_repro --gif spa-fastest.gif --plot spa-fastest.png
```

## 复现同预算奖励对照

从项目根目录运行。两组各用 8 个上海环境、训练种子 101 和 196,608 步。GitHub 克隆后先生成两组权重。下面的对照命令使用已归档的赛道几何和哈希，并跳过需要旧权重的固定线参照。

```bash
python ta_private/train_remaining_time.py --multi-line --line ta_private/runs/measured_pace_planners/shanghai_line.npz --out ta_private/runs/multi_line_repro --steps 196608 --envs 8 --seed 101
python ta_private/train_remaining_time.py --racecraft --line ta_private/runs/measured_pace_planners/shanghai_line.npz --out ta_private/runs/racecraft_repro --steps 196608 --envs 8 --seed 101
python ta_private/compare_racecraft.py --original ta_private/runs/multi_line_repro --racecraft ta_private/runs/racecraft_repro --previous ta_private/lateral_target_comparison --skip-fixed-benchmark --out ta_private/racecraft_comparison_repro
```

训练和比较使用新输出目录。新跑的对照会保存获准转移的权重，供上方回放命令读取。要重新标定参考线，运行 `python ta_private/calibrate_pace.py ta_private/runs/pace_repro`，并在两组训练命令中传入同一份新线。已有对照会校验参考线与上一轮基准的哈希，重新标定的线需要先独立验证。
