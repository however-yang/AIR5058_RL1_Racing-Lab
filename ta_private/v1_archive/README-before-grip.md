# TA 内部资料

给学生只发项目根目录的 `mujoco-racing-student.zip`。此目录里的奖励、训练种子、验证及留出起点、权重和结果用于 TA 参考。

## 课堂口述提纲

- 每个动作之后量一下沿赛道前进了多少。往终点方向走可以加分，逆向时应反映方向。原地停留也可以有代价。
- 出界应受到明确惩罚，完成一圈应得到奖励。可以用横向偏差、航向偏差、转向变化等信号帮助探索。
- 奖励需要兼顾速度和稳定性。过重的惩罚可能让车停着，过重的速度项可能让车在弯道冲出去。请观察实际回放与完赛率。
- 学生根据以上思路自己选择函数与权重，记录选择理由，再在相同环境交互步数下做对照。

## 本机复现

从项目根目录运行。Conda 环境前缀是 `.conda`，可运行 `conda activate ./.conda` 后使用 `python`。

```bash
python ta_private/check_env.py
python ta_private/reference.py --algo ppo --device auto --seed 7 --out ta_private/runs/new_ppo
python ta_private/reference.py --algo ppo --holdout ta_private/best/reference_policy.zip --out ta_private/best/recheck.json
```

重测时将最后一条命令的算法替换为 `best/results.json` 里记录的获选算法。训练默认 204800 步，输出目录必须是新目录。只按验证集选模，选定后再查看留出评测。留出评测仍用同一条赛道，仅改变起步位姿。

## 评分建议

| 项目 | 分数 | 判据 |
| --- | ---: | --- |
| 可复现性 | 20 | 配置完整、策略可加载 |
| 表现 | 35 | 20 × 完赛率，10 × 平均进度。完赛率达 80% 后，可按 44 秒至 29 秒线性换算 0 至 5 分圈速分 |
| 对照实验 | 30 | 同预算、清楚说明改动、证据与结论一致 |
| 失败分析 | 15 | 能用逐回合结果或回放定位一个失败案例 |

## 上海风格课程赛道

长度约 136.4 米，回合上限 48 秒。路线取自上海赛道的设计特征，收紧的首弯接反向出弯，内场有连续 S 弯，长后直道末端接发卡弯。弯道半径和路段间距按作业小车调整。学生包提供斜视全景和跟车镜头，车身装饰和赛道设施是可视化几何。

旧赛道的学生包和 TA 结果单独存放在 `v1_archive/`，复现旧权重时请用归档里的旧赛道。

新版完整回放可用项目的 Conda 环境运行。以下命令会打开 MuJoCo 窗口并循环展示单圈。

```bash
python .planning/demo_lap.py --live --camera chase
```

将 `chase` 改成 `overview` 可切换到斜视全景。

参考来源 [F1 上海赛道图与弯道说明](https://www.formula1.com/en/information/china-shanghai-interational-circuit-shanghai.7ESK3ZQ5DHqihx3DpPZKWL)。

## 新版参考成绩

在 80 个独立验证起点上，从训练与油门标定候选中按完赛率、平均进度、圈速选模。获选的是训练种子 7 的 PPO 检查点，对速度输出做了小幅收缩。对应的源权重和标定系数保存在评测 JSON 中。验证集 80/80 完赛，中位 31.28 秒。

| 40 个留出起点 | 完赛 | 完赛回合中位圈速 |
| --- | ---: | ---: |
| TA PPO | 39/40 | 31.26 秒 |
| 弯前减速循迹 | 40/40 | 33.37 秒 |
| 4.5 m/s 固定目标速度循迹 | 39/40 | 31.92 秒 |
| 5.0 m/s 固定目标速度循迹 | 26/40 | 29.47 秒 |

`best/results.json` 包含逐回合数据、未获选候选和基线结果。标定过程记录在 `.planning/shanghai_calibrate.py`，学生发布包不含这部分。

## 斯帕风格试跑

平面教学版约 115.3 米。保留 La Source、Eau Rouge–Raidillon 的平面变向、Kemmel 直道和 Pouhon 长弯，压缩急弯组合。原上海 PPO 权重直接用于评测。公开起点 20/20 完赛，中位圈速 26.565 秒，逐回合记录见 `spa_transfer.json`。

```bash
python .planning/demo_lap.py --live --camera sideline --layout spa
python student/evaluate.py --model ta_private/best/reference_policy.zip --layout spa
```

[F1 官方斯帕布局与赛道说明](https://www.formula1.com/en/information/belgium-circuit-de-spa-francorchamps.3LltuYaAXVRU8iezEsjzGw)
