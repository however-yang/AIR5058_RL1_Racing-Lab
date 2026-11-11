# TA 实验开发历史

当前入口见 [TA 训练与复现实验](README.md)。下文保留每轮开发时的配置、命令与观察。

学生只需收到项目根目录的 `mujoco-racing-student.zip`。新版包含统一车辆、上海和斯帕、训练及评测入口。私有奖励实现、策略权重和留出种子保存在本目录。

## 冻结规则与训练

先运行物理和比赛检查，再开始 PPO。环境每 10 ms 计算轮胎接触，每 50 ms 更新策略指令。训练和选模使用上海。锁定权重后，用相同权重在 40 个斯帕起点复跑。

```bash
conda activate ./.conda
python ta_private/calibrate_vehicle.py
python ta_private/reference.py --profile progress --device cpu --seed 7 --out ta_private/runs/new_progress
python ta_private/reference.py --profile racing --device cpu --seed 7 --out ta_private/runs/new_racing
python ta_private/reference.py --holdout ta_private/best/reference_policy.zip --validation-layout spa --out ta_private/new_spa_check.json
python scripts/demo_lap.py --live --camera sideline --layout spa
```

本版观测为 20 维，包括左右前轮实际角度、虚拟转向参考和速度参考。旧版 18 维权重可在归档的旧物理环境中复跑。详见 [车辆物理与比赛规则](车辆物理与比赛规则.md) 和 `physics_calibration.json`。

## 课堂口述提纲

- 给首次达到的合法进度加分，逆向扣分，记录用时、完赛和失败。
- 让学生观察同一弯中不同入弯速度导致的轨迹差异，思考怎样利用车道宽度、何时刹车以及何时加速。
- 可以让学生把进度奖励与贴边或抓地风险奖励做对照。发布包提供信号与占位函数，奖励权重由学生选择并解释。
- 用统一物理版本和相同交互步数比较结果。评分依据为可复现性、控制表现、对照实验和失败分析。圈速分要求至少 95% 完赛率。

## TA 奖励候选

A 组奖励首次到达的合法进度，逆向扣分，并加入时间、完赛、失败与超时项。B 组在相同基础上增加贴边和高抓地利用率的风险代价。B 组在车辆净空低于 0.15 m 时增加贴边代价，在轮胎利用率超过 0.85 时增加抓地代价。车轮接触力决定可用抓地，两组均允许使用整条合法车道。

## 首轮物理规则版对照

两个奖励均使用训练种子 7、13、23。共同预算为 81,920 步。种子 7 的两组另外延长到 204,800 步。表中为共同预算时的上海验证结果，每格为完赛数和成功回合中位圈速。

| 种子 | A 进度与终局 | B 加贴边和抓地风险 |
| --- | --- | --- |
| 7 | 20/20，32.72 秒 | 19/20，32.97 秒 |
| 13 | 4/20，32.58 秒 | 15/20，32.60 秒 |
| 23 | 4/20，33.34 秒 | 20/20，32.78 秒 |

按上海验证完赛率、进度和中位圈速选出的参考权重是 A 组种子 7 的第 122,880 步检查点，上海验证 20/20 完赛，中位 32.37 秒。随后锁定权重并分别进行留出评测。

| 留出赛道 | 完赛 | 成功回合中位圈速 |
| --- | --- | --- |
| 上海 | 39/40 | 32.42 秒 |
| 斯帕 | 34/40 | 26.65 秒 |

斯帕的 6 次失败均为第 3 至第 4 检查点之间出界，最大合法进度约 47.5%–48.1%。斯帕完赛率为 85%，低于课程设定的 95% 圈速计分门槛。首轮权重与逐回合报告归档在 `v1_archive/grip-baseline/`。旧物理版的权重、说明和对应学生包位于 `v1_archive/`。

## 延长训练后的参考模型

在上海比较三组 PPO，每组 409,600 次环境交互。两组直接训练分别用进度奖励与进度加风险奖励。示范组用平滑走线和前视速度规划生成 30,000 条上海驾驶样本，拟合两层各 96 单元的 PPO 策略，再以相同交互步数训练。部分训练回合从上海不同赛段起步，覆盖恢复走线的情况。评测策略输入统一的 20 维观测，输出转向和目标速度。

示范组的奖励保留首次合法进度、耗时和终局项。车体净空低于 0.24 m 时增加贴边代价，轮胎抓地利用率超过 85% 时增加风险代价，同时小幅惩罚转向反打。走线示范只用于预训练。PPO 优化阶段以环境奖励更新策略。

| 每组代表检查点 | 上海干地，40 次 | 上海低抓地，20 次 |
| --- | --- | --- |
| 直接 PPO，进度奖励，最快检查点 | 40/40，32.12 秒 | 16/20 |
| 直接 PPO，风险奖励，最快检查点 | 40/40，31.94 秒 | 19/20 |
| 示范预训练 + PPO，获选 | 40/40，31.50 秒 | 20/20 |

上海干地完赛率达到 95% 的候选优先比较低抓地完赛率，其后比较干地完赛率与中位圈速。获选检查点位于第 368,640 步。锁定权重和哈希后，才用种子 2500–2539 复测斯帕。新版为 40/40、26.73 秒，首轮物理规则版为 34/40、26.66 秒。圈速为成功回合中位数。

```bash
python ta_private/train_racing_line.py --envs 8 --steps 409600 --seed 37 --out ta_private/runs/new_racing_line
python scripts/demo_lap.py --live --camera sideline --layout spa --seed 2500
```

当前权重、逐回合对照、选模记录和单圈遥测都位于 `best/`。

[轨迹引导赛车研究](https://arxiv.org/abs/2306.07003) · [TUM 赛车走线优化项目](https://github.com/TUMFTM/global_racetrajectory_optimization)

## 整圈用时走线与连续控制算法对照

这轮把参考走线改为按预计整圈时间优化。每条赛道用周期样条表示车道内走线。求解器在车道余量内调整横向位置，按局部曲率、制动距离、加速能力和弯中可用抓地余量规划速度。整圈目标允许当前弯为下一弯或后续直道让位。环境继续使用统一的 MuJoCo 接触、驱动与转向限制。

训练了两组 SAC（各 327,680 步）和一组 TD3（196,608 步）。它们在上海学习给同类规划参考添加转向和速度修正。SAC 使用之前的贴边、抓地与动作平顺奖励，TD3 候选提高用时和失败代价。对照比较的是各个完整驾驶方案。上一版 PPO 使用局部观测，新方案另外用赛道地图生成参考线。

候选在上海分别做 40 次干地复测和 20 次低抓地复测。两个条件都至少 95% 完赛的方案，以干地中位圈速排序。获选的是 7.1 档整圈用时规划器，上海干地 40/40、中位 31.25 秒，低抓地 20/20。最激进的 8.1 档干地中位 30.96 秒，低抓地仅 15/20。锁定候选后，使用种子 3500–3539 在斯帕复跑。规划候选按照冻结参数从斯帕几何生成参考线。

| 方案 | 斯帕完赛 | 成功回合中位圈速 |
| --- | --- | --- |
| 整圈用时规划，获选 | 40/40 | 26.29 秒 |
| 整圈用时规划 + SAC | 40/40 | 26.52 秒 |
| 上一版 PPO | 40/40 | 26.73 秒 |
| 整圈用时规划 + TD3 | 40/40 | 27.03 秒 |

```bash
python ta_private/replay_time_sac.py ta_private/lap_time_best --replay ta_private/spa_fastest.gif
python ta_private/replay_time_sac.py ta_private/lap_time_best --candidate sac --replay ta_private/spa_sac.gif
```

`lap_time_best/` 中的上海和斯帕参考线文件与冻结参数一起组成最快驾驶版，完整逐回合数据见 `results.json`。最佳强化学习方案使用 `runs/lap_time_sac_51/checkpoint_196608.zip`。已有的 `best/reference_policy.zip` 是上一次发布的纯 PPO 策略。

[赛车轨迹优化参考](https://github.com/TUMFTM/global_racetrajectory_optimization) · [SAC 论文](https://arxiv.org/abs/1801.01290) · [TD3 论文](https://arxiv.org/abs/1802.09477)

## 实测加减速与剩余时间奖励

用原车在干地直线做全油门和全刹车测试，拟合速度相关的加减速曲线。规划器同时考虑弯道占用的抓地余量。SAC 在这条参考线上学习有限幅度的转向和目标速度修正。每步按真实耗时扣分，失败额外扣 55 秒，势函数取负的预计剩余时间，折扣系数为 0.9995，势函数差分权重为 0.45，最终奖励除以 5。车辆、赛道与判罚沿用发布包。

上海先用独立起点初筛，再以 40 次干地和 20 次低抓地（μ = 0.82）选模。两组完赛率均至少 95% 后，按干地成功回合中位圈速选定配置。锁定线路、参数和权重后，用斯帕 40 个新干地起点复测。斯帕规划线由赛道几何与冻结参数生成。

| 方案 | 上海干地 | 上海低抓地 | 斯帕干地 |
| --- | --- | --- | --- |
| 上一版规划器 | 40/40，31.238 秒 | 20/20 | 40/40，26.289 秒 |
| 实测曲线规划器 | 40/40，31.058 秒 | 19/20 | 40/40，26.243 秒 |
| 实测曲线规划器 + SAC | 40/40，30.972 秒 | 19/20 | 40/40，26.226 秒 |

TA 使用 CPU 训练了 196,608 步。获选配置为第 65,536 步检查点，推理使用原修正幅度的 45%。斯帕同起点配对，SAC 整圈中位节省 69 ms，Bruxelles 到下一段直道末端节省 30 ms，Eau Rouge 到 Kemmel 直道末端慢 6 ms。只换实测曲线、沿用旧规划参数的上海消融结果慢 25 ms，表中实测规划器同时重新筛选了横向加速度预算。逐回合时间、冻结记录和权重保存在 `remaining_time_comparison/`。

```bash
python ta_private/calibrate_pace.py ta_private/runs/pace_repro
python ta_private/train_remaining_time.py --line ta_private/runs/pace_repro/shanghai_line.npz --out ta_private/runs/sac_repro
python ta_private/compare_measured_pace.py --baseline ta_private/lap_time_best --calibrated ta_private/runs/pace_repro --sac ta_private/runs/sac_repro --out ta_private/pace_comparison_repro
python ta_private/replay_measured_pace.py ta_private/remaining_time_comparison --gif spa_replay.gif --plot spa_timing.png
```

## 多走线剩余时间奖励

上海训练使用同一实测曲线参考线和 SAC 参数。多走线估计比较原线、较窄、较宽、提前和延后的横向走线。用当前位置、朝向、速度和实际转角生成短接入轨迹，按转角、转向速率、整车净空和抓地预算筛选，并计算接入后直到终点的用时。训练器保留原先的耗时、失败与势函数差分权重。

上海干地与低抓地各用 40 个新起点。两种地面均至少 95% 完赛后，按上海干地中位圈速选检查点。两组都采用上一版获选的推理修正幅度。冻结权重后，在 40 个新斯帕起点上按固定计时线复测，圈速与分段时间都按相同起点配对比较。

```bash
python ta_private/train_remaining_time.py --multi-line --line ta_private/runs/measured_pace_planners/shanghai_line.npz --out ta_private/runs/multi_line_repro
python ta_private/compare_multi_line_sac.py --previous ta_private/remaining_time_comparison --sac ta_private/runs/multi_line_repro --out ta_private/multi_line_comparison_repro
python ta_private/replay_measured_pace.py ta_private/multi_line_comparison --gif spa_multi_line_comparison.gif --plot spa_multi_line_timing.png
```

多走线版获选于第 196,608 步。上海旧版干地 40/40、低抓地 36/40，多走线版两种地面均 40/40。上海干地中位圈速分别为 30.978 秒和 31.095 秒，按起点配对后多走线版慢 115 毫秒。低抓地共同完赛的 36 个起点上，多走线版快 118 毫秒。旧版低抓地未达到门槛。多走线版斯帕独立复测 40/40，中位 26.313 秒。结果、分段、冻结记录与权重位于 `multi_line_comparison/`。

## SAC 横向目标

在现有多走线 SAC 上加入一个连续横向目标。零值跟踪原参考线，左右极值分别指向对应的车道内边界。目标每秒最多改变 1.25 个归一化单位。转向与目标速度仍由原控制接口执行。两组从同一冻结权重开始，使用同一多走线奖励和混合抓地训练配方，各继续训练 131,072 步。约四分之一的训练回合使用低抓地（μ = 0.82）。

上海小样本先筛横向目标推理幅度 1.00、0.45、0.20。用新起点测试每种幅度的代表检查点。原冻结版、固定线继续训练版和横向目标版分别通过上海干地及低抓地 95% 门槛后，才将各自权重和幅度锁定并带到斯帕独立复测。

```bash
python ta_private/train_lateral_target.py --previous ta_private/multi_line_comparison --out ta_private/runs/fixed_line_repro
python ta_private/train_lateral_target.py --previous ta_private/multi_line_comparison --out ta_private/runs/lateral_target_repro --lateral-target
python ta_private/compare_lateral_target.py --previous ta_private/multi_line_comparison --fixed ta_private/runs/fixed_line_repro --lateral ta_private/runs/lateral_target_repro --out ta_private/lateral_target_comparison_repro
python ta_private/replay_measured_pace.py ta_private/lateral_target_comparison --gif spa_lateral_target.gif --plot spa_lateral_timing.png
```

正式复测中，三组上海干地、上海低抓地和斯帕均为 40/40。上海干地中位圈速为原冻结版 31.105 秒、固定线续训 31.129 秒、横向目标 31.457 秒。斯帕分别为 26.307 秒、26.258 秒和 26.579 秒。横向目标采用第 98,304 步及 0.45 幅度，相对固定线续训在上海和斯帕配对分别慢 331 和 326 毫秒。斯帕最快固定线续训采用第 65,536 步。冻结权重、逐圈数据与复放遥测见 `lateral_target_comparison/`。

## 赛车走线奖励

在 TA 训练脚本中传入 `--racecraft` 启用。每步扣实际耗时，失败额外扣 55 秒，再加入预计剩余用时的势函数差和实测风险代价。完赛与失败都令终止势函数归零。折扣系数为 0.9995，势函数系数为 0.45，最终奖励除以 5。

从当前位姿、速度和前轮转角出发，比较可接入的多条候选线。估计覆盖整圈余程，并将弯后长直道的预计用时最多加权 30%。较短路线、大半径走线与晚弯心由预计用时比较。低抓地按轮地摩擦比例收紧预测横向加速度预算。赛道内侧与外侧位置都可以用于合法走线。

稳定性扣分依据 MuJoCo 的负荷加权抓地风险，分别在整车净空小于 0.20 m 或实际转向变化率较高时触发。风险代价上限为该控制步耗时的 20%。训练目标使用实际耗时和失败规则，长直道权重只用于势函数。

```bash
python ta_private/train_remaining_time.py --racecraft --line ta_private/lateral_target_comparison/baseline/shanghai_line.npz --out ta_private/runs/racecraft
```

### 同预算训练对照

用种子 101、8 个上海环境和 196,608 步训练赛车走线奖励。原多走线奖励组使用同样预算的存档，推理时使用与新版相同的修正幅度。上轮固定线续训版同时参加复测，作为已有最快版的参考。两组同预算训练候选以同一批上海初筛起点各留三项，正式上海干地与低抓地各 40 圈，通过双门槛后才使用 40 个新的斯帕起点。

```bash
python ta_private/train_remaining_time.py --racecraft --line ta_private/runs/measured_pace_planners/shanghai_line.npz --out ta_private/runs/racecraft_repro --steps 196608 --envs 8 --seed 101
python ta_private/compare_racecraft.py --original ta_private/runs/multi_line_sac_101 --racecraft ta_private/runs/racecraft_repro --previous ta_private/lateral_target_comparison --out ta_private/racecraft_comparison_repro
```

正式复测中，三组上海干地和斯帕均为 40/40。上海低抓地原多走线为 39/40，新奖励和固定线续训均为 40/40。上海干地中位圈速分别为 31.093、31.111 和 31.116 秒。斯帕分别为 26.319、26.329 和 26.247 秒。新奖励相对原多走线在上海干地配对慢 21 毫秒，在斯帕配对慢 12 毫秒。逐回合结果和冻结权重保存在 `racecraft_comparison/`。
