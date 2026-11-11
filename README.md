# AIR5058 RL1 · MuJoCo 赛车实验

本工作区包含课程起步代码与 TA 实验。学生从两条赛道、统一物理环境和 PPO/SAC 训练入口开始，在相同交互预算下比较自己设计的奖励或算法。TA 实验用额外的赛道参考线、TA 奖励和独立起点分析稳定性与圈速。

| 入口 | 内容 |
| --- | --- |
| [早期作业设计](MuJoCo赛车作业设计.md) | 课程任务和交付方案的原始草稿 |
| [学生作业说明](student/README.md) | 安装、首跑、训练、评测、接口与评分 |
| [TA 实验说明](ta_private/README.md) | 代码职责、主要训练命令与复现入口 |
| [TA 设计说明](ta_private/DESIGN.md) | 车辆规则、参考线、各版奖励与策略控制 |
| [TA 版本对照](ta_private/RESULTS.md) | 上海与斯帕成绩、固定分段、完赛门槛和历史数据 |
| [学生发布脚本](scripts/build_student_release.py) | 根据学生文件清单生成 ZIP、核对文件或导出 GitHub 仓库内容 |

GitHub 提供源码、实验记录和历史对照，策略权重在本地训练生成。学生从 `student/` 开始，TA 从仓库根目录运行实验。原有的根目录训练、评测和 Conda 命令也可继续使用。

## 目录

```text
student/
  racing/                 MuJoCo 车辆、赛道、环境和奖励接口
  train.py                公开训练入口
  evaluate.py             公开评测与回放
ta_private/
  experiments/            共用评测、分段计时和冻结资产校验
  train_*.py              TA 训练入口
  compare_*.py            同批次对照
  *_comparison/           逐圈数据、冻结权重和赛道参考线
  runs/                   训练日志与检查点
  v1_archive/             旧版环境与历史资产
scripts/                  学生版构建与导出
```

## 发布学生版

从项目根目录运行。发布脚本只读取学生文件清单，ZIP 与 GitHub 源码可由同一份学生文件重建。

```bash
python scripts/build_student_release.py
python scripts/build_student_release.py --check
python scripts/build_student_release.py --export PATH_TO_PUBLIC_GIT_CHECKOUT
```

可直接解压 [学生起步 ZIP](mujoco-racing-student.zip) 取得独立的课程框架。
