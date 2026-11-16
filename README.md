# AIR5058 · Q-learning 赛车作业

[学生作业说明与运行命令](student/README.md)

学生在 MuJoCo 上海赛道设计自己的表格 Q-learning。框架提供中心线几何、车辆观测、物理和固定中线基线。学生负责状态分箱、动作和奖励设计。提交同一份冻结 Q 表，在干地和低抓地分别评测。

两种路面分别按圈速 70 分、弯中有效抓地 30 分计分，通过各自的完赛门槛后等权平均。评测器能生成同起点的车速与抓地图。

```bash
cd student
conda env create -f environment.yml
conda activate air5058-racing-student
python evaluate.py --count 2
```

| 文件 | 用途 |
| --- | --- |
| [student_policy.py](student/student_policy.py) | 学生的 Q 表训练和推理模板 |
| [lab.py](student/lab.py) | 上海环境接口和固定速度中线基线 |
| [evaluate.py](student/evaluate.py) | 双路面评测和结果导出 |
| [scoring.py](student/scoring.py) | 圈速计分和有效抓地测量 |
| [performance_plot.py](student/performance_plot.py) | 实测车速与抓地对照图 |
