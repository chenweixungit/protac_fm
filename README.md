# PROTAC Joint Pose + Linker Flow-Matching

基于Flow Matching的PROTAC linker生成模型，支持片段位姿联合生成。兼容LinkerNet数据格式。

## 项目结构

```
protac_fm/
├── configs/
│   └── protac.yml          # 训练配置
├── data/
│   └── protac/             # PROTAC数据集 (从LinkerNet拷贝)
│       ├── 3d_index.pkl
│       ├── index.pkl
│       └── *.csv
├── scripts/
│   ├── train_protac.py     # 训练脚本
│   ├── sample_protac.py    # 采样脚本
│   ├── eval_protac.py      # 评估脚本
│   ├── run_train.bat       # Windows训练启动脚本
│   ├── run_sample.bat      # Windows采样启动脚本
│   └── run_eval.bat        # Windows评估启动脚本
├── protac_fm/
│   ├── utils/
│   │   ├── misc.py         # 工具函数
│   │   ├── reconstruct.py  # 分子重建
│   │   ├── so3.py          # SO(3)操作
│   │   └── geometry.py     # 几何变换
│   ├── data/
│   │   ├── linker_dataset.py  # LinkerNet兼容数据集
│   │   ├── transforms.py      # 数据变换
│   │   ├── featurize.py       # 特征化
│   │   └── collate.py         # 批处理
│   ├── model/
│   │   ├── egnn.py            # SE(3)-等变GNN
│   │   └── protac_flow.py     # Flow Matching模型
│   ├── diffusion/
│   │   └── mask.py            # 离散mask调度
│   └── chem/
│       ├── constraints.py     # 化学约束
│       ├── properties.py      # 属性计算
│       └── filter.py          # 后过滤
└── requirements.txt
```

## 安装

```bash
mamba activate linker
pip install -r requirements.txt
```

## 数据准备

将LinkerNet的PROTAC数据拷贝到 `data/protac/` 目录：
```
data/protac/
├── 3d_index.pkl
├── index.pkl
├── protac.csv
├── warhead.csv
├── e3_ligand.csv
└── linker.csv
```

## 使用

### 训练
```bash
# Windows
scripts\run_train.bat

# 或直接运行
python scripts/train_protac.py configs/protac.yml --device cuda --logdir ./logs
```

### 采样
```bash
# Windows
scripts\run_sample.bat

# 或直接运行
python scripts/sample_protac.py configs/protac.yml --ckpt ./logs/xxx/checkpoints/best.pt --device cuda --output_dir ./outputs
```

### 评估
```bash
# Windows
scripts\run_eval.bat

# 或直接运行
python scripts/eval_protac.py --results ./outputs/sampling_results.pt --config configs/protac.yml
```

## 特性

- **Flow Matching**: 使用线性插值路径进行连续坐标生成
- **Mask Denoising**: 离散原子类型和键类型的mask去噪
- **位姿生成**: 联合生成两个片段的刚体位姿
- **LinkerNet兼容**: 直接使用LinkerNet的数据格式和预处理
- **硬约束**: anchor连接度、价态、连通性检查
- **后过滤**: RDKit属性过滤（SA/QED/LogP/TPSA/MW/RotB等）

## 配置说明

主要配置项在 `configs/protac.yml`:

```yaml
dataset:
  path: ./data/protac       # 数据路径
  split_mode: warhead       # 划分模式: warhead/ligase/random

model:
  h_dim: 256               # 隐藏层维度
  n_layers: 6              # EGNN层数
  num_steps: 500           # 扩散步数

train:
  batch_size: 4            # 批大小
  max_iters: 200000        # 最大迭代次数
  lr: 1e-4                 # 学习率
```
