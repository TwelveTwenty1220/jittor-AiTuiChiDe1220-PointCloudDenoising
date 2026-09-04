#!/usr/bin/env bash
# 公共路径变量。两种用法二选一:
#   (a) 直接改下面带 /path/to 的行;
#   (b) 不改文件, 运行前 export 同名环境变量(环境变量优先, 例如 DATASET_TEST_B=/data/b bash 11_xxx.sh)。
export DATASET_TRAIN=${DATASET_TRAIN:-/path/to/dataset_train}          # A 榜官方训练集(OBJ 网格)
export DATASET_TEST=${DATASET_TEST:-/path/to/dataset_test_noisy}       # A 榜官方测试集(noisy.npy)
export DATASET_TEST_B=${DATASET_TEST_B:-/path/to/b/dataset_test_noisy} # B 榜官方测试集(200 模型, noisy.npy)

export WORK=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # 仓库根目录
export OUTPUTS=${OUTPUTS:-$WORK/outputs}      # 所有产物根(数据采样/权重/推理结果/日志), 默认不入库
export DATA_ROOT=$OUTPUTS/data                # A 榜训练采样 npy (01_prepare_data.sh 生成)
export DATA_B=$OUTPUTS/data_b                 # B 榜训练采样 npy (10_bboard_train_arms.sh 生成)
export RUNS=$OUTPUTS/runs                     # 训练 checkpoint / 日志
export OUT_A=$OUTPUTS/out_a                   # A 榜推理输出
export OUT_B=$OUTPUTS/out_b                   # B 榜推理输出
export SEED=${SEED:-2023}                     # 训练随机种子(线上成绩所用值)
export JITTOR_HOME=${JITTOR_HOME:-$OUTPUTS/jt_home}   # Jittor JIT 编译缓存
mkdir -p "$OUTPUTS"
