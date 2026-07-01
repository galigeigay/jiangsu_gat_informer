#!/bin/bash

# 设置项目根目录
PROJECT_DIR="/data"

# 添加项目目录到PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 切换到项目目录
cd "${PROJECT_DIR}" || exit 1
echo "Current working directory: $(pwd)"

# 执行Python脚本
python jiangsu_power.py train

python jiangsu_power.py predict
