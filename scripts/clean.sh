#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录（假设是MTID文件夹）
PROJECT_ROOT="$SCRIPT_DIR/.."

# 结束相关进程
ps aux | grep main_distributed | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

# 打印并删除文件
for dir in checkpoint/whl checkpoint_mlp/whl save_max save_max_mlp; do
    echo "Removing files in $PROJECT_ROOT/$dir:"
    ls -l "$PROJECT_ROOT/$dir"/epoch* 2>/dev/null || echo "No matching files"
    rm -f "$PROJECT_ROOT/$dir"/epoch*
done

# 单独处理 out 文件夹
echo "Removing files in $PROJECT_ROOT/out:"
ls -l "$PROJECT_ROOT"/out/* 2>/dev/null || echo "No files"
rm -rf "$PROJECT_ROOT"/out/*

echo "Cleanup completed."
