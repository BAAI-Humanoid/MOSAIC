#!/bin/bash

# 1. 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <输入文件夹> <输出文件夹>"
    echo "示例: $0 ./my_videos ./converted_videos"
    exit 1
fi

# 2. 获取输入和输出目录变量
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# 3. 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入文件夹 '$INPUT_DIR' 不存在。"
    exit 1
fi

# 4. 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 开启 nullglob，防止在没有匹配文件时循环运行
shopt -s nullglob

# 5. 开始循环处理
for f in "$INPUT_DIR"/*.[mM][oO][vV]; do
    
    # 获取文件名（带后缀，不带路径）
    base_name=$(basename "$f")
    # 获取文件名（不带后缀）
    filename="${base_name%.*}"

    echo "--------------------------------"
    echo "正在转换: $f"
    echo "保存至: $OUTPUT_DIR/${filename}.mp4"

    # 执行转换
    # -y 表示如果输出文件已存在则自动覆盖
    ffmpeg -i "$f" -vcodec h264 -acodec aac -pix_fmt yuv420p "$OUTPUT_DIR/${filename}.mp4" -y

done

echo "--------------------------------"
echo "所有转换任务已完成！文件保存在: $OUTPUT_DIR"