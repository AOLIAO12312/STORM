#!/bin/bash

# 配置变量
CURRENT_DIR="."
TARGET_DIR="./token_reduction"
LOG_FILE="duplicate_removal_log_$(date +%F_%H%M%S).txt"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}开始检查重复文件...${NC}"
echo "源目录: $(realpath $CURRENT_DIR)"
echo "目标目录: $(realpath $TARGET_DIR)"
echo "日志文件: $LOG_FILE"
echo "----------------------------------------"

# 检查 token_reduction 目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}错误: 目录 '$TARGET_DIR' 不存在。${NC}"
    exit 1
fi

# 临时文件用于存储哈希值
HASH_CURRENT=$(mktemp)
HASH_TARGET=$(mktemp)

# 清理临时文件函数
cleanup() {
    rm -f "$HASH_CURRENT" "$HASH_TARGET"
}
trap cleanup EXIT

# 1. 计算当前目录所有文件的 MD5 (排除脚本自身和日志文件)
echo "正在计算当前目录文件哈希..."
find "$CURRENT_DIR" -maxdepth 1 -type f ! -name "$(basename "$0")" ! -name "*.log" -print0 | \
    xargs -0 md5sum 2>/dev/null | sort > "$HASH_CURRENT"

# 2. 计算 token_reduction 目录所有文件的 MD5
echo "正在计算 token_reduction 目录文件哈希..."
find "$TARGET_DIR" -maxdepth 1 -type f -print0 | \
    xargs -0 md5sum 2>/dev/null | sort > "$HASH_TARGET"

# 3. 找出重复的哈希值
# 我们只关心哈希值相同的情况
DUPLICATES=$(comm -12 <(cut -d' ' -f1 "$HASH_CURRENT" | sort -u) <(cut -d' ' -f1 "$HASH_TARGET" | sort -u))

if [ -z "$DUPLICATES" ]; then
    echo -e "${GREEN}未发现重复文件。${NC}"
    exit 0
fi

echo -e "${YELLOW}发现以下重复文件（将在 token_reduction 中删除）:${NC}"
echo ""

DELETE_COUNT=0

# 遍历每一个重复的哈希值
while IFS= read -r hash; do
    if [ -z "$hash" ]; then continue; fi

    # 在目标目录中找到具有该哈希的文件
    # 注意：md5sum 输出格式可能是 "hash  filename" 或 "hash filename"
    while IFS= read -r line; do
        file_path=$(echo "$line" | sed 's/^[a-f0-9]*  *//')

        if [ -f "$file_path" ]; then
            echo "标记删除: $file_path (哈希: $hash)"
            echo "DELETE: $file_path (Hash: $hash)" >> "$LOG_FILE"
            ((DELETE_COUNT++))

            # 执行删除 (如果确认要直接删除，去掉下面的 if 判断注释即可)
            # rm "$file_path"
        fi
    done < <(grep "^$hash" "$HASH_TARGET")

done <<< "$DUPLICATES"

echo "----------------------------------------"
echo "共发现 $DELETE_COUNT 个重复文件。"

# 交互式确认
read -p "是否真的要从 '$TARGET_DIR' 中删除这些文件？(y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "正在执行删除..."

    while IFS= read -r hash; do
        if [ -z "$hash" ]; then continue; fi

        while IFS= read -r line; do
            file_path=$(echo "$line" | sed 's/^[a-f0-9]*  *//')
            if [ -f "$file_path" ]; then
                rm "$file_path"
                echo -e "${RED}已删除: $file_path${NC}"
            fi
        done < <(grep "^$hash" "$HASH_TARGET")
    done <<< "$DUPLICATES"

    echo -e "${GREEN}清理完成！详细信息已记录在 $LOG_FILE${NC}"
else
    echo "操作已取消。没有文件被删除。"
    echo "(提示: 如果想自动删除，可以编辑脚本注释掉交互确认部分)"
fi