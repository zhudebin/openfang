#!/usr/bin/env bash
# OpenFang Docker 启动脚本
# 用法: ./start.sh [选项] [-- 自定义命令]
#
# 选项:
#   --build   启动前重新构建镜像
#   --detach  后台运行容器（-d 模式）
#
# 示例:
#   ./start.sh                        # 前台启动 openfang (默认 CMD: start)
#   ./start.sh --detach               # 后台启动
#   ./start.sh --build --detach       # 重新构建并后台启动
#   ./start.sh -- bash                # 进入容器 bash 交互
#   ./start.sh -- openfang --version  # 执行自定义命令

set -euo pipefail

IMAGE_NAME="openfang:latest"
CONTAINER_NAME="openfang"

# 本地配置目录，映射到容器内 OPENFANG_HOME
OPENFANG_HOME="${OPENFANG_HOME:-$HOME/.openfang}"

# 从 config.toml 中提取 api_listen 端口，默认 4200
if [[ -f "$OPENFANG_HOME/config.toml" ]]; then
    PORT=$(grep -E '^\s*api_listen\s*=' "$OPENFANG_HOME/config.toml" \
        | head -1 \
        | sed -E 's/.*:([0-9]+).*/\1/')
fi
PORT="${PORT:-4200}"

BUILD=false
DETACH=false
CUSTOM_CMD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)  BUILD=true; shift ;;
        --detach) DETACH=true; shift ;;
        --)       shift; CUSTOM_CMD=("$@"); break ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $BUILD; then
    echo "Building image $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" "$(dirname "$0")"
fi

# 清理同名旧容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container '$CONTAINER_NAME' ..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

RUN_FLAGS=()
$DETACH && RUN_FLAGS+=("-d")

# 自定义命令时：覆盖 entrypoint，分配 tty 以支持交互式 shell
if [[ ${#CUSTOM_CMD[@]} -gt 0 ]]; then
    RUN_FLAGS+=("-it" "--entrypoint" "${CUSTOM_CMD[0]}")
    # 去掉第一个元素，剩余作为参数
    CMD_ARGS=("${CUSTOM_CMD[@]:1}")
else
    CMD_ARGS=()
fi

echo "Starting OpenFang container ..."
echo "  Config dir : $OPENFANG_HOME -> /data"
echo "  API port   : $PORT"
if [[ ${#CUSTOM_CMD[@]} -gt 0 ]]; then
    echo "  Command    : ${CUSTOM_CMD[*]}"
fi

docker run \
    "${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"}" \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:${PORT}" \
    -v "$OPENFANG_HOME:/data" \
    "$IMAGE_NAME" \
    "${CMD_ARGS[@]+"${CMD_ARGS[@]}"}"
