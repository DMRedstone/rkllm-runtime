#!/bin/bash
# mock-flock — 测试用 flock 模拟（本机无真实 flock）
# flock -n FD：默认获取成功；MOCK_FLOCK_BUSY=1 时模拟"已有实例在运行"
for a in "$@"; do
  case "$a" in
    -n) if [ -n "${MOCK_FLOCK_BUSY:-}" ]; then echo "mock-flock: 锁被占用" >&2; exit 1; fi; exit 0 ;;
  esac
done
exit 0
