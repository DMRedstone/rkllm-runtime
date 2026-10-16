#!/bin/bash
# mock-rsync — rsync-backup.sh 测试用 rsync 模拟器
# 覆盖脚本实际使用的参数：-aHAX --numeric-ids --delete --partial-dir= --stats
#   --human-readable --one-file-system --bwlimit= --timeout= --exclude= --link-dest=
# 语义对齐真实 rsync：
#   - --link-dest：参考快照硬链接入场（未改动文件不占双倍空间）；
#   - 与参考副本一致的文件保留硬链接；不一致的先删后拷（换 inode，不污染旧快照）；
#   - --delete：清掉 dest 中源已不存在的条目；
#   - exclude：无斜杠模式匹配任意层级 basename（rsync 语义）。
# 环境变量：MOCK_RSYNC_RC（强制退出码，默认 0）、MOCK_RSYNC_SLEEP（睡眠秒数）、
#           MOCK_RSYNC_ARGV_LOG（记录命令行）、MOCK_RSYNC_DEBUG（记录动作明细）
unset -f rm 2>/dev/null || true
set -u
src=""; dest=""; delete=0; linkdest=""
excludes=()
for a in "$@"; do
  case "$a" in
    --exclude=*)    excludes+=("${a#--exclude=}") ;;
    --link-dest=*)  linkdest="${a#--link-dest=}" ;;
    --delete)       delete=1 ;;
    --partial-dir=*|--stats|--human-readable|--one-file-system|-aHAX|--numeric-ids|--bwlimit=*|--timeout=*) ;;
    -*) ;;
    *) if [ -z "$src" ]; then src="$a"; else dest="$a"; fi ;;
  esac
done
DBG="${MOCK_RSYNC_DEBUG:-}"
dbg() { [ -n "$DBG" ] && printf '%s dest=%s: %s\n' "$(date +%H:%M:%S)" "$dest" "$*" >> "$DBG"; return 0; }
[ -n "${MOCK_RSYNC_ARGV_LOG:-}" ] && printf '%s\n' "$*" >> "$MOCK_RSYNC_ARGV_LOG"
[ -n "${MOCK_RSYNC_SLEEP:-}" ] && sleep "$MOCK_RSYNC_SLEEP"
if [ -z "$src" ] || [ -z "$dest" ]; then echo "mock-rsync: 参数解析失败: $*" >&2; exit 2; fi
dbg "START src=$src linkdest=${linkdest:-无} delete=$delete excludes=${excludes[*]:-无}"

is_excluded() {
  local p="$1" b pat
  b="${p##*/}"
  for pat in "${excludes[@]:-}"; do
    [ -n "$pat" ] || continue
    pat="${pat%/}"
    case "$pat" in
      */*) case "$p" in $pat|$pat/*|*/$pat|*/$pat/*) return 0 ;; esac ;;
      *)   case "$b" in $pat) return 0 ;; esac
           case "$p" in $pat|$pat/*|*/$pat|*/$pat/*) return 0 ;; esac ;;
    esac
  done
  return 1
}

# 1) --link-dest：参考快照硬链接入场
if [ -n "$linkdest" ] && [ -d "$linkdest" ]; then
  mkdir -p "$dest"
  dbg "link-dest 硬链接入场: $linkdest -> $dest"
  ( cd "$linkdest" && find . -mindepth 1 -maxdepth 1 ! -name '.rsync-partial' -exec cp -al -- {} "$dest/" \; )
fi
# 2) 拷源（带 exclude）；一致的保留硬链接，不一致先删后拷
mkdir -p "$dest"
( cd "$src" && find . -mindepth 1 -print0 | while IFS= read -r -d '' p; do
    is_excluded "$p" && { dbg "exclude 跳过: $p"; continue; }
    if [ -d "$src/$p" ]; then
      mkdir -p "$dest/$p"
    else
      d="${p%/*}"; [ "$d" = "$p" ] && d="."
      mkdir -p "$dest/$d"
      if [ -e "$dest/$p" ] && cmp -s "$src/$p" "$dest/$p"; then
        dbg "一致，保留硬链接: $p (nlink=$(stat -c %h "$dest/$p" 2>/dev/null))"
      else
        dbg "不一致，先删后拷换 inode: $p (旧 nlink=$(stat -c %h "$dest/$p" 2>/dev/null || echo -))"
        command rm -f -- "$dest/$p"; cp -p "$src/$p" "$dest/$p"
      fi
    fi
  done )
# 3) --delete：清掉 dest 中源已不存在的条目
if (( delete )); then
  ( cd "$dest" && find . -mindepth 1 -print0 | while IFS= read -r -d '' p; do
      case "$p" in .rsync-partial*) continue ;; esac
      if [ ! -e "$src/$p" ]; then dbg "delete 移除: $p"; command rm -rf -- "$dest/$p"; fi
    done )
fi
# 4) --stats 摘要（脚本只记录、不解析）
printf 'Number of files: %s\n' "$(find "$dest" -type f | wc -l)"
printf 'total size is 0 bytes\n'
dbg "DONE"
exit "${MOCK_RSYNC_RC:-0}"
