#!/bin/bash
# harness.sh — rsync-backup.sh 函数级隔离测试（T1–T11）
# 方法：从被测脚本动态定位并抽取全部函数（起点/终点断言自证），
#       在桩全局配置下逐场景运行，只验证函数行为，不跑主流程。
set -uo pipefail
SB="$(cd "$(dirname "$0")" && pwd)"
SRC="$SB/rsync-backup.sh"
LOG="$SB/harness.log"
: > "$LOG"
# 幂等：清空上一轮场景目录（否则复跑会读到上轮产物，如 T4c 的已提升快照）
command rm -rf "$SB/t2" "$SB/t3" "$SB/t4a" "$SB/t4b" "$SB/t4c" "$SB/t4d" "$SB/t5" "$SB/t6a" "$SB/t6b" "$SB/t8" "$SB/t9" "$SB/t10" "$SB/t11" "$SB/h-back"

# ---------- 抽取自证 ----------
# 动态定位函数区：起点为 log() 定义，终点为 backup_one 收尾的 }。
# 不硬编码行号——被测脚本任何编辑都会使行号失效（2026-10-22 即因此 FATAL）。
FN_START="$(grep -n '^log() {$' "$SRC" | head -n1 | cut -d: -f1)"
if [ -z "$FN_START" ]; then echo "FATAL: 未找到 log() { 定义"; exit 1; fi
FN_END="$(awk -v s="$FN_START" 'NR>=s && /^backup_one\(\) \{/{b=NR} b && NR>b && /^}$/{print NR; exit}' "$SRC")"
if [ -z "$FN_END" ]; then echo "FATAL: 未找到 backup_one 收尾"; exit 1; fi
sed -n "${FN_START},${FN_END}p" "$SRC" > "$SB/funcs.sh"
if [ "$(head -n 1 "$SB/funcs.sh")" != "log() {" ]; then echo "FATAL: 抽取起点不是 log() {"; exit 1; fi
if [ "$(tail -n 1 "$SB/funcs.sh")" != "}" ]; then echo "FATAL: 抽取终点不是 backup_one 收尾"; exit 1; fi
printf '函数抽取自证通过：%s 行（源 %s-%s）\n' "$(wc -l < "$SB/funcs.sh")" "$FN_START" "$FN_END"

# ---------- 桩全局配置（与脚本可配置段同构） ----------
BACKUP_BASE="$SB/h-back"
KEEP=3
FAILED_KEEP_DAYS=14
FAILED_KEEP_COUNT=2
LOG_MAX_SIZE=500
LOG_KEEP_DAYS=30
ONE_FILE_SYSTEM=0
SHRINK_RATIO=0
CHMOD_SNAPSHOT_RO=1
MIN_FREE_PCT=1
MIN_FREE_INODE_PCT=1
REQUIRE_ROOT=0
ALLOW_ROOT_SRC=0
REJECT_SRC_ON_ROOTFS=0
CHECK_SRC_EMPTY=1
BW_LIMIT=""
RSYNC_TIMEOUT=0
EXCLUDES=()
OK_RSYNC_CODES=(0 24)
WARN_RSYNC_CODES=(23)
RUN=(nice -n 19)
overall_rc=0
CURRENT_BASE=""; CURRENT_DATE=""; CURRENT_WORK=""

source "$SB/funcs.sh"

# ---------- 断言工具 ----------
PASS=0; FAIL=0
chk() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$d"; else FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$d"; fi; }
eq()  { if [ "$2" = "$3" ]; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; else FAIL=$((FAIL+1)); printf '  FAIL  %s (期望 [%s] 实得 [%s])\n' "$1" "$2" "$3"; fi; }
rc_is(){ if [ "$2" = "$3" ]; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; else FAIL=$((FAIL+1)); printf '  FAIL  %s (期望 rc=%s 实得 rc=%s)\n' "$1" "$2" "$3"; fi; }
chkg(){ if grep -q "$2" "$LOG"; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; else FAIL=$((FAIL+1)); printf '  FAIL  %s (日志缺: %s)\n' "$1" "$2"; fi; }

printf '\n=== T1 退出码分类 ===\n'
eq "T1.1 in_array 命中返回 0"  "0" "$(in_array 24 0 24; echo $?)"
eq "T1.2 in_array 未命中返回 1" "1" "$(in_array 11 0 24; echo $?)"
eq "T1.3 rsync 0 → ok"   "ok"   "$(rsync_class 0)"
eq "T1.4 rsync 24 → ok"  "ok"   "$(rsync_class 24)"
eq "T1.5 rsync 23 → warn" "warn" "$(rsync_class 23)"
eq "T1.6 rsync 11 → fail" "fail" "$(rsync_class 11)"
eq "T1.7 rsync 1 → fail"  "fail" "$(rsync_class 1)"

printf '\n=== T2 文件计数 ===\n'
cf="$SB/t2"; mkdir -p "$cf/sub"; : > "$cf/a"; : > "$cf/b"; : > "$cf/sub/c"
eq "T2.1 ONE_FILE_SYSTEM=0 计数=3" "3" "$(count_files "$cf")"
ONE_FILE_SYSTEM=1
eq "T2.2 ONE_FILE_SYSTEM=1 计数=3（同文件系统）" "3" "$(count_files "$cf")"
ONE_FILE_SYSTEM=0

printf '\n=== T3 完整快照列举 ===\n'
lc="$SB/t3"
mkdir -p "$lc/2026-10-01_000000" "$lc/2026-10-02_000000" "$lc/2026-10-03_000000.failed.9" "$lc/.2026-10-04_000000.inprogress.9" "$lc/2026-10-05_000000"
: > "$lc/2026-10-01_000000/.complete"; : > "$lc/2026-10-05_000000/.complete"
: > "$lc/2026-10-03_000000.failed.9/.complete"; : > "$lc/.2026-10-04_000000.inprogress.9/.complete"
mapfile -d '' -t _t3 < <(list_complete_snapshots "$lc")
eq "T3.1 只列带 .complete 的正式快照、新的在前" \
  "2026-10-05_000000 2026-10-01_000000" "${_t3[*]}"
eq "T3.1b 输出为 NUL 分隔（文件名含换行也不会串）" "2" "${#_t3[@]}"

printf '\n=== T4 残留目录处理（handle_leftovers） ===\n'
hl="$SB/t4a"
mkdir -p "$hl/.2026-10-10_000000.inprogress.111" "$hl/.2026-10-11_000000.inprogress.222"
for i in $(seq 1 50); do : > "$hl/.2026-10-10_000000.inprogress.111/f$i"; done
: > "$hl/.2026-10-11_000000.inprogress.222/a"; : > "$hl/.2026-10-11_000000.inprogress.222/b"; : > "$hl/.2026-10-11_000000.inprogress.222/c"
RESUME=""
handle_leftovers "$hl" RESUME
eq "T4.1 续传选中文件数最多者（非最新时间戳）" "$hl/.2026-10-10_000000.inprogress.111" "$RESUME"
chk "T4.2 薄残留被标记 failed（后缀为当前 PID）" [ -d "$hl/2026-10-11_000000.failed.$$" ]
chk "T4.3 薄残留原目录已改名" [ ! -e "$hl/.2026-10-11_000000.inprogress.222" ]
chkg "T4.4 日志记录完成度统计" "已含 50 个文件" "$LOG"
chkg "T4.5 日志记录续传选择" "本次续传" "$LOG"

hl2="$SB/t4b"; mkdir -p "$hl2/.2026-10-12_000000.inprogress.333"; : > "$hl2/.2026-10-12_000000.inprogress.333/x"
RESUME=""
handle_leftovers "$hl2" RESUME
eq "T4b.1 仅一份时直接续传" "$hl2/.2026-10-12_000000.inprogress.333" "$RESUME"
chk "T4b.2 未做全树统计（省遍历）" bash -c '! grep -q "已含 1 个文件" "$1"' _ "$LOG"

hl3="$SB/t4c"; mkdir -p "$hl3/.2026-10-13_000000.inprogress.444"
: > "$hl3/.2026-10-13_000000.inprogress.444/x"; : > "$hl3/.2026-10-13_000000.inprogress.444/.complete"
RESUME=""
handle_leftovers "$hl3" RESUME
chk "T4c.1 已 complete 的残留被提升为正式快照" [ -f "$hl3/2026-10-13_000000/.complete" ]
eq "T4c.2 无续传目标" "" "$RESUME"
chkg "T4c.3 日志记录提升" "提升上次已完成但未改名的快照" "$LOG"

hl4="$SB/t4d"; mkdir -p "$hl4/2026-10-14_000000" "$hl4/.2026-10-14_000000.inprogress.555"
: > "$hl4/2026-10-14_000000/.complete"; : > "$hl4/.2026-10-14_000000.inprogress.555/.complete"
RESUME=""
handle_leftovers "$hl4" RESUME
chk "T4d.1 目标已存在时改判 failed（不覆盖）" [ -d "$hl4/2026-10-14_000000.failed.$$" ]
chk "T4d.2 已存在的正式快照不受影响" [ -f "$hl4/2026-10-14_000000/.complete" ]

printf '\n=== T5 快照淘汰（prune_snapshots） ===\n'
ps="$SB/t5"; mkdir -p "$ps"
for i in 1 2 3 4 5 6; do d="$ps/2026-10-0${i}_000000"; mkdir -p "$d"; : > "$d/.complete"; : > "$d/f"; done
chmod a-w "$ps/2026-10-01_000000" "$ps/2026-10-02_000000"
prune_snapshots "$ps"
mapfile -d '' -t _t5 < <(list_complete_snapshots "$ps")
eq "T5.1 保留 3 份最新" "3" "${#_t5[@]}"
eq "T5.2 剩余确为最新三份" "2026-10-06_000000 2026-10-05_000000 2026-10-04_000000" "${_t5[*]}"
chk "T5.3 删除路径对只读顶层也生效（chmod u+w 已调用）" [ ! -d "$ps/2026-10-01_000000" ]

printf '\n=== T6 失败目录清理（prune_failed） ===\n'
pf="$SB/t6a"; mkdir -p "$pf"
for i in 1 2 3 4 5; do mkdir -p "$pf/2026-10-0${i}_000000.failed.9$i"; done
prune_failed "$pf"
eq "T6.1 超数量时保留最新 2 份" "2" "$(find "$pf" -maxdepth 1 -mindepth 1 -type d -name '*.failed.*' | wc -l)"
pf2="$SB/t6b"
mkdir -p "$pf2/2026-10-01_000000.failed.1" "$pf2/2026-10-05_000000.failed.5" "$pf2/2026-10-06_000000.failed.6"
touch -d "40 days ago" "$pf2/2026-10-01_000000.failed.1"
prune_failed "$pf2"
chk "T6.2 超龄（>14 天）失败目录被删" [ ! -d "$pf2/2026-10-01_000000.failed.1" ]
chk "T6.3 未超龄的保留" [ -d "$pf2/2026-10-06_000000.failed.6" ]

printf '\n=== T7 空间检查（check_free_space） ===\n'
MIN_FREE_PCT=101; MIN_FREE_INODE_PCT=0
check_free_space "$SB" 1; rc=$?
rc_is "T7.1 阈值 101% 必然触发软失败" "1" "$rc"
MIN_FREE_PCT=1
check_free_space "$SB" 1; rc=$?
rc_is "T7.2 阈值 1% 正常通过" "0" "$rc"
MIN_FREE_PCT=5
check_free_space "/nonexistent-xyz-123" 1; rc=$?
rc_is "T7.3 df 解析失败返回 0（跳过而非中止）" "0" "$rc"
chkg "T7.4 解析失败有日志留痕（C2 行为）" "无法解析 df 空间输出" "$LOG"
MIN_FREE_INODE_PCT=1
check_free_space "$SB" 1; rc=$?
rc_is "T7.5 inode 检查不误报" "0" "$rc"

printf '\n=== T8 日志轮转（rotate_logs） ===\n'
rl="$SB/t8/log"; command rm -rf "$rl"; mkdir -p "$rl"; LOG="$rl/rsync-backup.log"
printf 'x%.0s' $(seq 1 600) > "$LOG"
rotate_logs
eq "T8.1 超限触发轮转" "1" "$(find "$rl" -name 'rsync-backup.log.*' -type f | wc -l)"
chk "T8.2 主日志已被移走" [ ! -f "$LOG" ]
old="$(find "$rl" -name 'rsync-backup.log.*' -type f)"
touch -d "40 days ago" "$old"
printf 'y%.0s' $(seq 1 600) > "$LOG"; rotate_logs
chk "T8.3 超龄（>30 天）轮转文件被清" [ ! -f "$old" ]
eq "T8.4 新轮转文件保留" "1" "$(find "$rl" -name 'rsync-backup.log.*' -type f | wc -l)"

printf '\n=== T9 失败标记（mark_failed） ===\n'
mf="$SB/t9"; mkdir -p "$mf/work"; : > "$mf/work/f1"; : > "$mf/work/f2"
mark_failed "$mf/work" "$mf" "2026-10-14_120000"
chk "T9.1 生成 .failed.\$\$ 目录" [ -d "$mf/2026-10-14_120000.failed.$$" ]
chk "T9.2 work 目录已改名" [ ! -d "$mf/work" ]
chk "T9.3 内容完整保留" [ -f "$mf/2026-10-14_120000.failed.$$/f1" ]

printf '\n=== T10 EXIT trap 清理（cleanup） ===\n'
cl="$SB/t10"; mkdir -p "$cl/work"; : > "$cl/work/f"
( trap cleanup EXIT; CURRENT_BASE="$cl"; CURRENT_DATE="2026-10-14_130000"; CURRENT_WORK="$cl/work"; false )
rc=$?
rc_is "T10.1 trap 保留原退出码（1）" "1" "$rc"
chk "T10.2 trap 把残留目录改名 failed" [ -d "$cl/2026-10-14_130000.failed.$$" ]

printf '\n=== T11 backup_one 输入校验（不触达 rsync） ===\n'
bv="$SB/t11"; mkdir -p "$bv/empty-src"; BACKUP_BASE="$bv/base"
overall_rc=0; backup_one "a|b|c";                 eq "T11.1 三段配置置 overall_rc=1" "1" "$overall_rc"; chkg "T11.1b" "无效源配置" "$LOG"
overall_rc=0; backup_one "/x|../evil";            eq "T11.2 路径穿越 NAME 被拒" "1" "$overall_rc"; chkg "T11.2b" "无效备份名" "$LOG"
overall_rc=0; backup_one "/|root";                eq "T11.3 SRC=/ 被拒" "1" "$overall_rc"; chkg "T11.3b" "拒绝备份根目录" "$LOG"
overall_rc=0; backup_one "$bv/nope|n";            eq "T11.4 源不存在被拒" "1" "$overall_rc"; chkg "T11.4b" "源不存在或不是目录" "$LOG"
overall_rc=0; backup_one "$bv/empty-src|e";       eq "T11.5 空源被拒" "1" "$overall_rc"; chkg "T11.5b" "源为空" "$LOG"
chk "T11.6 校验失败均未创建快照目录" [ -z "$(find "$bv" -mindepth 1 -maxdepth 1 -type d -name base -exec ls {} \; 2>/dev/null)" ]

printf '\n===== 函数级测试合计: PASS=%s FAIL=%s =====\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
