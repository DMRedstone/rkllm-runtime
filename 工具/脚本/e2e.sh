#!/bin/bash
# e2e.sh — rsync-backup.sh 端到端场景测试
# 用法：bash e2e.sh [FROM] [TO]   （场景编号 1..13，默认 1..99 全量）
# 环境：本机 MSYS2 bash 5.3.15；rsync/flock/mountpoint 用 mock（语义见 bin/rsync 注释）；
#       ionice 缺失，正好覆盖脚本自身的回退分支。
# 说明：分段运行（TO<13）仅用于排查；全量结果以 FROM=1 的完整运行为准。
set -uo pipefail
# SB 用 MSYS 风格路径：本机 MSYS bash 的 command -v 不解析 Windows 风格（C:/...）的
# PATH 项，脚本启动自检会误报"缺少命令"；/tmp 与 C:/Users/user1/AppData/Local/Temp
# 是同一目录，文件位置不变。
SB="/tmp/rsync-test-sb"
SRC="$SB/rsync-backup.sh"
export PATH="$SB/bin:$PATH"
LOG="$SB/log/rsync-backup.log"
OUT="$SB/e2e.out"
ARGV="$SB/argv.log"
FROM="${1:-1}"
TO="${2:-99}"
PASS=0; FAIL=0; off=0; log_ino=0

chk() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$d"; else FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$d"; fi; }
eq()  { if [ "$2" = "$3" ]; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; else FAIL=$((FAIL+1)); printf '  FAIL  %s (期望 [%s] 实得 [%s])\n' "$1" "$2" "$3"; fi; }
# 轮转会把 $LOG 整体 mv 走（inode 变化），轮转后新文件从 0 开始记账，
# 轮转前的字节偏移会大于新文件长度导致 tail 为空；inode 变化时整读新文件。
newlog() {
  local cur_ino
  cur_ino="$(stat -c %i "$LOG" 2>/dev/null || echo 0)"
  if [ "$cur_ino" != "$log_ino" ]; then cat "$LOG" 2>/dev/null; else tail -c +"$((off + 1))" "$LOG" 2>/dev/null; fi
}
chkgn() { if newlog | grep -q "$2"; then PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; else FAIL=$((FAIL+1)); printf '  FAIL  %s (本轮日志缺: %s)\n' "$1" "$2"; fi; }

snaps() { find "$SB/back/$1" -mindepth 1 -maxdepth 1 -type d -name '20*' ! -name '*.failed.*' 2>/dev/null | wc -l; }
latest() { find "$SB/back/$1" -mindepth 1 -maxdepth 1 -type d -name '20*' 2>/dev/null | sort | tail -n 1; }

gen_copy() { # gen_copy <dst> [K=V ...] —— 仅替换配置区，1:1 行替换，行数不变
  local dst="$1"; shift
  awk -v sb="$SB" '
    /^SOURCES=\(/ { print "SOURCES=("; print "  \"" sb "/src/data1|data1\""; print "  \"" sb "/src/data2|data2\""; print ")"; blk=1; next }
    blk && /^\)$/ { blk=0; next }
    blk { next }
    /^BACKUP_BASE=/ { print "BACKUP_BASE=\"" sb "/back\""; next }
    /^LOG=/ { print "LOG=\"" sb "/log/rsync-backup.log\""; next }
    /^LOCK=/ { print "LOCK=\"" sb "/lock/rsync-backup.lock\""; next }
    /^REQUIRE_ROOT=/ { print "REQUIRE_ROOT=0"; next }
    /^REJECT_SRC_ON_ROOTFS=/ { print "REJECT_SRC_ON_ROOTFS=0"; next }
    /^export PATH=/ { print "export PATH=\"" sb "/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\""; next }
    { print }
  ' "$SRC" > "$dst"
  local kv
  for kv in "$@"; do sed -i "s|^${kv%%=*}=.*|$kv|" "$dst"; done
  bash -n "$dst" || { printf 'FATAL: 生成的副本 %s 语法错误\n' "$dst"; exit 1; }
}

run_script() { # run_script <copy> [VAR=VAL ...]
  local c="$1"; shift
  log_ino="$(stat -c %i "$LOG" 2>/dev/null || echo 0)"
  off=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  env -u 'BASH_FUNC_rm%%' -u 'BASH_FUNC_unlink%%' -u 'BASH_FUNC_rmdir%%' \
    MOCK_RSYNC_DEBUG="${MOCK_RSYNC_DEBUG:-/dev/null}" "$@" \
    bash -c 'unset -f rm unlink rmdir 2>/dev/null; exec bash "$1"' _ "$c" >>"$OUT" 2>&1
}

sec() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }

# ============ 1. 初始化 ============
if sec 1; then
printf '=== 1. 重置沙箱、生成测试副本并自证 ===\n'
command rm -rf "$SB/back" "$SB/log" "$SB/lock" "$SB/src" "$SB/e2e.out" "$SB/rsync-test.sh" "$SB/rt-"*.sh \
  "$SB/h-back" "$SB/t2" "$SB/t3" "$SB/t4a" "$SB/t4b" "$SB/t4c" "$SB/t4d" "$SB/t5" "$SB/t6a" "$SB/t6b" \
  "$SB/t8" "$SB/t9" "$SB/t10" "$SB/t11" "$SB/zone-orig.txt" "$SB/zone-copy.txt" "$SB/funcs.sh" "$SB/harness.log"
mkdir -p "$SB/src/data1/sub" "$SB/src/data2" "$SB/log" "$SB/lock"
: > "$OUT"; : > "$ARGV"; : > "$LOG"
gen_copy "$SB/rsync-test.sh" KEEP=3 LOG_MAX_SIZE=500
# 原文件末行无换行符，wc -l 会少数一行；awk END{print NR} 不受末行换行影响
eq "1.1 副本与原件行数一致" "$(awk 'END{print NR}' "$SRC")" "$(awk 'END{print NR}' "$SB/rsync-test.sh")"
sed -n '144,705p' "$SRC" > "$SB/zone-orig.txt"
sed -n '144,705p' "$SB/rsync-test.sh" > "$SB/zone-copy.txt"
if diff -q "$SB/zone-orig.txt" "$SB/zone-copy.txt" >/dev/null; then
  PASS=$((PASS+1)); printf '  PASS  1.2 函数区（144-705 行）逐字节一致，逻辑零改动\n'
else
  FAIL=$((FAIL+1)); printf '  FAIL  1.2 函数区被改动\n'
fi
printf -- '--- 测试副本与原件的配置差异（应仅这些行）---\n'
diff "$SRC" "$SB/rsync-test.sh" || true

# 构造测试源
i=1; while [ "$i" -le 20 ]; do printf 'data1 file %s\n' "$i" > "$SB/src/data1/f$i.txt"; i=$((i+1)); done
printf 'sub a\n' > "$SB/src/data1/sub/a.txt"
printf 'd2 one\n' > "$SB/src/data2/one.txt"; printf 'd2 two\n' > "$SB/src/data2/two.txt"
mkdir -p "$SB/src/data1/.cache" "$SB/src/data1/lost+found"
printf 'junk\n' > "$SB/src/data1/.cache/junk.bin"; printf 'junk\n' > "$SB/src/data1/lost+found/junk"; printf 'tmpjunk\n' > "$SB/src/data1/scratch.tmp"
fi

# ============ 2. E1 首次完整运行 ============
if sec 2; then
printf '\n=== E1 首次完整运行 ===\n'
: > "$ARGV"
run_script "$SB/rsync-test.sh" MOCK_RSYNC_ARGV_LOG="$ARGV"; rc=$?
eq "E1.1 退出码 0" "0" "$rc"
eq "E1.2 data1 生成 1 份快照" "1" "$(snaps data1)"
eq "E1.3 data2 生成 1 份快照" "1" "$(snaps data2)"
eq "E1.4 两个源均记录成功" "2" "$(newlog | grep -c '备份成功')"
eq "E1.5 结束日志退出码 0" "1" "$(newlog | grep -c '全部任务结束，退出码 0')"
eq "E1.6 rsync 调用次数=2（每源一次）" "2" "$(grep -c . "$ARGV")"
chk "E1.7 argv 含 -aHAX" grep -q -- '-aHAX' "$ARGV"
chk "E1.8 argv 含 --delete" grep -q -- '--delete' "$ARGV"
chk "E1.9 argv 含 --exclude=.cache/" grep -q -- '--exclude=.cache/' "$ARGV"
chk "E1.10 首跑无 --link-dest（无上代快照）" bash -c '! grep -q -- "--link-dest" "$1"' _ "$ARGV"
s1="$(latest data1)"
chk "E1.11 .complete 标记存在" [ -f "$s1/.complete" ]
chk "E1.12 普通文件入库" [ -f "$s1/f20.txt" ]
chk "E1.13 子目录完整" [ -f "$s1/sub/a.txt" ]
chk "E12.1 .cache/ 被排除" [ ! -e "$s1/.cache" ]
chk "E12.2 lost+found/ 被排除" [ ! -e "$s1/lost+found" ]
chk "E12.3 *.tmp 被排除" [ ! -e "$s1/scratch.tmp" ]
fi

# ============ 3. E2 二次运行（硬链接去重） ============
if sec 3; then
printf '\n=== E2 二次运行（无变化 → 硬链接去重）===\n'
: > "$ARGV"
run_script "$SB/rsync-test.sh" MOCK_RSYNC_ARGV_LOG="$ARGV"; rc=$?
eq "E2.1 退出码 0" "0" "$rc"
eq "E2.2 data1 现有 2 份快照" "2" "$(snaps data1)"
s2="$(latest data1)"
eq "E2.3 未改动文件 nlink=2（硬链接共享）" "2" "$(stat -c %h "$s1/f1.txt")"
eq "E2.4 两代快照同一 inode" "$(stat -c %i "$s1/f1.txt")" "$(stat -c %i "$s2/f1.txt")"
chk "E2.5 argv 出现 --link-dest" grep -q -- '--link-dest=' "$ARGV"
chk "E2.6 link-dest 指向上一代快照" grep -q -- "--link-dest=$s1" "$ARGV"
fi

# ============ 4. E3 源变更与快照不可变性 ============
if sec 4; then
printf '\n=== E3 源变更（增/删/改）与快照不可变性 ===\n'
command rm -f "$SB/src/data1/f19.txt" "$SB/src/data1/f20.txt"
printf 'MODIFIED\n' > "$SB/src/data1/f1.txt"
printf 'new file\n' > "$SB/src/data1/f21.txt"
run_script "$SB/rsync-test.sh"; rc=$?
eq "E3.1 退出码 0" "0" "$rc"
s3="$(latest data1)"
chk "E3.2 新快照中已删文件不存在（--delete 生效）" [ ! -e "$s3/f19.txt" ]
chk "E3.3 旧快照仍保留已删文件（不可变）" [ -f "$s1/f19.txt" ]
eq "E3.4 新快照反映修改内容" "MODIFIED" "$(cat "$s3/f1.txt")"
eq "E3.5 旧快照内容未被污染" "data1 file 1" "$(cat "$s1/f1.txt")"
chk "E3.6 新增文件入库" [ -f "$s3/f21.txt" ]
# 注意：本机 MSYS 的 stat -c %h 在 unlink 后会虚高（find -inum 实测才准），
# 因此用「inode 是否变化 + 实际链接数」来验证，而非 stat %h。
chk "E3.7 改动文件在新快照中换了 inode" bash -c '[ "$(stat -c %i "$1")" != "$(stat -c %i "$2")" ]' _ "$s1/f1.txt" "$s3/f1.txt"
eq "E3.7b 旧 inode 实际链接数=2（find -inum 实测）" "2" "$(find "$SB/back" -inum "$(stat -c %i "$s1/f1.txt")" | wc -l)"
fi

# ============ 5. E4 KEEP 淘汰 ============
if sec 5; then
printf '\n=== E4 KEEP 淘汰（KEEP=3）===\n'
run_script "$SB/rsync-test.sh"; rc=$?
run_script "$SB/rsync-test.sh"; rc2=$?
eq "E4.1 第 4 次退出码 0" "0" "$rc"
eq "E4.2 第 5 次退出码 0" "0" "$rc2"
eq "E4.3 五轮后仅保留 3 份" "3" "$(snaps data1)"
chk "E4.4 最旧快照已淘汰" [ ! -d "$s1" ]
chk "E4.5 第 2 份已淘汰" [ ! -d "$s2" ]
chk "E4.6 第 3 份仍在" [ -d "$s3" ]
chk "E4.7 日志有删除记录" grep -q "删除旧快照" "$LOG"
fi

# ============ 6. E5 SHRINK_RATIO 缩小保护 ============
if sec 6; then
printf '\n=== E5 SHRINK_RATIO=10 缩小保护 ===\n'
# 注意：count_files 统计源全量文件（含被 rsync 排除的 .cache/lost+found/*.tmp），
# 必须把这些一并删掉，让 src_count 真正降到 1，才能触发 1*10 < 21 的阈值。
command rm -f "$SB/src/data1"/f*.txt "$SB/src/data1/scratch.tmp"
command rm -rf "$SB/src/data1/.cache" "$SB/src/data1/lost+found"
before="$(snaps data1)"
newest_before="$(latest data1)"
count_before="$(find "$newest_before" -type f ! -name '.complete' | wc -l)"
run_script "$SB/rsync-test.sh"; rc=$?
eq "E5.1 退出码 1" "1" "$rc"
chkgn "E5.2 日志记录异常缩小" "源文件数异常缩小"
eq "E5.3 未生成新快照" "$before" "$(snaps data1)"
eq "E5.4 最新快照未被替换（无 --delete 误杀）" "$newest_before" "$(latest data1)"
eq "E5.4b 最新快照文件数不变" "$count_before" "$(find "$(latest data1)" -type f ! -name '.complete' | wc -l)"
chk "E5.5 失败源不影响后续源（data2 仍备份）" [ "$(snaps data2)" -gt 1 ]
fi

# ============ 7. E5b 掉电窗口：已 complete 未改名 → 提升 ============
if sec 7; then
printf '\n=== E5b 掉电窗口：已 complete 未改名的残留提升 ===\n'
i=1; while [ "$i" -le 21 ]; do printf 'data1 file %s\n' "$i" > "$SB/src/data1/f$i.txt"; i=$((i+1)); done
mkdir -p "$SB/src/data1/.cache" "$SB/src/data1/lost+found"
printf 'junk\n' > "$SB/src/data1/.cache/junk.bin"; printf 'junk\n' > "$SB/src/data1/lost+found/junk"; printf 'tmpjunk\n' > "$SB/src/data1/scratch.tmp"
prom="$SB/back/data1/.2026-12-31_235959.inprogress.9999"; mkdir -p "$prom"; : > "$prom/.complete"; printf 'promoted\n' > "$prom/promoted.txt"
run_script "$SB/rsync-test.sh"; rc=$?
eq "E5b.1 退出码 0" "0" "$rc"
chkgn "E5b.2 日志记录提升" "提升上次已完成但未改名的快照"
chk "E5b.3 提升为正式快照（含 .complete）" [ -f "$SB/back/data1/2026-12-31_235959/.complete" ]
chk "E5b.4 提升内容完整" [ -f "$SB/back/data1/2026-12-31_235959/promoted.txt" ]
chk "E5b.5 inprogress 原目录已消失" [ ! -e "$prom" ]
fi

# ============ 8. E6 崩溃续传（按完成度挑选） ============
if sec 8; then
printf '\n=== E6 崩溃续传（按完成度挑选，非时间戳）===\n'
r1="$SB/back/data1/.2026-10-14_080000.inprogress.7777"; mkdir -p "$r1"
i=1; while [ "$i" -le 8 ]; do printf 'stale\n' > "$r1/f$i.txt"; i=$((i+1)); done
r2="$SB/back/data1/.2026-10-14_090000.inprogress.6666"; mkdir -p "$r2"; : > "$r2/only"
# 前后差分锁定本轮新建的快照：不受未来日期提升快照干扰，也不依赖硬编码当天日期
e6_before="$(find "$SB/back/data1" -mindepth 1 -maxdepth 1 -type d -name '20*' ! -name '*.failed.*' | sort)"
run_script "$SB/rsync-test.sh"; rc=$?
e6_new="$(comm -13 <(printf '%s\n' "$e6_before") <(find "$SB/back/data1" -mindepth 1 -maxdepth 1 -type d -name '20*' ! -name '*.failed.*' | sort) | head -n 1)"
eq "E6.1 退出码 0" "0" "$rc"
chkgn "E6.2 工作目录即原厚残留（续传而非新建）" "工作目录: $r1"
chkgn "E6.3 薄残留标记失败" "多余的未完成目录，标记为失败"
chk "E6.4 薄残留已成 .failed 目录" [ -n "$(find "$SB/back/data1" -maxdepth 1 -type d -name '2026-10-14_090000.failed.*')" ]
chk "E6.5 厚残留已完成并改名" [ ! -e "$r1" ]
eq "E6.6 续传目录中的陈旧文件被刷新为当前内容" "data1 file 8" "$(cat "$e6_new/f8.txt")"
fi

# ============ 9. E7 rsync 退出码分类 ============
if sec 9; then
printf '\n=== E7a rsync 退出码 23（warn：保留快照但报警）===\n'
run_script "$SB/rsync-test.sh" MOCK_RSYNC_RC=23; rc=$?
eq "E7a.1 退出码 1" "1" "$rc"
chkgn "E7a.2 日志记录警告" "备份完成但有警告"
chk "E7a.3 快照仍然生成" [ "$(snaps data1)" -ge 1 ]
printf '\n=== E7b rsync 退出码 11（fail：标记失败）===\n'
b1="$(snaps data1)"; b2="$(snaps data2)"
run_script "$SB/rsync-test.sh" MOCK_RSYNC_RC=11; rc=$?
eq "E7b.1 退出码 1" "1" "$rc"
chkgn "E7b.2 日志记录失败" "备份失败"
eq "E7b.3 data1 未增加快照" "$b1" "$(snaps data1)"
eq "E7b.4 data2 未增加快照" "$b2" "$(snaps data2)"
chk "E7b.5 生成 .failed 目录" [ -n "$(find "$SB/back/data1" -maxdepth 1 -type d -name '*.failed.*')" ]
chk "E7b.6 无残留 inprogress" [ -z "$(find "$SB/back/data1" -maxdepth 1 -type d -name '.*inprogress.*')" ]
fi

# ============ 10. E8 启动自检与输入校验 ============
if sec 10; then
printf '\n=== E8 启动自检与输入校验 ===\n'
gen_copy "$SB/rt-c1.sh" KEEP=abc
run_script "$SB/rt-c1.sh"; rc=$?
eq "E8.1 KEEP=abc 拒绝启动" "1" "$rc"
chk "E8.1b 提示非负整数" grep -q "KEEP 必须是非负整数" "$OUT"
gen_copy "$SB/rt-c2.sh" KEEP=0
run_script "$SB/rt-c2.sh"; rc=$?
eq "E8.2 KEEP=0 拒绝启动" "1" "$rc"
chk "E8.2b 提示正整数" grep -q "KEEP 必须是正整数" "$OUT"
gen_copy "$SB/rt-c3.sh" REJECT_SRC_ON_ROOTFS=yes
run_script "$SB/rt-c3.sh"; rc=$?
eq "E8.3 布尔开关填 yes 被拒" "1" "$rc"
chk "E8.3b 提示只能是 0 或 1" grep -q "REJECT_SRC_ON_ROOTFS 只能是 0 或 1" "$OUT"
gen_copy "$SB/rt-c4.sh" SHRINK_RATIO=x
run_script "$SB/rt-c4.sh"; rc=$?
eq "E8.4 SHRINK_RATIO=x 被拒" "1" "$rc"
chk "E8.4b 提示非负整数" grep -q "SHRINK_RATIO 必须是非负整数" "$OUT"
gen_copy "$SB/rt-c5.sh"
sed -i 's/|data2"/|data1"/' "$SB/rt-c5.sh"
run_script "$SB/rt-c5.sh"; rc=$?
eq "E8.5 NAME 重复拒绝启动" "1" "$rc"
chk "E8.5b 提示 NAME 重复" grep -q "SOURCES 中 NAME 重复" "$OUT"
gen_copy "$SB/rt-c6.sh"
sed -i 's/|data2"/data2"/' "$SB/rt-c6.sh"
run_script "$SB/rt-c6.sh"; rc=$?
eq "E8.6 缺 | 分隔符拒绝启动" "1" "$rc"
chk "E8.6b 提示格式错误" grep -q "SOURCES 项格式错误" "$OUT"
gen_copy "$SB/rt-c7.sh"
sed -i 's/|data2"/|data2|extra"/' "$SB/rt-c7.sh"
run_script "$SB/rt-c7.sh"; rc=$?
eq "E8.7 三段配置退出 1" "1" "$rc"
chkgn "E8.7b 提示无效源配置" "无效源配置"
gen_copy "$SB/rt-c8.sh"
sed -i '/^SOURCES=(/,/^)$/c\SOURCES=()' "$SB/rt-c8.sh"
run_script "$SB/rt-c8.sh"; rc=$?
eq "E8.8 SOURCES 为空拒绝启动" "1" "$rc"
chk "E8.8b 提示 SOURCES 为空" grep -q "SOURCES 为空" "$OUT"
gen_copy "$SB/rt-c9.sh"
sed -i 's/|data1"/|..\/evil"/' "$SB/rt-c9.sh"
run_script "$SB/rt-c9.sh"; rc=$?
eq "E8.9 坏 NAME 退出 1" "1" "$rc"
chkgn "E8.9b 提示无效备份名" "无效备份名"
chk "E8.9c 未逃出 BACKUP_BASE" [ ! -e "$SB/back/../evil" ]
gen_copy "$SB/rt-c10.sh"
sed -i 's#^  ".*|data1"$#  "/|root"#' "$SB/rt-c10.sh"
run_script "$SB/rt-c10.sh"; rc=$?
eq "E8.10 SRC=/ 拒绝" "1" "$rc"
chkgn "E8.10b 提示拒绝备份根目录" "拒绝备份根目录"
gen_copy "$SB/rt-c11.sh"
sed -i "s#^  \".*|data1\"\$#  \"$SB/src/nope|nope\"#" "$SB/rt-c11.sh"
run_script "$SB/rt-c11.sh"; rc=$?
eq "E8.11 源不存在退出 1" "1" "$rc"
chkgn "E8.11b 提示源不存在" "源不存在或不是目录"
mkdir -p "$SB/src/empty-src"
gen_copy "$SB/rt-c12.sh"
sed -i "s#^  \".*|data1\"\$#  \"$SB/src/empty-src|empty\"#" "$SB/rt-c12.sh"
run_script "$SB/rt-c12.sh"; rc=$?
eq "E8.12 空源退出 1" "1" "$rc"
chkgn "E8.12b 提示源为空" "源为空"
fi

# ============ 11. E9 并发互斥 ============
if sec 11; then
printf '\n=== E9 并发互斥（mock flock）===\n'
chk "E9.1 锁文件已创建" [ -f "$SB/lock/rsync-backup.lock" ]
run_script "$SB/rsync-test.sh" MOCK_FLOCK_BUSY=1; rc=$?
eq "E9.2 第二个实例退出 1" "1" "$rc"
chkgn "E9.3 日志记录已有实例运行" "已有备份在运行"
fi

# ============ 12. E10 日志轮转与超龄清理 ============
if sec 12; then
printf '\n=== E10 日志轮转与超龄清理（LOG_MAX_SIZE=500）===\n'
printf 'z%.0s' $(seq 1 900) > "$LOG"
run_script "$SB/rsync-test.sh"; rc=$?
chk "E10.1 超限主日志被轮转" [ -n "$(find "$SB/log" -name 'rsync-backup.log.*' -type f)" ]
recent="$(find "$SB/log" -name 'rsync-backup.log.*' -type f | sort | head -n 1)"
touch -d "40 days ago" "$SB/log/rsync-backup.log.20990101000000"
run_script "$SB/rsync-test.sh"; rc=$?
chk "E10.2 超龄（>30 天）轮转文件被清" [ ! -f "$SB/log/rsync-backup.log.20990101000000" ]
chk "E10.3 未超龄轮转文件保留" [ -f "$recent" ]
fi

# ============ 13. E11 限速/超时参数变体 ============
if sec 13; then
printf '\n=== E11 限速/超时参数变体 ===\n'
gen_copy "$SB/rt-bw.sh" 'BW_LIMIT=1M' RSYNC_TIMEOUT=300
: > "$ARGV"
run_script "$SB/rt-bw.sh" MOCK_RSYNC_ARGV_LOG="$ARGV"; rc=$?
eq "E11.1 退出码 0" "0" "$rc"
chk "E11.2 argv 含 --bwlimit=1M" grep -q -- '--bwlimit=1M' "$ARGV"
chk "E11.3 argv 含 --timeout=300" grep -q -- '--timeout=300' "$ARGV"
chk "E11.4 argv 含 --one-file-system" grep -q -- '--one-file-system' "$ARGV"
chk "E11.5 快照正常生成" [ "$(snaps data2)" -ge 1 ]
fi

printf '\n===== 端到端测试合计（节 %s-%s）: PASS=%s FAIL=%s =====\n' "$FROM" "$TO" "$PASS" "$FAIL"
printf '（完整日志: %s ；stderr 捕获: %s ；rsync argv: %s ）\n' "$LOG" "$OUT" "$ARGV"
[ "$FAIL" -eq 0 ]
