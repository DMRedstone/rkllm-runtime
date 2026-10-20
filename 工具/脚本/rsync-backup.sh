#!/bin/bash
# =============================================================================
# rsync 硬链接快照备份
# -----------------------------------------------------------------------------
# 每个源目录在 $BACKUP_BASE/<名称>/ 下按时间戳留一份快照。
# 未改动的文件通过 --link-dest 硬链接到上一份成功快照，不占双倍空间。
#
# 典型 cron（每天 03:00）：
#   0 3 * * * /usr/local/sbin/rsync-backup.sh
#
# 依赖：bash 4.4+、GNU find、GNU df、rsync、flock（util-linux）
# 必须以 root 跑，否则 owner / ACL / xattr 无法完整保留。
# =============================================================================

# nounset：引用未定义变量即退出
# errexit：命令失败即退出（if/while 测试里的失败除外）
# pipefail：管道中任一命令失败，整段管道算失败
set -euo pipefail

# =============================================================================
# 可配置项 —— 只改这一段
# =============================================================================

# 快照根目录。必须已经挂在独立磁盘上（见 MOUNT_CHECKS），避免写到系统盘。
BACKUP_BASE="/back/rsync-snapshots"

# 每个源只保留最近几份「成功」快照。超出的从最旧的开始删。
# 硬链接占用 inode，机械盘上 KEEP 过大可能先耗尽 inode 而不是空间。
KEEP=8

# 失败目录（*.failed.*）按天数淘汰。
FAILED_KEEP_DAYS=14

# 失败目录同时按数量封顶，防止一次次失败把盘写满。
# 与 FAILED_KEEP_DAYS 同时生效：超龄或超出数量都会删。0 表示不按数量限制。
FAILED_KEEP_COUNT=2

# 主日志。每次 log() 同时追加到此文件和 stderr（方便 cron 寄信 / journal）。
LOG="/var/log/rsync-backup.log"

# 防重叠锁。flock 是进程退出即释放的劝告锁，不会出现「上次崩溃后锁死」。
LOCK="/var/lock/rsync-backup.lock"

# rsync 带宽上限，空字符串表示不限制。
# SMR / 归档盘建议 "30M"，避免持续满速把盘写到掉队。单位随 rsync 版本：3.1+ 可用 30M。
BW_LIMIT=""

# rsync I/O 超时（秒）。0 表示不设。
# 本地机械盘建议保持 0：SMR 偶发卡顿会被误杀。NFS / 不稳定磁盘可设 300。
RSYNC_TIMEOUT=0

# 不跟随源上的其他挂载点（bind mount、意外挂进来的盘）。
# 1 打开对应 rsync -x。源就是要跨盘备份时改为 0。
ONE_FILE_SYSTEM=1

# 非 root 直接退出。-aHAX --numeric-ids 在非 root 下会静默丢 owner/ACL。
REQUIRE_ROOT=1

# 禁止把 SRC=/ 当源，避免一次配错把整个根文件系统灌进备份盘。
ALLOW_ROOT_SRC=0

# 源与 / 若在同一块设备上则拒绝。
# /data 未挂载时，挂载点空目录落在根文件系统上，这一项能拦住「备份空目录 + --delete 删光历史」。
# 若数据确实就在根分区上，改为 0。
# 注意：btrfs 子卷的 st_dev 与父卷相同（属同一文件系统），源若是独立 btrfs 子卷会被误判为未挂载。
#       这种场景请置 0，改由 MOUNT_CHECKS 把关（mountpoint 能识别子卷）。
REJECT_SRC_ON_ROOTFS=1

# 源在 maxdepth=1 下没有任何条目（忽略 lost+found）则拒绝。代价只是列一层目录。
CHECK_SRC_EMPTY=1

# 源文件数 < 上次成功快照文件数 / N 则拒绝，防止挂错盘、源被清空后 --delete 做出空快照再把旧快照淘汰掉。
# 源侧仍要在 rsync 前 walk 一遍；上一份快照的文件数读 .filecount 缓存，不再重走。
SHRINK_RATIO=10

# 备份盘可用空间 / 可用 inode 低于该百分比则拒绝开跑。0 表示不检查。
MIN_FREE_PCT=5
MIN_FREE_INODE_PCT=5

# rsync 这些退出码仍然把目录提升为正式快照。
# 0  = 完全成功
# 24 = 传输中源文件消失（活跃系统常见，可接受）
OK_RSYNC_CODES=(0 24)

# 这些退出码：快照保留（已经拷了大部分），但 overall_rc=1 以便 cron 报警。
# 23 = 部分文件失败（权限、读错误等）。不要整棵改名为 .failed，否则既占空间又没有可用点。
WARN_RSYNC_CODES=(23)

# 成功快照的顶层目录去掉写权限，降低非 root 进程误删/误写的风险。
# 注意：脚本以 root 运行（REQUIRE_ROOT=1），root 受 CAP_DAC_OVERRIDE 保护、不受权限位约束，
#       所以这一项防不住 root 自身的误操作，只对其他身份有效。别高估它的强度。
# 只 chmod 顶层，不递归（数百万文件上 chmod -R 会把机械盘卡很久）。
# 删除旧快照前会先 chmod u+w 顶层，否则 rm -rf 无法进入。
CHMOD_SNAPSHOT_RO=1

# 日志超过该字节就轮转。0 表示不按大小切。
LOG_MAX_SIZE=$((10 * 1024 * 1024))

# 已轮转日志保留天数，超时删除。0 表示不删旧日志。
LOG_KEEP_DAYS=30

# rsync 排除规则。不要把 "--exclude=" 漏掉，数组元素就是完整参数。
EXCLUDES=(
  "--exclude=.cache/"
  "--exclude=*.tmp"
  "--exclude=lost+found/"
)

# 源目录|备份子目录名。NAME 只允许 [A-Za-z0-9._-]，禁止以点开头（避免跟 .inprogress 混淆）。
SOURCES=(
  "/data/nextdata|nextdata"
  "/data/otherdata|otherdata"
)

# 必须已挂载的路径。未挂载就退出，防止：
#   1) 备份根目录掉到系统盘；
#   2) 源是空挂载点，rsync --delete 生成空快照。
# 填真正的 mount 点（findmnt 能看到的），不要填挂载点下面的子目录。
MOUNT_CHECKS=(
  "/back"
  "/data"
)

# 目标目录里出现「孤儿项」时的处置。孤儿项 = 既不是成功快照（20* 且含 .complete），
# 也不是 .failed.* / .inprogress.* / _orphaned.* 等机制目录——通常是手工放进去的文件、
# 旧工具残留，或「完全重建 / 首次接管他人备份目录」时留下的东西。
#   ask     有终端时逐个询问（m=移走 d=删除 s=跳过）；无终端（cron）等同 skip 并告警
#   move    自动移到 $BACKUP_BASE/<名称>/_orphaned.<时间戳>/（保留可回溯）
#   delete  自动删除（清单先写日志）
#   skip    不处理
ORPHAN_ACTION="ask"

# =============================================================================
# 内部状态 —— 下面不要改
# =============================================================================

# trap 用这三项定位「正在写入、尚未改名为正式快照」的目录。
CURRENT_BASE=""
CURRENT_DATE=""
CURRENT_WORK=""

# 任意源失败都记下来，最后以该值退出；单个源失败不耽误后续源。
overall_rc=0

# cron 的 PATH 经常只有 /usr/bin，ionice/mountpoint 可能在 /usr/sbin。
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# =============================================================================
# 基础工具函数
# =============================================================================

# 同时写入日志文件和 stderr。写文件失败不得把备份进程打死（盘满时还要把 inprogress 改名）。
log() {
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') [$$] $*"
  # 目录可能尚未建好、或 /var 已满：忽略写文件错误，stderr 至少还能进 cron 邮件。
  printf '%s\n' "$line" >> "$LOG" 2>/dev/null || true
  printf '%s\n' "$line" >&2
}

# 致命错误：记日志并让 EXIT trap 做清理。
die() {
  log "致命错误: $*"
  exit 1
}

# 按大小切分主日志，再按天数清掉过旧的轮转文件。必须在 flock 之后调用，避免并发 mv。
rotate_logs() {
  local current_size=0
  local log_dir log_base log_target

  log_dir="$(dirname "$LOG")"
  log_base="$(basename "$LOG")"

  if (( LOG_MAX_SIZE > 0 )) && [ -f "$LOG" ]; then
    # GNU stat；失败当 0，避免把非数字喂给 (( ))。
    current_size="$(stat -c%s "$LOG" 2>/dev/null || echo 0)"
    if [[ "$current_size" =~ ^[0-9]+$ ]] && (( current_size > LOG_MAX_SIZE )); then
      # 时间戳只到秒：同一秒内轮转两次会得到同名目标，mv 直接覆盖前一份
      # （日志静默丢失）。先试纯时间戳保持常见命名不变，撞名再加 pid。
      log_target="$LOG.$(date +%Y%m%d%H%M%S)"
      [ -e "$log_target" ] && log_target="$log_target.$$"
      mv "$LOG" "$log_target" 2>/dev/null || true
    fi
  fi

  if (( LOG_KEEP_DAYS > 0 )); then
    # 只删「主日志名.后缀」，不会碰到正在写的 $LOG。
    # -mtime +N 表示「修改时间超过 N 天」；用 +N 保留完整 N 天。
    find "$log_dir" -maxdepth 1 -type f -name "${log_base}.*" \
      -mtime +"$LOG_KEEP_DAYS" -delete 2>/dev/null || true
  fi
}

# 成功快照目录名列表，新的在前。
# 判定成功：名字像 20* 时间戳、不是 *.failed.* / *.recovered.*、里面有 .complete 标记文件。
list_complete_snapshots() {
  local base="$1"
  find "$base" -maxdepth 1 -mindepth 1 -type d \
    -name '20*' ! -name '*.failed.*' \
    -exec test -f '{}/.complete' \; \
    -printf '%f\0' 2>/dev/null \
    | LC_ALL=C sort -z -r || true
}

# 统计文件数。-printf '.' | wc -c 避免 wc -l 按行、大树时少造很多换行。
# ONE_FILE_SYSTEM=1 时加 -xdev，与 rsync -x 看到的范围一致。
count_files() {
  local dir="$1"
  local n
  if (( ONE_FILE_SYSTEM )); then
    n="$(find "$dir" -xdev -type f -printf '.' 2>/dev/null | wc -c)" || n=0
  else
    n="$(find "$dir" -type f -printf '.' 2>/dev/null | wc -c)" || n=0
  fi
  # GNU wc 可能在数字前补空格
  printf '%s\n' "${n// /}"
}

# 上一份快照的文件数：优先读快照完成时写入的 .filecount。
# 该值取自本次 rsync --stats 的 "Number of files (reg: N)"，与 find -type f
# 口径一致（都已应用 exclude 与 -x），所以无需再对旧快照走一遍全树。
# 老快照（本机制上线前生成的）没有该文件，回退到遍历；不补写——它们已是只读。
count_snapshot_files() {
  local dir="$1"
  local n
  if [ -r "$dir/.filecount" ]; then
    n="$(< "$dir/.filecount")"
    if [[ "$n" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$n"
      return 0
    fi
  fi
  count_files "$dir"
}

# 整数是否在数组里。用于 rsync 退出码分类。
in_array() {
  local needle="$1"
  local x
  shift
  for x in "$@"; do
    if [ "$x" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

# 把 rsync 退出码分成 ok / warn / fail。
rsync_class() {
  local rc="$1"
  if in_array "$rc" "${OK_RSYNC_CODES[@]}"; then
    printf '%s\n' "ok"
  elif in_array "$rc" "${WARN_RSYNC_CODES[@]}"; then
    printf '%s\n' "warn"
  else
    printf '%s\n' "fail"
  fi
}

# 空间/inode 不足的统一出口。
# soft=1：只记日志并返回 1，不终止脚本——供「逐源检查」跳过当前源、继续下一个源。
# soft=0：直接 die，用于启动前的一次性检查。
space_shortage() {
  local msg="$1"
  local soft="${2:-0}"
  if (( soft )); then
    log "$msg（跳过当前源，后续源继续）"
    return 1
  fi
  die "$msg"
}

# 备份盘空间与 inode。硬链接快照经常先把 inode 用光。
# 用 df --output 固定字段顺序，避免文件系统名（NFS/CIFS）含空格时列错位。
# 关键：GNU df 的 --output 与 -i / -P / -T 互斥，只能二选一。
#   - inode 字段用 --output=itotal,iavail 指定，不能再加 -i；
#   - 固定列序由 --output 本身保证，不能再加 -P（加了会直接报错退出）。
# soft=1 时软失败，见 space_shortage。
check_free_space() {
  local path="$1"
  local soft="${2:-0}"
  local total avail
  local itotal ifree

  if (( MIN_FREE_PCT > 0 )); then
    read -r total avail < <(
      df -B1 --output=size,avail "$path" 2>/dev/null | tail -n 1
    ) || true
    if [[ "${total:-0}" =~ ^[0-9]+$ ]] && [[ "${avail:-0}" =~ ^[0-9]+$ ]] \
       && (( total > 0 )); then
      # df -B1 给的是字节，avail*100 在 9.2e18 处整数溢出：92PB 以上的盘
      # 会把「空间不足」静默算成永假（等于关掉检查）。awk 用 double，不回绕。
      if awk -v a="$avail" -v t="$total" -v p="$MIN_FREE_PCT" \
           'BEGIN { exit !(a * 100 < t * p) }'; then
        space_shortage "备份盘空间不足: $path 可用 $avail / 总共 $total （阈值 ${MIN_FREE_PCT}%）" "$soft" || return 1
      fi
    else
      # 解析不出来必须留痕，否则又是一次「静默跳过检查」
      log "无法解析 df 空间输出，本次跳过空间检查: $path"
    fi
  fi

  if (( MIN_FREE_INODE_PCT > 0 )); then
    read -r itotal ifree < <(
      df --output=itotal,iavail "$path" 2>/dev/null | tail -n 1
    ) || true
    # 部分网络盘没有 inode，itotal 会是 0 或 -
    if [[ "${itotal:-0}" =~ ^[0-9]+$ ]] && [[ "${ifree:-0}" =~ ^[0-9]+$ ]] \
       && (( itotal > 0 )); then
      # 同上：inode 数在超大文件系统上同样会溢出
      if awk -v a="$ifree" -v t="$itotal" -v p="$MIN_FREE_INODE_PCT" \
           'BEGIN { exit !(a * 100 < t * p) }'; then
        space_shortage "备份盘 inode 不足: $path 可用 $ifree / 总共 $itotal （阈值 ${MIN_FREE_INODE_PCT}%）" "$soft" || return 1
      fi
    else
      log "无法解析 df inode 输出，本次跳过 inode 检查: $path"
    fi
  fi
  return 0
}

# 处理上次崩溃留下的 .*.inprogress.*：
#   - 已有 .complete：提升为正式快照（touch 与 mv 之间掉电会出现这种目录）
#     若目标已存在，视为重复数据，归入 .failed 交给 prune_failed 清理。
#   - 没有 .complete：挑「已完成程度最高」的一份给本次续传，其余改名为 .failed
#     判定依据是文件数而不是时间戳：连续崩溃时最新的那份可能才跑了 5%，
#     更早的一份却已经跑完 90%，按时间戳取最新会把绝大部分已完成的工作白扔掉。
# 禁止无条件 rm -rf：大目录 90% 时中断，续传比从头跑更重要。
# 列出目标目录下的孤儿项（  分隔）：非成功快照、非机制目录。
list_orphans() {
  local base="$1" p name
  find "$base" -maxdepth 1 -mindepth 1 -print0 2>/dev/null | while IFS= read -r -d '' p; do
    name="${p##*/}"
    case "$name" in
      .20*.inprogress.*) continue ;;              # 崩溃残留：handle_leftovers 负责
      20*.failed.*|20*.recovered.*) continue ;;   # 失败/恢复目录：淘汰逻辑负责
      _orphaned.*) continue ;;                    # 本脚本自己的归档区，不再二次处置
    esac
    if [ -d "$p" ] && [ -f "$p/.complete" ]; then continue; fi   # 成功快照
    printf '%s\0' "$p"
  done
}

# 处置孤儿项：ask / move / delete / skip。返回 0 表示流程可继续（失败只记日志）。
handle_orphans() {
  local base="$1" action="${ORPHAN_ACTION:-ask}"
  local -a items=()
  mapfile -d '' -t items < <(list_orphans "$base")
  ((${#items[@]})) || return 0

  local p n kind
  log "目标目录存在 ${#items[@]} 项非快照残留（$base）："
  for p in "${items[@]}"; do
    n="$(basename "$p")"
    if [ -d "$p" ]; then kind="目录"; else kind="文件"; fi
    log "  [$kind] $n"
  done

  # 非交互（cron / 管道）下 ask 降级为 skip：备份不能因为一个提问而挂住
  if [ "$action" = ask ] && [ ! -t 0 ]; then
    log "非交互环境无法询问，按 skip 处理（可设 ORPHAN_ACTION=move 或 delete）"
    action=skip
  fi
  if [ "$action" = ask ]; then
    local reply=""
    log "如何处置？m=移到 _orphaned.<时间戳>/  d=删除  s=跳过"
    read -r -p "输入 m/d/s [s]: " reply || reply=s
    case "${reply:-s}" in
      m|M) action=move ;;
      d|D) action=delete ;;
      *)   action=skip ;;
    esac
  fi

  case "$action" in
    move)
      local dest="$base/_orphaned.$(date +%Y-%m-%d_%H%M%S)"
      if ! mkdir -p "$dest"; then
        log "创建归档目录失败: $dest（残留项保持不动）"
        return 0
      fi
      for p in "${items[@]}"; do
        if mv "$p" "$dest/" 2>/dev/null; then
          log "已移走: $(basename "$p") -> $dest"
        else
          log "移走失败: $p"
        fi
      done
      ;;
    delete)
      for p in "${items[@]}"; do
        if rm -rf "$p" 2>/dev/null; then
          log "已删除: $p"
        else
          log "删除失败: $p"
        fi
      done
      ;;
    skip)
      log "按配置跳过残留项（未做任何处理，下次运行仍会提示）"
      ;;
  esac
  return 0
}

handle_leftovers() {
  local base="$1"
  local -n _resume_out="$2"
  local leftover path dest_name dest failed
  local best best_n n
  local -a leftovers=()
  local -a incomplete=()

  _resume_out=""

  mapfile -d '' -t leftovers < <(
    find "$base" -maxdepth 1 -mindepth 1 -type d -name '.20*.inprogress.*' \
      -printf '%f\0' 2>/dev/null | LC_ALL=C sort -z -r || true
  )

  # 第一遍：清掉已完成的残留，把未完成的收集起来
  for leftover in "${leftovers[@]}"; do
    [ -z "$leftover" ] && continue
    path="$base/$leftover"
    # .2026-09-17_080102.inprogress.12345 → 2026-09-17_080102
    dest_name="${leftover#.}"
    dest_name="${dest_name%.inprogress.*}"

    if [ -f "$path/.complete" ]; then
      dest="$base/$dest_name"
      if [ -e "$dest" ]; then
        # 目标已存在，本份是重复数据：标记为 failed，交给 prune_failed 按天/按数量清理
        failed="$base/${dest_name}.failed.$$"
        log "残留 inprogress 已 complete 但目标已存在，改为 failed: $path -> $failed"
        if ! mv "$path" "$failed" 2>/dev/null; then
          log "移动残留目录失败: $path"
        fi
      else
        log "提升上次已完成但未改名的快照: $path -> $dest"
        if mv "$path" "$dest"; then
          if (( CHMOD_SNAPSHOT_RO )); then
            chmod a-w "$dest" 2>/dev/null || true
          fi
        else
          log "提升残留快照失败: $path"
        fi
      fi
    else
      incomplete+=("$leftover")
    fi
  done

  (( ${#incomplete[@]} )) || return 0

  # 第二遍：在未完成的里面挑最「厚」的一份续传
  if (( ${#incomplete[@]} == 1 )); then
    # 只有一份就不必再遍历整棵树
    best="${incomplete[0]}"
  else
    best=""
    best_n=-1
    for leftover in "${incomplete[@]}"; do
      n="$(find "$base/$leftover" -xdev -type f -printf '.' 2>/dev/null | wc -c)" || n=0
      n="${n// /}"
      [[ "$n" =~ ^[0-9]+$ ]] || n=0
      log "未完成目录 $leftover 已含 $n 个文件"
      if (( n > best_n )); then
        best_n="$n"
        best="$leftover"
      fi
    done
  fi

  for leftover in "${incomplete[@]}"; do
    path="$base/$leftover"
    dest_name="${leftover#.}"
    dest_name="${dest_name%.inprogress.*}"
    if [ "$leftover" = "$best" ]; then
      _resume_out="$path"
      log "发现未完成快照，本次续传: $path"
    else
      log "多余的未完成目录，标记为失败: $path"
      mv "$path" "$base/${dest_name}.failed.$$" 2>/dev/null || log "移动多余 inprogress 失败: $path"
    fi
  done
}

# 每个源只留最近 KEEP 份成功快照。
prune_snapshots() {
  local base="$1"
  local -a snapshots=()
  local count i old

  mapfile -d '' -t snapshots < <(list_complete_snapshots "$base")
  count="${#snapshots[@]}"

  if (( count > KEEP )); then
    for ((i = KEEP; i < count; i++)); do
      [ -z "${snapshots[$i]:-}" ] && continue
      old="$base/${snapshots[$i]}"
      log "删除旧快照 $old"
      # 顶层曾 chmod a-w，必须先加回写权限，rm 才能删目录项
      chmod u+w "$old" 2>/dev/null || true
      if ! "${RUN[@]}" rm -rf -- "$old"; then
        log "删除旧快照失败: $old"
        overall_rc=1
      fi
    done
  else
    log "快照数 $count <= 保留数 $KEEP，无需清理"
  fi
}

# 失败目录：超龄删除；仍超 FAILED_KEEP_COUNT 则从最旧的开始再删。
prune_failed() {
  local base="$1"
  local f
  local -a old_failed=() leftover_failed=()

  mapfile -d '' -t old_failed < <(
    find "$base" -maxdepth 1 -mindepth 1 -type d -name '*.failed.*' \
      -mtime +"$FAILED_KEEP_DAYS" -print0 2>/dev/null || true
  )
  for f in "${old_failed[@]}"; do
    [ -z "$f" ] && continue
    log "删除超龄失败目录 $f"
    chmod u+w "$f" 2>/dev/null || true
    if ! "${RUN[@]}" rm -rf -- "$f"; then
      log "删除超龄失败目录失败: $f"
      overall_rc=1
    fi
  done

  if (( FAILED_KEEP_COUNT > 0 )); then
    mapfile -d '' -t leftover_failed < <(
      find "$base" -maxdepth 1 -mindepth 1 -type d -name '*.failed.*' \
        -printf '%f\0' 2>/dev/null | LC_ALL=C sort -z -r || true
    )
    local n="${#leftover_failed[@]}"
    local i
    if (( n > FAILED_KEEP_COUNT )); then
      for ((i = FAILED_KEEP_COUNT; i < n; i++)); do
        [ -z "${leftover_failed[$i]:-}" ] && continue
        f="$base/${leftover_failed[$i]}"
        log "失败目录超过 ${FAILED_KEEP_COUNT} 个，删除旧的 $f"
        chmod u+w "$f" 2>/dev/null || true
        if ! "${RUN[@]}" rm -rf -- "$f"; then
          log "删除多余失败目录失败: $f"
          overall_rc=1
        fi
      done
    fi
  fi
}

# 把未完成目录改名为 .failed.<pid>。trap 与备份失败路径共用。
mark_failed() {
  local work="$1"
  local base="$2"
  local date_tag="$3"
  local failed

  if [ -n "$work" ] && [ -d "$work" ]; then
    failed="$base/${date_tag}.failed.$$"
    log "标记不完整快照: $work -> $failed"
    mv "$work" "$failed" 2>/dev/null || log "重命名不完整快照失败: $work"
  fi
}

# EXIT 时若 CURRENT_WORK 仍在，说明 rsync 中途被杀或脚本异常退出。
cleanup() {
  # 必须先保存退出码：set +e 本身成功，会把 $? 改成 0
  local rc=$?
  # trap 里不要再被 set -e 打断，否则 inprogress 来不及改名
  set +e
  if [ -n "${CURRENT_WORK:-}" ] && [ -d "$CURRENT_WORK" ]; then
    mark_failed "$CURRENT_WORK" "$CURRENT_BASE" "$CURRENT_DATE"
    CURRENT_WORK=""
  fi
  # EXIT trap 里用 exit 保留原退出码，比 return 更稳
  exit "$rc"
}

# =============================================================================
# 单个源的备份
# 失败只置 overall_rc=1 并 return 0：set -e 下不能 return 非零，否则后续源不会跑。
# =============================================================================

backup_one() {
  local entry="$1"
  local SRC NAME EXTRA
  local BASE DATE DEST WORK LATEST
  local -a LINK_OPT=() RSYNC_OPTS=()
  local rc class t0 src_count prev_count src_dev root_dev entries _reg stats_file
  local RESUME_WORK=""

  # IFS 只作用于这条 read，不会改掉整个脚本的 IFS
  IFS='|' read -r SRC NAME EXTRA <<< "$entry"
  if [ -z "${SRC:-}" ] || [ -z "${NAME:-}" ] || [ -n "${EXTRA:-}" ]; then
    log "无效源配置（需要恰好 SRC|NAME）: $entry"
    overall_rc=1
    return 0
  fi

  # 防 NAME 写成 ../etc 或 .hidden 逃出 BACKUP_BASE
  if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$NAME" == .* ]]; then
    log "无效备份名: $NAME"
    overall_rc=1
    return 0
  fi

  if (( ! ALLOW_ROOT_SRC )) && [ "$SRC" = "/" ]; then
    log "拒绝备份根目录 / （ALLOW_ROOT_SRC=0）"
    overall_rc=1
    return 0
  fi

  if [ ! -d "$SRC" ]; then
    log "源不存在或不是目录: $SRC"
    overall_rc=1
    return 0
  fi

  # 未挂载的挂载点落在根文件系统上。stat %d 是设备号，不随路径字符串变化。
  # -L 跟随符号链接：SRC 若是指向其他盘的符号链接，必须看目标设备号。
  if (( REJECT_SRC_ON_ROOTFS )); then
    src_dev="$(stat -L -c '%d' "$SRC")"
    root_dev="$(stat -L -c '%d' /)"
    if [ "$src_dev" = "$root_dev" ]; then
      log "源 $SRC 与 / 同一设备，可能未挂载，拒绝备份"
      overall_rc=1
      return 0
    fi
  fi

  # 只看一层：空盘或只有 lost+found 的未挂载 ext 都能拦住，几乎不耗时。
  if (( CHECK_SRC_EMPTY )); then
    entries="$(find "$SRC" -mindepth 1 -maxdepth 1 ! -name 'lost+found' -printf '.' 2>/dev/null | wc -c)" || entries=0
    entries="${entries// /}"
    if (( entries == 0 )); then
      log "源为空（忽略 lost+found）: $SRC，拒绝备份以免 --delete 清空历史"
      overall_rc=1
      return 0
    fi
  fi

  # 源与备份根互为包含时必须拒绝：任一方向成立，每份快照都会把上一份完整
  # 吞进去（--link-dest 只硬链接 identical 文件，路径层级不同照样复制），
  # KEEP 轮之内盘就被指数级吃光。pwd -P 先解掉符号链接再比，否则
  # /data -> /back/data 这类配置能绕过字符串比较。
  if [ -d "$BACKUP_BASE" ]; then
    local src_real base_real
    src_real="$(cd "$SRC" && pwd -P)"
    base_real="$(cd "$BACKUP_BASE" && pwd -P)"
    case "$src_real/" in
      "$base_real"/*)
        log "源 $SRC 位于备份根 $BACKUP_BASE 之内，拒绝备份（快照会自我吞并）"
        overall_rc=1
        return 0
        ;;
    esac
    case "$base_real/" in
      "$src_real"/*)
        log "备份根 $BACKUP_BASE 位于源 $SRC 之内，拒绝备份（备份盘会被当数据备份）"
        overall_rc=1
        return 0
        ;;
    esac
  fi

  BASE="$BACKUP_BASE/$NAME"
  mkdir -p "$BASE"
  handle_orphans "$BASE"   # 孤儿项检测（完全重建 / 首次接管旧目录时的残留处置）

  # 秒级时间戳碰撞时最多再等 3 秒，避免 overall_rc=1 却其实什么都没做错
  DATE="$(date +%Y-%m-%d_%H%M%S)"
  DEST="$BASE/$DATE"
  local tries=0
  while [ -e "$DEST" ] && (( tries < 3 )); do
    sleep 1
    DATE="$(date +%Y-%m-%d_%H%M%S)"
    DEST="$BASE/$DATE"
    tries=$((tries + 1))
  done
  if [ -e "$DEST" ]; then
    log "目标已存在，跳过: $DEST"
    overall_rc=1
    return 0
  fi

  handle_leftovers "$BASE" RESUME_WORK

  # 有未完成目录就在原目录上续传（rsync 可增量）；否则新建带 pid 的工作目录
  if [ -n "$RESUME_WORK" ] && [ -d "$RESUME_WORK" ]; then
    WORK="$RESUME_WORK"
  else
    WORK="$BASE/.${DATE}.inprogress.$$"
  fi

  # 续传时 WORK 必然已存在，所以只在「新建」路径上把已存在当异常
  if [ -z "$RESUME_WORK" ] && [ -e "$WORK" ]; then
    log "临时目录已存在，跳过: $WORK"
    overall_rc=1
    return 0
  fi

  # 上一份成功快照作为硬链接参考。必须用绝对路径（相对路径是相对 dest 解释的）。
  local -a all_snaps=()
  mapfile -d '' -t all_snaps < <(list_complete_snapshots "$BASE")
  LATEST="${all_snaps[0]:-}"

  if (( SHRINK_RATIO > 0 )) && [ -n "$LATEST" ] && [ -d "$BASE/$LATEST" ]; then
    log "检查源是否异常缩小（SHRINK_RATIO=$SHRINK_RATIO）..."
    src_count="$(count_files "$SRC")"
    prev_count="$(count_snapshot_files "$BASE/$LATEST")"
    if [[ "$src_count" =~ ^[0-9]+$ ]] && [[ "$prev_count" =~ ^[0-9]+$ ]] && (( prev_count > 0 )); then
      # 源文件数 * SHRINK_RATIO < 上次快照文件数，视为异常骤降（例如源只剩 lost+found 以外的几个文件）
      if (( src_count * SHRINK_RATIO < prev_count )); then
        log "源文件数异常缩小: $SRC 有 $src_count 个文件，上次快照 $LATEST 有 $prev_count 个，拒绝备份"
        overall_rc=1
        return 0
      fi
    fi
    log "源文件数 $src_count，上次快照 $prev_count，通过缩小检查"
  fi

  LINK_OPT=()
  if [ -n "$LATEST" ] && [ -d "$BASE/$LATEST" ]; then
    LINK_OPT=(--link-dest="$BASE/$LATEST")
  fi

  # 开跑前再查一次盘：启动时的检查只覆盖第一个源，前一个源可能已经把盘写满。
  # 用 soft 模式，空间不足时跳过本源而不是终止整轮，后面的源还能继续。
  if ! check_free_space "$BACKUP_BASE" 1; then
    overall_rc=1
    return 0
  fi

  log "开始备份 $SRC -> $DEST （参考: ${LATEST:-无}，工作目录: $WORK）"

  # -a  归档（权限、时间、符号链接、设备文件）
  # -H  保留源内部硬链接
  # -A  ACL    -X xattr
  # --numeric-ids  不把 uid/gid 映射成本机用户名，恢复到别的机器时更准
  # --delete       源里已删的文件，快照里也删（相对本次 dest，不影响旧快照）
  # --human-readable 让 --stats 更好读
  RSYNC_OPTS=(
    -aHAX
    --numeric-ids
    --delete
    # --partial-dir     中断文件放进隐藏子目录，不在 dest 留下半截文件名。
    #                   值为相对路径时 rsync 会自动在过滤规则末尾补一条 protect 规则，
    #                   不会被 --delete 提前删掉，因此无需（也不应）手工再追加过滤规则。
    --partial-dir=.rsync-partial
    --stats
    --human-readable
  )

  if (( ONE_FILE_SYSTEM )); then
    RSYNC_OPTS+=(--one-file-system)
  fi
  if [ -n "$BW_LIMIT" ]; then
    RSYNC_OPTS+=(--bwlimit="$BW_LIMIT")
  fi
  if (( RSYNC_TIMEOUT > 0 )); then
    RSYNC_OPTS+=(--timeout="$RSYNC_TIMEOUT")
  fi
  if (( ${#EXCLUDES[@]} )); then
    RSYNC_OPTS+=("${EXCLUDES[@]}")
  fi
  if (( ${#LINK_OPT[@]} )); then
    RSYNC_OPTS+=("${LINK_OPT[@]}")
  fi

  # 先登记再跑 rsync：中途 SIGTERM 时 trap 才找得到这个目录
  CURRENT_BASE="$BASE"
  CURRENT_DATE="$DATE"
  CURRENT_WORK="$WORK"

  t0=$SECONDS
  rc=0
  # rsync 输出要留一份原始副本：--stats 的 "Number of files" 会在快照完成时
  # 写进 .filecount，下次缩小检查直接读它，省掉对旧快照的一遍全树遍历。
  # 日志仍按原顺序整段追加，观察到的输出不变。
  stats_file="$BASE/.rsync-stats.$$"
  # if 测试中的失败不会触发 set -e；else 里的 $? 才是 rsync 真正的退出码
  if "${RUN[@]}" rsync "${RSYNC_OPTS[@]}" "$SRC/" "$WORK/" >"$stats_file" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  cat "$stats_file" >> "$LOG" 2>/dev/null || true
  class="$(rsync_class "$rc")"

  if [ "$class" = "fail" ]; then
    rm -f -- "$stats_file" 2>/dev/null || true
    log "备份失败 $SRC -> $WORK，退出码 $rc，耗时 $((SECONDS - t0))s"
    mark_failed "$WORK" "$BASE" "$DATE"
    CURRENT_WORK=""
    overall_rc=1
    prune_failed "$BASE"
    return 0
  fi

  # 不把 .rsync-partial 带进正式快照，否则会沿硬链接传到以后每一份
  rm -rf -- "$WORK/.rsync-partial" 2>/dev/null || true

  # 尽量把数据刷到盘上再打 .complete，缩小「标记完成但数据还在页缓存」的窗口
  if command -v sync >/dev/null 2>&1; then
    sync -f "$WORK" 2>/dev/null || sync 2>/dev/null || true
  fi

  # 把本次 rsync 看到的普通文件数写进快照，供下次缩小检查直接读取。
  # 用 --stats 而不是再 find 一遍：rsync 刚走完源，这里是零额外 I/O。
  # 兼容两种格式：真实 rsync 是 "Number of files: 1,234 (reg: 1,200, ...)"，
  # mock-rsync 只输出 "Number of files: 1234"。
  if [ -r "$stats_file" ]; then
    _reg="$(awk '
      /^Number of files:/ {
        if (match($0, /reg: *[0-9][0-9,]*/)) {
          v = substr($0, RSTART, RLENGTH); sub(/reg: */, "", v)
        } else {
          v = $0; sub(/^[^:]*: */, "", v); sub(/[^0-9,].*$/, "", v)
        }
        gsub(/,/, "", v); print v; exit
      }' "$stats_file" 2>/dev/null || true)"
    if [[ "$_reg" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$_reg" > "$WORK/.filecount" 2>/dev/null || true
    fi
  fi
  rm -f -- "$stats_file" 2>/dev/null || true

  if ! touch "$WORK/.complete"; then
    log "写入 .complete 失败: $WORK"
    mark_failed "$WORK" "$BASE" "$DATE"
    CURRENT_WORK=""
    overall_rc=1
    return 0
  fi

  # 同文件系统 rename 是原子的：要么还是 inprogress，要么已经是正式 DEST
  if ! mv "$WORK" "$DEST"; then
    log "完成快照改名失败: $WORK -> $DEST"
    mark_failed "$WORK" "$BASE" "$DATE"
    CURRENT_WORK=""
    overall_rc=1
    return 0
  fi

  # 改名成功后 trap 就不要再动它
  CURRENT_WORK=""

  if (( CHMOD_SNAPSHOT_RO )); then
    chmod a-w "$DEST" 2>/dev/null || true
  fi

  if [ "$class" = "warn" ]; then
    log "备份完成但有警告 $SRC -> $DEST，rsync 退出码 $rc，耗时 $((SECONDS - t0))s（快照已保留）"
    overall_rc=1
  else
    log "备份成功 $SRC -> $DEST，rsync 退出码 $rc，耗时 $((SECONDS - t0))s"
  fi

  prune_snapshots "$BASE"
  prune_failed "$BASE"
}

# =============================================================================
# 启动前检查
# =============================================================================

# bash 4.4 以下 set -u 遇到空数组 "${arr[@]}" 会直接退出
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "需要 bash 4.4 或更高版本，当前: ${BASH_VERSION:-未知}" >&2
  exit 1
fi

if (( REQUIRE_ROOT )) && [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "请以 root 运行（REQUIRE_ROOT=1）" >&2
  exit 1
fi

# 这些值一旦配错，在 (( )) 里会被当成未定义变量名求值为 0，
# 结果是「保护项被静默关闭」，所以全部在这里拦住。10# 前缀避免 08 被当成八进制。
for _pair in \
  "KEEP:$KEEP" \
  "FAILED_KEEP_DAYS:$FAILED_KEEP_DAYS" \
  "FAILED_KEEP_COUNT:$FAILED_KEEP_COUNT" \
  "SHRINK_RATIO:$SHRINK_RATIO" \
  "MIN_FREE_PCT:$MIN_FREE_PCT" \
  "MIN_FREE_INODE_PCT:$MIN_FREE_INODE_PCT" \
  "LOG_MAX_SIZE:$LOG_MAX_SIZE" \
  "LOG_KEEP_DAYS:$LOG_KEEP_DAYS" \
  "RSYNC_TIMEOUT:$RSYNC_TIMEOUT"
do
  _name="${_pair%%:*}"
  _val="${_pair#*:}"
  if ! [[ "$_val" =~ ^[0-9]+$ ]]; then
    echo "$_name 必须是非负整数" >&2
    exit 1
  fi
done
unset _pair _name _val

# 开关类只能是 0 或 1。填成 yes / on / 2 都会在 (( )) 里被静默当成 0，
# 等于把对应的安全检查关掉却毫无提示。
for _pair in \
  "ONE_FILE_SYSTEM:$ONE_FILE_SYSTEM" \
  "REQUIRE_ROOT:$REQUIRE_ROOT" \
  "ALLOW_ROOT_SRC:$ALLOW_ROOT_SRC" \
  "REJECT_SRC_ON_ROOTFS:$REJECT_SRC_ON_ROOTFS" \
  "CHECK_SRC_EMPTY:$CHECK_SRC_EMPTY" \
  "CHMOD_SNAPSHOT_RO:$CHMOD_SNAPSHOT_RO"
do
  _name="${_pair%%:*}"
  _val="${_pair#*:}"
  if [ "$_val" != "0" ] && [ "$_val" != "1" ]; then
    echo "$_name 只能是 0 或 1" >&2
    exit 1
  fi
done
unset _pair _name _val

if ! [[ "$KEEP" =~ ^[0-9]+$ ]] || (( 10#$KEEP < 1 )); then
  echo "KEEP 必须是正整数" >&2
  exit 1
fi
if ! [[ "$FAILED_KEEP_DAYS" =~ ^[0-9]+$ ]] || (( 10#$FAILED_KEEP_DAYS < 1 )); then
  echo "FAILED_KEEP_DAYS 必须是正整数" >&2
  exit 1
fi

if [ "${#SOURCES[@]}" -eq 0 ]; then
  echo "SOURCES 为空" >&2
  exit 1
fi

# NAME 重复会让两个源共用同一个快照目录，彼此把对方的快照 prune 掉；
# 表现为「备份都成功了，旧快照却莫名消失」，排查成本很高，这里直接拦住。
declare -A _seen_name=()
for _entry in "${SOURCES[@]}"; do
  _n="${_entry#*|}"
  _n="${_n%%|*}"
  if [ -z "$_n" ] || [ "$_n" = "$_entry" ]; then
    echo "SOURCES 项格式错误（需要 SRC|NAME）: $_entry" >&2
    exit 1
  fi
  if [ -n "${_seen_name[$_n]:-}" ]; then
    echo "SOURCES 中 NAME 重复，会互相淘汰快照: $_n" >&2
    exit 1
  fi
  _seen_name["$_n"]=1
done
unset _seen_name _entry _n

# mapfile 是 builtin，不必探测。head 已不再依赖。
# date/sleep/wc/tail/dirname/basename 也在用：缺 tail 会让空间检查被静默跳过。
for cmd in rsync flock find sort mountpoint nice stat df chmod touch mv rm mkdir sync awk \
           date sleep wc tail dirname basename; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "缺少命令: $cmd" >&2
    exit 1
  fi
done

# -printf 是 GNU find 扩展，BusyBox find 能找到但不会按我们预期过滤
if ! find /dev/null -maxdepth 0 -printf '%f' >/dev/null 2>&1; then
  echo "需要 GNU find（支持 -printf）" >&2
  exit 1
fi

# df --output 是 GNU coreutils 扩展，BusyBox df 不支持。
# 注意不能再加 -P：GNU df 里 --output 与 -i / -P / -T 互斥，加了会直接报错退出。
if ! df --output=size,avail / >/dev/null 2>&1; then
  echo "需要 GNU df（支持 --output）" >&2
  exit 1
fi

# =============================================================================
# 主流程：建目录 → 抢锁 → 轮转日志 → 挂载检查 → 逐源备份
# 顺序很重要：轮转必须在 flock 之后，否则并发 cron 会把正在写的日志 mv 走。
# =============================================================================

LOG_DIR="$(dirname "$LOG")"
mkdir -p "$LOG_DIR" "$BACKUP_BASE" "$(dirname "$LOCK")"

exec 9>>"$LOCK"
if ! flock -n 9; then
  log "已有备份在运行，本次退出"
  exit 1
fi

rotate_logs

# 挂载检查在抢到锁之后做：未挂载属于本次运行失败，应写进本轮日志
for m in "${MOUNT_CHECKS[@]}"; do
  [ -z "$m" ] && continue
  if ! mountpoint -q "$m"; then
    log "挂载点未挂载: $m，退出"
    exit 1
  fi
done

check_free_space "$BACKUP_BASE"

# 降低对前台任务和机械盘的影响。ionice 只在 CFQ/BFQ 上有效，mq-deadline 下是空操作。
if command -v ionice >/dev/null 2>&1; then
  RUN=(nice -n 19 ionice -c2 -n7)
else
  RUN=(nice -n 19)
fi

trap cleanup EXIT
# INT/TERM 默认会走 EXIT；显式再挂一次，保证 rc 有意义（130/143）
trap 'exit 130' INT
trap 'exit 143' TERM

log "========== 备份开始 host=$(hostname -s 2>/dev/null || hostname) pid=$$ =========="

for entry in "${SOURCES[@]}"; do
  backup_one "$entry"
done

log "全部任务结束，退出码 $overall_rc"
exit "$overall_rc"
