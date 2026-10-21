# -*- coding: utf-8 -*-
"""soak_test.py —— 服务端常驻压测（部署后的验收测试）。

与 run_all.py 的区别：run_all 是「跑一遍看指标」，本脚本是「长时间挂着看恶化」——
只关心三件事：服务是否一直活着、错误计数是否增长、RSS 是否单调上升
（审查建议：RSS 上涨先查 _keep/_run_keep；rejected 涨而 requests 不涨说明
max_queue/queue_timeout 需要调）。

用法（在仓库任意位置）：

    python soak_test.py --host 192.0.2.10:8080 --key <KEY> --hours 2 \\
        --ssh "ssh -i <pem> user@192.0.2.10"

负载形态（单 worker 顺序轮换，模拟真实客户端节奏）：
  流式短问 / 非流式短问 / 非流式长 prompt(~2000 字) / 长输出(max_tokens=1024) /
  stop 序列 / 错误路径（401、413、404 轮流——错误路径也要跑，否则关连接的
  分支永远没有被真实踩过）。

监控采样（默认 5 分钟一次）：
  SSH 到板端取 free -m 的 available 与 rkllm 进程 RSS，叠加 /health 的 stats。

退出码：0 = 通过（全程存活、errors 不增、RSS 无泄漏迹象）；1 = 有任一恶化；
2 = 参数/环境错误。只读，不改被测服务。

依赖：仅标准库；SSH 采样需要本机 ssh 与私钥（无 SSH 时跳过资源采样，只监控 /health）。
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import subprocess
import sys
import threading
import time

CHAT = "/v1/chat/completions"

LONG_PROMPT = ("请把下面的会议记录整理成要点清单，保留原意，不超过十条：\n" +
               "会议讨论了全县城镇建筑垃圾处理设施的选址原则，" * 60 +
               "最后强调要于本月底前完成实地踏勘并形成书面报告。")


def cjk_tokens(text: str) -> int:
    """与服务端同源的估算：(字节数+2)//3。"""
    return (len(text.encode("utf-8")) + 2) // 3


class Soak:
    def __init__(self, host, key, ssh_cmd):
        self.host, self.key, self.ssh_cmd = host, key, ssh_cmd
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.counts = {"ok": 0, "http_err": 0, "conn_err": 0, "err_path_ok": 0}
        self.samples = []            # 监控时间线
        self.errors = []             # 非预期错误明细（最多留 50 条）
        self.status_history = []     # (ts, alive, note)

    # ---------------- HTTP ----------------
    def _conn(self, timeout=120):
        return http.client.HTTPConnection(self.host, timeout=timeout)

    def _request(self, method, path, payload, hdrs, timeout=120):
        c = self._conn(timeout)
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            headers = dict(hdrs)
            if body is not None:
                headers["Content-Type"] = "application/json"
            c.request(method, path, body=body, headers=headers)
            r = c.getresponse()
            data = r.read()
            return r.status, data, dict(r.getheaders())
        finally:
            c.close()

    def _chat(self, payload, stream=False):
        """流式时返回 (status, 组装文本, finish)；非流式返回 (status, dict, None)。"""
        hdrs = {"Authorization": "Bearer " + self.key}
        if stream:
            payload = {**payload, "stream": True}
            c = self._conn(180)
            try:
                c.request("POST", CHAT, body=json.dumps(payload).encode(),
                          headers={**hdrs, "Content-Type": "application/json"})
                r = c.getresponse()
                buf, finish = [], None
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    obj = json.loads(chunk)
                    ch = obj.get("choices") or [{}]
                    buf.append(ch[0].get("delta", {}).get("content", ""))
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
                return r.status, "".join(buf), finish
            finally:
                c.close()
        status, data, _ = self._request("POST", CHAT, payload, hdrs)
        return status, (json.loads(data) if status == 200 else data[:200]), None

    # ---------------- 负载 ----------------
    def worker(self, rounds_max=0):
        i = 0
        while not self.stop.is_set() and (rounds_max == 0 or i < rounds_max):
            i += 1
            kind = i % 6
            try:
                if kind == 0:
                    st, txt, fin = self._chat({"messages": [{"role": "user", "content": "用一句话说明什么是垃圾分类。"}],
                                               "stream": True, "max_tokens": 256}, stream=True)
                    self._record(st == 200 and txt and fin, "stream short", st)
                elif kind == 1:
                    st, d, _ = self._chat({"messages": [{"role": "user", "content": "把「尽快完成」改写得更正式。"}],
                                           "max_tokens": 128})
                    self._record(st == 200 and d["choices"][0]["finish_reason"] in ("stop", "length"),
                                 "short chat", st)
                elif kind == 2:
                    st, d, _ = self._chat({"messages": [{"role": "user", "content": LONG_PROMPT}],
                                           "max_tokens": 512})
                    self._record(st == 200, "long prompt (~%d tok)" % cjk_tokens(LONG_PROMPT), st)
                elif kind == 3:
                    st, d, _ = self._chat({"messages": [{"role": "user", "content": "写一篇 800 字的关于节约用水的宣传短文。"}],
                                           "max_tokens": 1024})
                    fr = d["choices"][0]["finish_reason"] if st == 200 else "-"
                    self._record(st == 200, "long output (finish=%s)" % fr, st)
                elif kind == 4:
                    st, d, _ = self._chat({"messages": [{"role": "user", "content": "你好"}],
                                           "stop": ["。"], "max_tokens": 128})
                    self._record(st == 200, "stop-seq", st)
                else:
                    # 错误路径轮询：关连接的分支必须被真实踩过
                    ep = i % 18
                    if ep == 5:
                        st, _, _ = self._request("GET", "/health", None, {})   # 401
                        ok = st == 401
                    elif ep == 11:
                        s = socket.create_connection(tuple(self.host.split(":")), timeout=15)
                        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
                                  + self.key.encode() + b"\r\nContent-Length: 10485760\r\n\r\n")
                        s.settimeout(15)
                        head = s.recv(512).decode("latin-1", "replace")
                        s.close()
                        ok = " 413 " in head.split("\r\n")[0] and "close" in head.lower()
                    else:
                        st, _, _ = self._request("POST", "/v1/nope", {}, {"Authorization": "Bearer " + self.key})
                        ok = st == 404
                    with self.lock:
                        self.counts["err_path_ok"] += 1 if ok else 0
                        if not ok:
                            self.errors.append("error-path unexpected: kind=%s" % kind)
                time.sleep(1.5)
            except (http.client.HTTPException, socket.timeout, OSError) as exc:
                with self.lock:
                    self.counts["conn_err"] += 1
                    if len(self.errors) < 50:
                        self.errors.append("conn: %r" % exc)
                time.sleep(5)
            except Exception as exc:                       # noqa: BLE001
                with self.lock:
                    self.counts["conn_err"] += 1
                    if len(self.errors) < 50:
                        self.errors.append("unexpected: %r" % exc)
                time.sleep(5)

    def _record(self, ok, label, status):
        with self.lock:
            if ok:
                self.counts["ok"] += 1
            else:
                self.counts["http_err"] += 1
                if len(self.errors) < 50:
                    self.errors.append("%s -> HTTP %s" % (label, status))

    # ---------------- 监控 ----------------
    def sample_once(self):
        ts = time.time()
        row = {"ts": ts}
        try:
            c = self._conn(15)
            c.request("GET", "/health", headers={"Authorization": "Bearer " + self.key})
            r = c.getresponse()
            h = json.loads(r.read())
            c.close()
            row["alive"] = True
            row["stats"] = h.get("stats", {})
            row["rss_mb"] = h.get("rss_mb")  # 服务端没有这个字段，RSS 从 SSH 取
        except Exception as exc:                       # noqa: BLE001
            row["alive"] = False
            row["error"] = repr(exc)
        if self.ssh_cmd:
            try:
                out = subprocess.run(self.ssh_cmd + ["free -m | sed -n 2p; ps -o rss= -C rkllm | head -1"],
                                     capture_output=True, text=True, timeout=30).stdout
                m = re.search(r"Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\d+)", out)
                if m:
                    row["mem_total_mb"], row["mem_used_mb"] = int(m.group(1)), int(m.group(2))
                    row["available_mb"] = int(m.group(4))
                rss = [int(x) for x in out.strip().splitlines()[-1].split()] if out.strip() else []
                if rss:
                    row["rss_mb"] = rss[0] // 1024
            except Exception as exc:                   # noqa: BLE001
                row["ssh_error"] = repr(exc)
        with self.lock:
            self.samples.append(row)
            self.status_history.append((ts, row.get("alive", False), ""))
        return row

    def monitor(self, interval_s):
        while not self.stop.is_set():
            self.sample_once()
            self.stop.wait(interval_s)

    # ---------------- 判定 ----------------
    def verdict(self):
        issues = []
        alive_bad = [1 for _, a, _ in self.status_history if not a]
        if alive_bad:
            issues.append("监控期间出现 %d 次服务不健康" % len(alive_bad))
        if self.counts["conn_err"]:
            issues.append("连接错误 %d 次" % self.counts["conn_err"])
        if self.counts["http_err"]:
            issues.append("非预期 HTTP 响应 %d 次" % self.counts["http_err"])
        rss = [s["rss_mb"] for s in self.samples if s.get("rss_mb")]
        leak = None
        if len(rss) >= 6:
            head = sum(rss[:2]) / 2.0
            tail = sum(rss[-2:]) / 2.0
            leak = tail - head
            if leak > 150:                              # MB；噪声容忍 150MB
                issues.append("RSS 增长 %.0f MB（首段均值 %.0f -> 尾段均值 %.0f），疑似泄漏" % (leak, head, tail))
        avail = [s["available_mb"] for s in self.samples if s.get("available_mb")]
        if len(avail) >= 6 and avail[0] - avail[-1] > 300:
            issues.append("可用内存下降 %d MB（%d -> %d）"
                          % (avail[0] - avail[-1], avail[0], avail[-1]))
        return issues, {"rss_first": rss[0] if rss else None,
                        "rss_last": rss[-1] if rss else None,
                        "rss_delta_mb": leak,
                        "available_first": avail[0] if avail else None,
                        "available_last": avail[-1] if avail else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True, help="host:port")
    ap.add_argument("--key", required=True)
    ap.add_argument("--hours", type=float, default=2.0, help="压测时长（小时），默认 2")
    ap.add_argument("--sample-interval", type=float, default=300, help="监控采样间隔秒，默认 300")
    ap.add_argument("--ssh", nargs="*", default=None,
                    help="SSH 采样命令前缀（列表形式），如 --ssh ssh -i pem user@192.0.2.10；"
                         "不提供则跳过资源采样")
    ap.add_argument("--out", default=None, help="结果 JSON 路径（默认 results/soak_<ts>.json）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：只跑 60 秒，用于验证脚本本身")
    args = ap.parse_args()

    hours = 1.0 / 60 if args.smoke else args.hours
    interval = 20 if args.smoke else args.sample_interval
    soak = Soak(args.host, args.key, args.ssh)

    print("soak: %s for %.2fh, sample every %ds" % (args.host, hours, interval), flush=True)
    base = soak.sample_once()
    print("baseline: alive=%s requests=%s rejected=%s rss=%s"
          % (base.get("alive"), base.get("stats", {}).get("requests"),
             base.get("stats", {}).get("rejected"), base.get("rss_mb")), flush=True)
    if not base.get("alive"):
        print("服务一开始就不可达，退出")
        return 2

    mon = threading.Thread(target=soak.monitor, args=(interval,), daemon=True)
    mon.start()
    workers = [threading.Thread(target=soak.worker, daemon=True) for _ in range(1)]
    w = threading.Thread(target=soak.worker, kwargs={"rounds_max": 0}, daemon=True)
    workers.append(w)
    for t in set(workers):
        pass
    wrk = threading.Thread(target=soak.worker, daemon=True)
    t_end = time.time() + hours * 3600
    wrk.start()
    try:
        while time.time() < t_end:
            time.sleep(60 if not args.smoke else 10)
            with soak.lock:
                c = dict(soak.counts)
            last = soak.samples[-1] if soak.samples else {}
            print("[%5.0fs] ok=%d http_err=%d conn_err=%d err_path_ok=%d | alive=%s "
                  "req=%s rej=%s err=%s rss=%s"
                  % (hours * 3600 - (t_end - time.time()), c["ok"], c["http_err"],
                     c["conn_err"], c["err_path_ok"], last.get("alive"),
                     last.get("stats", {}).get("requests"), last.get("stats", {}).get("rejected"),
                     last.get("stats", {}).get("errors"), last.get("rss_mb")), flush=True)
    except KeyboardInterrupt:
        print("被中断")
    finally:
        soak.stop.set()
        wrk.join(timeout=10)
        final = soak.sample_once()
        issues, memo = soak.verdict()
        result = {
            "target": "http://" + args.host,
            "hours": hours,
            "counts": soak.counts,
            "errors": soak.errors[:20],
            "samples": soak.samples,
            "verdict_issues": issues,
            "memory": memo,
            "final_stats": final.get("stats"),
        }
        out = args.out or ("results/soak_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print("=" * 60, flush=True)
        print("soak 完成: ok=%d http_err=%d conn_err=%d err_path_ok=%d"
              % (soak.counts["ok"], soak.counts["http_err"],
                 soak.counts["conn_err"], soak.counts["err_path_ok"]), flush=True)
        print("内存: %s" % memo, flush=True)
        if issues:
            print("结论: 有问题 ——")
            for i in issues:
                print("  - %s" % i, flush=True)
            code = 1
        else:
            print("结论: 通过（全程存活、错误不增、RSS 无泄漏迹象）", flush=True)
            code = 0
        print("结果写入 %s" % out, flush=True)
        return code


if __name__ == "__main__":
    sys.exit(main())
