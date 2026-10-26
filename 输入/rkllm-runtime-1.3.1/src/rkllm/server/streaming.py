"""流式基础设施（F-1 / O-2）：token 泵、心跳、断连取消，一处定义。

改造前 `openai.py` 与 `responses.py` 各有一份约 80 行的骨架，三处取消逻辑
互相漂移：① `queue.run(cancelled=)` 的两个调用方都不传（死参数）；
② `watch_disconnect` 只 abort 一次、不回传主循环；③ pump 的 `Full` 处理
一条 `break`、另一条完全无保护（P0-3 已修但仍是两份代码）。

本模块收敛为单一实现，并落地 vLLM 式**两段式取消契约**：

- 第一段 `CancelToken.cancel()`：框架侧（客户端断开、生成器被外部关闭）
  只置位，幂等，任意线程可调；
- 第二段 `CancelToken.abort_once()`：归还 NPU 推理槽（`supervisor.abort()`），
  幂等、只执行一次，避免重复打断后续请求；
- 取消后产生的异常不再作为帧发给客户端（客户端已经走了），泵只投
  `STREAM_DONE`，主循环据 `DISCONNECT` 直接结束。

与 llama.cpp `SERVER_TASK_TYPE_CANCEL` 的差别：worker 侧没有可插队的取消
任务，取消靠 `abort()` 打断 worker 的 socket 读——这是本 runtime 的唯一
可行路径，故 `cancel()` 会立即触发第二段（`TokenStream.cancel()`）。

效率：主循环用 `asyncio.wait(q.get(), heartbeat)` 等待，替代原先
`get_nowait() + sleep(0.02)` 的 20ms 忙轮询——无 token 时只在心跳到点醒来。
"""

from __future__ import annotations

import asyncio
import logging
import threading

log = logging.getLogger("rkllm.server.streaming")

# ---- 事件种类（iterate_stream 的产出）------------------------------------
ITEM = "item"
HEARTBEAT = "heartbeat"
DISCONNECT = "disconnect"
DONE = "done"

# ---- 哨兵（原先 openai/responses 各定义一份 `_STREAM_DONE`）---------------
STREAM_DONE = object()
TRUNCATED = {"truncated": True}   # 不可变标记：泵因消费停滞而截断
DISCONNECT_ITEM = object()        # 断连监视投进同一队列：见 iterate_stream

# ---- 魔法数集中（原理由：两文件各写一份 512/5/0.02）----------------------
# 消费停滞的判定：积压到上限即截断（PUT_TIMEOUT_S / BACKPRESSURE_POLL_S 的
# 轮询等待已去掉，见 TokenStream._offer）。
ITEM_QUEUE_MAXSIZE = 512
WATCH_IDLE_SLEEP_S = 0.05  # 断连监视：receive() 空转时的退避，见 watch()


class CancelToken:
    """取消的单一事实源（两段式契约，见模块 docstring）。"""

    __slots__ = ("_supervisor", "_event", "_lock", "_aborted")

    def __init__(self, supervisor=None):
        self._supervisor = supervisor
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._aborted = False

    def cancel(self) -> None:
        """第一段：置位。可在任意线程/事件循环调用，幂等。"""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def abort_once(self) -> bool:
        """第二段：归还 NPU。返回本次是否真的执行了 abort。"""
        with self._lock:
            if self._aborted:
                return False
            self._aborted = True
        supervisor = self._supervisor
        if supervisor is not None:
            try:
                supervisor.abort()
            except Exception as exc:   # 断连路径不能因 abort 失败而泄漏推理槽
                log.warning("abort 失败（已忽略）：%s", exc)
        return True


class TokenStream:
    """把 `queue.run()` 的同步 token 迭代桥到事件循环。

    生命周期：`start(loop, queue)` → 泵线程产出 item / 异常 / 截断标记 →
    `STREAM_DONE` 收尾。`cancel()` 供断连或异常收尾调用。
    """

    def __init__(self, svc, pipe, *, name: str = "sse-pump"):
        self.svc = svc
        self.pipe = pipe
        self.name = name
        self.token = CancelToken(getattr(svc, "supervisor", None))
        self.settled = False           # 调用方已完成收尾（错误帧路径）
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._thread: threading.Thread | None = None

    # -- 生命周期 ---------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop,
              queue: asyncio.Queue) -> None:
        self._loop = loop
        self._queue = queue
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=self.name)
        self._thread.start()

    def cancel(self) -> None:
        """两段式取消：置位 + 立即归还 NPU（幂等）。"""
        self.token.cancel()
        self.token.abort_once()

    def mark_settled(self) -> None:
        """调用方已发出终帧（错误帧等）：收尾不再 abort。"""
        self.settled = True

    # -- 泵线程 -----------------------------------------------------------
    def _pump(self) -> None:
        truncated = False
        try:
            for item in self.svc.queue.run(
                    lambda: self.pipe.iter_tokens(),
                    cancelled=lambda: self.token.cancelled):
                if self.token.cancelled:
                    break
                if not self._offer(item):
                    truncated = True
                    break
        except BaseException as exc:
            # 取消导致的异常（QueueTimeout("cancelled while waiting") 等）
            # 不作为错误帧投递：客户端已经断开，没有人会读它。
            if not self.token.cancelled:
                self._put(exc)
        finally:
            if truncated:
                self._put(TRUNCATED)
            self._put(STREAM_DONE)
            # 兜底第二段：泵退出时若已取消，确保归还（abort_once 幂等）
            if self.token.cancelled:
                self.token.abort_once()

    def _offer(self, item) -> bool:
        """投递一条 item；消费停滞（队列积压到上限）即截断，不等消费者。

        此前这里用 `sleep(0.01)` 轮询等消费者最多 5s：泵线程被卡在投递上，
        NPU 那头却还在出 token——等的是永远不会及时消费的对端。截断语义本
        就在契约里（`TRUNCATED` → `finish_reason=length`），与其等满 5s 不
        如立刻截断并归还 NPU（审计 P-背压去轮询）。
        """
        if self.token.cancelled:
            return False
        if self._queue is not None and \
                self._queue.qsize() >= ITEM_QUEUE_MAXSIZE:
            return False
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
            return True
        except RuntimeError:              # 事件循环已关闭（进程收尾）
            return False

    def _put(self, item) -> None:
        """尾部投递（异常/哨兵）：事件循环可能已关闭，失败即放弃。"""
        if self._queue is None or self._loop is None:
            return
        # asyncio.Queue 无界，`put_nowait` 不会抛 Full：这里的重试只可能来自
        # 事件循环已关闭（RuntimeError，直接放弃），循环退避已随背压轮询一并去掉。
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            return


async def iterate_stream(request, stream: TokenStream, heartbeat_s: int):
    """消费 `stream`，产出 `(kind, payload)` 事件。

    - `("item", obj)`：一条 token 或管线元数据（dict）；
    - `("heartbeat", None)`：空闲满 `heartbeat_s`，调用方发心跳帧；
    - `("disconnect", None)`：客户端断开（已取消 + 归还 NPU），调用方应立即结束；
    - `("done", None)`：泵结束（正常完成或截断）。

    断连为什么走队列而不是和 getter 一起 `asyncio.wait`：实测（Python
    3.13 + uvicorn 0.54）`asyncio.wait` 的集合里只要还有一个长期挂起的
    task，另一个 task 的完成就**不会提前唤醒等待**——每条 token 都要等满
    一个 `heartbeat_s`（15s）才被取走，流式从毫秒级退化成 15s/token。故
    这里主循环只等 `queue.get()` 一个源，断连由 `watch()` 投 `DISCONNECT_ITEM`
    到同一队列唤醒：断连仍毫秒级生效，token 也不再被心跳节拍限速。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stream.start(loop, queue)
    completed = False

    async def watch() -> None:
        """直接长轮询 receive()：uvicorn 在 connection_lost 时推 http.disconnect。

        不用 request.is_disconnected()：它内部用 CancelScope 立即取消等待，
        只读取"已经入队"的消息，流式期间极易漏掉（实测断连后永不触发）。

        ASGI 并未承诺「body 读完后 receive() 会挂起」：某些 server/版本会
        立刻返回非 disconnect 的消息，裸 while 循环随即变成空转、饿死事件
        循环。故每轮未收到 disconnect 都退避一拍：真断连不受影响，空转时
        让出循环。
        """
        try:
            while True:
                message = await request.receive()
                if message.get("type") == "http.disconnect":
                    log.info("client disconnected -> abort worker")
                    stream.cancel()        # 第一段 + 第二段（归还 NPU）
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait,
                                                  DISCONNECT_ITEM)
                    except RuntimeError:     # 事件循环已关闭
                        pass
                    return
                await asyncio.sleep(WATCH_IDLE_SLEEP_S)
        except Exception:
            return

    watcher = asyncio.create_task(watch())
    getter: asyncio.Task | None = None
    try:
        while True:
            if getter is None:
                getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({getter}, timeout=heartbeat_s)
            if getter not in done:
                yield HEARTBEAT, None
                continue
            item = getter.result()
            getter = None
            if item is DISCONNECT_ITEM:
                completed = True
                yield DISCONNECT, None
                return
            if item is STREAM_DONE:
                completed = True
                yield DONE, None
                return
            yield ITEM, item
    finally:
        watcher.cancel()
        if getter is not None and not getter.done():
            getter.cancel()
        # 未正常收尾（生成器被外部关闭/异常）：确保泵退出并归还 NPU
        if not completed and not stream.settled:
            stream.cancel()
