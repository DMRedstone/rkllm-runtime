"""Bearer 鉴权（设计 N4：`server.api_key` / `server.api_key_file`）。

设计取舍：
- 默认拒绝：除 EXEMPT_PATHS 外，一切路径都要校验；新增路由不需要记得加依赖。
- 只豁免 `/ready`：它仅返回布尔与状态枚举，供编排/探活使用；`/health` 会暴露
  模型名、队列深度、库指纹，必须与数据面同级保护（旧栈 1.3.1-15 即如此，
  `mcp/backend.py` 的 status 也带上了 Authorization）。
- `api_key` 为空（且文件为空/不可读）时整条链不启用——与 15 版「空=不鉴权」
  的行为一致，保证升级不改变既有部署的可用性。
- 常量时间比较，避免按字节计时猜测；key 永不写日志。
"""
from __future__ import annotations

import hmac
import logging
import os
import stat

__all__ = ["EXEMPT_PATHS", "resolve_api_key", "make_auth_middleware"]

log = logging.getLogger("rkllm.server.auth")

EXEMPT_PATHS = frozenset({"/ready"})


def resolve_api_key(cfg) -> str:
    """api_key_file 优先于 api_key（与 config.py 的 desc 一致）。"""
    path = (getattr(cfg.server, "api_key_file", "") or "").strip()
    if path:
        try:
            st = os.stat(path)
        except OSError as exc:
            # A-3：走日志（不记 key 本身，只记路径与异常）
            log.warning("api_key_file 不可读（%s）：%s；回退到 api_key",
                        path, exc)
        else:
            if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                log.warning("%s 权限过宽（建议 0600），当前 %o", path,
                            stat.S_IMODE(st.st_mode))
            try:
                with open(path, encoding="utf-8") as fh:
                    key = fh.read().strip()
                if key:
                    return key
            except OSError as exc:
                log.warning("api_key_file 读取失败：%s", exc)
    return (getattr(cfg.server, "api_key", "") or "").strip()


def make_auth_middleware(cfg):
    """返回 ASGI 中间件函数；key 为空时返回 None（不挂载）。"""
    expected = resolve_api_key(cfg)
    if not expected:
        return None
    expected_bytes = expected.encode("utf-8")

    async def middleware(request, call_next):
        from fastapi.responses import JSONResponse

        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        token = ""
        if header[:7].lower() == "bearer ":
            token = header[7:].strip()
        if not token or not hmac.compare_digest(token.encode("utf-8"),
                                                expected_bytes):
            return JSONResponse(
                {"error": {"type": "authentication_error",
                           "message": "missing or invalid API key"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    return middleware
