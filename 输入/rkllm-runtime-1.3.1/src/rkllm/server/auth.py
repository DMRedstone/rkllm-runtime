"""Bearer 鉴权（设计 N4：`server.api_key` / `server.api_key_file`）。

设计取舍（2026-11-20 审核 P1-S1 后修订）：
- 默认拒绝：除 EXEMPT_PATHS 外，一切路径都要校验；新增路由不需要记得加依赖。
- 只豁免 `/ready`：它仅返回布尔与状态枚举，供编排/探活使用；`/health` 会暴露
  模型名、队列深度、库指纹，必须与数据面同级保护（旧栈 1.3.1-15 即如此，
  `mcp/backend.py` 的 status 也带上了 Authorization）。
- **fail-closed**：`server.api_key_file` 一旦配置就必须可用（普通文件、可读、
  非空），否则拒绝启动；不再静默回退到 `api_key`（更不会回退到空 key）。
  显式清空鉴权只允许「删掉 api_key_file 配置」或写 `server.api_key = ""`。
- **非回环 + 无 key 拒绝启动**：无鉴权只允许绑定回环地址。该检查在
  `config._cross_validate` 的 serve 路径（require_model_path=True）执行，
  与 `mcp.transport=http` 必须配 allow_remote_origin 同属安全边界校验。
- 启动横幅与响应头必须显示**解析后**的鉴权状态（enabled/disabled + 来源），
  避免"显示已配置、实际未鉴权"。
- 常量时间比较，避免按字节计时猜测；key 永不写日志。
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import stat

__all__ = ["EXEMPT_PATHS", "AUTH_HEADER", "AUTH_SOURCE_HEADER",
           "is_loopback_host", "resolve_api_key", "auth_status",
           "security_errors", "make_auth_middleware"]

log = logging.getLogger("rkllm.server.auth")

EXEMPT_PATHS = frozenset({"/ready"})

# 响应头：让任何客户端（包括未鉴权时的 /health）都能读到**解析后**的状态。
AUTH_HEADER = "X-RKLLM-Auth"
AUTH_SOURCE_HEADER = "X-RKLLM-Auth-Source"

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})
_BEARER_PREFIX = "bearer "


def _api_key_file(cfg) -> str:
    return (getattr(cfg.server, "api_key_file", "") or "").strip()


def _api_key(cfg) -> str:
    return (getattr(cfg.server, "api_key", "") or "").strip()


def _bind_host(cfg) -> str:
    return str(getattr(cfg.server, "bind_host", "") or "").strip()


def is_loopback_host(host: str) -> bool:
    """该监听地址是否只对本机可见（无鉴权时的唯一安全绑定）。"""
    value = (host or "").strip()
    if not value:
        # 空 host 在 uvicorn 里等于全部接口；必须按非回环处理。
        return False
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _read_key_file(path: str) -> tuple[str, str]:
    """读取 key 文件，返回 (key, 错误说明)。错误说明为空表示读取成功。"""
    try:
        st = os.stat(path)
    except OSError as exc:
        return "", "不存在或不可访问（%s）" % exc
    if not stat.S_ISREG(st.st_mode):
        return "", "不是普通文件"
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        # 只提示，不阻断：老部署可能有 0640 的组可读文件
        log.warning("%s 权限过宽（建议 0600），当前 %o", path,
                    stat.S_IMODE(st.st_mode))
    try:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    except OSError as exc:
        return "", "读取失败（%s）" % exc
    if not key:
        return "", "文件为空"
    return key, ""


def _key_file_error(path: str, reason: str) -> str:
    return ("server.api_key_file（fail-closed）：%s %s。已配置 key 文件但不可用，"
            "拒绝启动且不会静默回退到空 key；请修复路径/权限/内容，或删除该项并"
            "显式配置 server.api_key" % (path, reason))


def resolve_api_key(cfg, *, strict: bool = False) -> str:
    """api_key_file 优先于 api_key（与 config.py 的 desc 一致）。

    strict=True（serve 启动/应用构造）：key 文件不可用时抛 ConfigError；
    strict=False（只读状态展示）：记日志并按 api_key 兜底。
    """
    path = _api_key_file(cfg)
    explicit = _api_key(cfg)
    if path:
        key, reason = _read_key_file(path)
        if reason:
            if strict:
                from ..config import ConfigError
                raise ConfigError([_key_file_error(path, reason)])
            log.warning("api_key_file %s %s；回退到 server.api_key", path, reason)
            return explicit
        return key
    return explicit


def auth_status(cfg) -> dict:
    """解析后的鉴权状态（启动横幅/响应头/健康检查共用，绝不返回 key 本身）。"""
    path = _api_key_file(cfg)
    key = resolve_api_key(cfg)
    if key:
        source = "api_key_file" if path else "api_key"
    else:
        source = "none"
    bind = _bind_host(cfg)
    return {"enabled": bool(key), "source": source, "bind_host": bind,
            "loopback_only": is_loopback_host(bind)}


def security_errors(cfg) -> list[str]:
    """P1-S1 fail-closed 校验；返回可读错误列表（空=通过）。

    serve 路径（config.load_config(require_model_path=True)）把它并入
    ConfigError；非 serve 场景（config show 等）不调用，便于排障时看生效值。
    """
    errors: list[str] = []
    path = _api_key_file(cfg)
    key = ""
    if path:
        key, reason = _read_key_file(path)
        if reason:
            # 已配置 key 文件但不可用：即使 api_key 有值也拒绝（不猜运维意图）
            errors.append(_key_file_error(path, reason))
            key = ""
    else:
        key = _api_key(cfg)
    bind = _bind_host(cfg)
    if not key and not is_loopback_host(bind):
        errors.append(
            "server.bind_host=%r（fail-closed）：非回环地址且未配置有效 API key。"
            "无鉴权只允许绑定 127.0.0.1/::1；对局域网提供服务必须同时配置 "
            "server.api_key 或 server.api_key_file" % bind)
    return errors


def make_auth_middleware(cfg):
    """返回 ASGI 中间件：无 key 时放行但如实标注状态；有 key 时默认拒绝。

    即使鉴权未启用也返回中间件——这样 `/health` 等任何响应都带
    `X-RKLLM-Auth: enabled|disabled` 与来源头，客户端/运维能直接看出真实状态，
    不再出现"配置了 key 文件所以显示已鉴权，实际空 key 放行"。
    构造时 strict 解析：key 文件在 load_config 之后被删/改坏也会拒绝启动。
    """
    expected = resolve_api_key(cfg, strict=True)
    status = auth_status(cfg)
    # strict 已保证 key 文件可用；这里用解析结果对齐 enabled/source
    status["enabled"] = bool(expected)
    status["source"] = ("api_key_file" if expected and _api_key_file(cfg)
                        else ("api_key" if expected else "none"))
    expected_bytes = expected.encode("utf-8")
    enabled_header = "enabled" if expected else "disabled"

    async def middleware(request, call_next):
        # 惰性挂到 app.state，测试/其它中间件无需改 create_app 即可读真实状态
        try:
            if getattr(request.app.state, "auth", None) is None:
                request.app.state.auth = dict(status)
        except AttributeError:
            pass
        if not expected or request.url.path in EXEMPT_PATHS:
            response = await call_next(request)
        else:
            from fastapi.responses import JSONResponse

            token = ""
            header = request.headers.get("authorization", "")
            if header[:len(_BEARER_PREFIX)].lower() == _BEARER_PREFIX:
                token = header[len(_BEARER_PREFIX):].strip()
            if token and hmac.compare_digest(token.encode("utf-8"),
                                             expected_bytes):
                response = await call_next(request)
            else:
                response = JSONResponse(
                    {"error": {"type": "authentication_error",
                               "message": "missing or invalid API key"}},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"})
        response.headers[AUTH_HEADER] = enabled_header
        response.headers[AUTH_SOURCE_HEADER] = status["source"]
        return response

    # create_app 用同一份解析结果写 app.state.auth（避免二次读 key 文件/重复告警）
    middleware.auth_state = dict(status)
    return middleware
