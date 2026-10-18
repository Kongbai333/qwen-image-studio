#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Image 2.1 GGUF Studio —— 官方 API 客户端 + Hugging Face 账号池
====================================================================

目标 Space : assembledchaos/qwen-image-2-1-studio
官方源码   : https://huggingface.co/spaces/assembledchaos/qwen-image-2-1-studio/raw/main/app.py
官方 Host  : https://assembledchaos-qwen-image-2-1-studio.hf.space

本模块的常量、参数顺序、取值范围、校验文案、输出结构**逐项对齐官方 app.py 的
generate()**，只是把本地执行换成远程调用，并在外面加了一层 HF 账号池，用来
分摊 ZeroGPU 的每日配额（配额按账号计，一个号跑满了就轮到下一个号）。

官方签名（位置顺序不可变）::

    generate(prompt, mode, reference, aspect_ratio, steps, seed, randomize_seed)
        -> (result_image, download, seed, details)

``reference`` 既可以是单张路径，也可以是路径列表 —— Qwen-Image-2.1 模型原生支持
最多 10 张参考图（模型卡：Support up to 10 reference images）。但官方 Space 的
``gr.Image`` 只有一格，所以多图按 Space 能力走两条路：

* Space 支持 Gallery 输入（自建 fork，见 ``hf_space_multi/``）→ 整批 list 传进去，
  每张图各自成一个 ``<imageN>`` 块，顺序即 prompt 里的「图1/图2」；
* 单图 Space（官方那个）→ 本地并排拼成一张再传，prompt 照样按「图1/图2」指代。

命令行用法::

    python qwen_client.py --pool
    python qwen_client.py --prompt "a small red apple on a white table" --steps 8
    python qwen_client.py --prompt "把背景换成黄昏" --mode "Edit an image" --ref in.png
    python qwen_client.py --prompt "把第一张图的花放进第二张的场景" -m "Edit an image" --refs a.png b.png
    python qwen_client.py --prompt "一只卡通龙贴纸" --mode "Transparent PNG" --aspect "Square · 1:1"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

from gradio_client import Client, handle_file
from PIL import Image

import fingerprint

log = logging.getLogger("qwen.client")

# ==========================================================================
# 1. 常量区 —— 与官方 app.py 保持一致，不要随意改动
# ==========================================================================

SPACE_ID = os.getenv("QWEN_SPACE_ID", "assembledchaos/qwen-image-2-1-studio")
API_NAME = "/generate"

#: Space 的直连地址（owner-space.hf.space）。
#: 把**完整 URL** 作为 Client 的 src 时，gradio_client 会跳过 huggingface_hub 的
#: ``space_info`` / ``get_space_runtime`` 两条请求（``_get_space_state`` 在
#: ``space_id`` 为 None 时直接返回 None）。少一次到 huggingface.co 的往返，
#: 更重要的是：只放行 hf.space、不放行 huggingface.co 的代理也能跑通。
SPACE_URL = os.getenv("QWEN_SPACE_URL") or ""


def space_url(space_id: str) -> str:
    """``owner/name`` → ``https://owner-name.hf.space``。

    HF 的规则：斜杠、下划线、**点** 全部转横线，且全小写。
    漏了点会踩证书 hostname mismatch（``qwen-image-2.1`` → ``qwen-image-2-1``）。
    """
    slug = (
        space_id.strip()
        .replace("/", "-")
        .replace("_", "-")
        .replace(".", "-")
        .lower()
    )
    return f"https://{slug}.hf.space"


#: url（默认）= 直接连 hf.space；id = 走 huggingface.co 解析（老行为）
SPACE_SRC_MODE = os.getenv("QWEN_SPACE_SRC", "url").strip().lower()


def space_src(space_id: str | None = None) -> str:
    """传给 gradio_client.Client 的 src。"""
    sid = space_id or SPACE_ID
    if SPACE_SRC_MODE == "id":
        return sid
    if space_id is None and SPACE_URL:
        return SPACE_URL
    return space_url(sid)


#: 可轮换的 Space 列表（签名必须一致）。默认只启用主站；
#: 其他兼容 Space 由部署者通过 QWEN_SPACES 显式配置。
DEFAULT_SPACES = [SPACE_ID]
SPACE_IDS: list[str] = [
    s.strip() for s in os.getenv("QWEN_SPACES", "").split(",") if s.strip()
] or DEFAULT_SPACES


def space_urls() -> list[tuple[str, str]]:
    """[(space_id, src_url), ...]"""
    return [(sid, space_src(sid)) for sid in SPACE_IDS]

#: 官方 MODES = ["Create an image", "Edit an image", "Transparent PNG"]
MODES: list[str] = ["Create an image", "Edit an image", "Transparent PNG"]

#: 官方 SIZES —— 键名里的 "·" 是 U+00B7 中黑点，不是英文句点，改错会校验失败
SIZES: dict[str, tuple[int, int]] = {
    "Square · 1:1": (1024, 1024),
    "Landscape · 16:9": (1344, 768),
    "Portrait · 9:16": (768, 1344),
    "Landscape · 4:3": (1152, 864),
    "Portrait · 3:4": (864, 1152),
}

MAX_SEED = 2**31 - 1
STEPS_MIN, STEPS_MAX = 4, 40
PROMPT_MAX = 4000

#: 参考图张数上限 —— 对齐 Qwen-Image-2.1 模型卡 "Support up to 10 reference images"
MAX_REFS = 10

#: 哪些 Space 的 reference 入参是 gr.Gallery（list）。逗号分隔，子串匹配，
#: owner/name 和 https://xxx.hf.space 两种写法都认。
#: 例：set QWEN_MULTI_SPACES=me/qwen-image-2-1-multi
MULTI_SPACES: list[str] = [
    x.strip().lower() for x in os.getenv("QWEN_MULTI_SPACES", "").split(",") if x.strip()
]
#: 无视名单，全部按多图传（自建 fork 联调时省事）
MULTI_ALL = os.getenv("QWEN_MULTI_ALL", "").strip().lower() in ("1", "true", "yes", "on")

DEFAULT_MODE = MODES[0]
DEFAULT_ASPECT = "Square · 1:1"
DEFAULT_STEPS = 40
DEFAULT_SEED = 42

#: 调用路线策略 —— 决定「先匿名走代理」还是「先花 token 账号的配额」
STRATEGIES = ("proxy_first", "token_first", "proxy_only", "token_only")
DEFAULT_STRATEGY = os.getenv("QWEN_STRATEGY", "proxy_first")

#: 匿名路线配额耗尽后的代理冷却（秒）。ZeroGPU 匿名额度按 IP 计，换 IP 即可续。
PROXY_QUOTA_COOLDOWN = 1800.0
#: 代理本身故障（连不上 / 407 / SOCKS 错）后的冷却（秒），比配额冷却短
PROXY_ERROR_COOLDOWN = 300.0

#: 官方输出文件名模板
OUTPUT_NAME = "qwen-image-2.1-gguf-{seed}.png"

#: 官方生成函数的 docstring，原样保留，方便对照
OFFICIAL_DOC = """Generate or edit an image with abenzerps' Qwen-Image-2.1 Q4_K_M GGUF.

    Choose Create an image, Edit an image (requires a reference), or
    Transparent PNG. Disable randomize_seed to reuse a seed. The result
    includes a PNG download, the actual seed, and generation details.
    """


# ==========================================================================
# 2. 异常
# ==========================================================================


class QwenError(RuntimeError):
    """所有客户端异常的基类。"""


class ValidationError(QwenError):
    """本地校验不通过 —— 文案与官方 gr.Error 完全一致。"""


class QuotaExceeded(QwenError):
    """账号 ZeroGPU 配额耗尽，需要换号。"""


class ProxyError(QwenError):
    """代理不可用（连不上 / 认证失败 / SOCKS 错），需要换代理。"""


class AnonQuotaExceeded(QuotaExceeded):
    """匿名额（按 IP）耗尽或限流，需要换代理而不是换账号。"""


class SpaceUnavailable(QwenError):
    """这个 Space 本身用不了（排队满 / 没启动 / 域名错），换下一个 Space。"""


class Cancelled(QwenError):
    """任务被取消。"""


class PoolExhausted(QwenError):
    """账号池里所有账号都试过了，仍然失败。"""


# ==========================================================================
# 3. 参数校验 —— 逐条对齐官方 generate() 里的检查
# ==========================================================================


def norm_refs(reference: Any) -> list[str]:
    """单个路径 / 路径列表 / None → 干净的路径列表。空值和 None 直接丢掉。"""
    if reference is None:
        return []
    seq = list(reference) if isinstance(reference, (list, tuple, set)) else [reference]
    out: list[str] = []
    for r in seq:
        if r is None:
            continue
        path = str(r).strip()
        if path:
            out.append(path)
    return out


def _space_key(text: str) -> str:
    """把 owner/name 和 https://host.hf.space 归一成同一种可比较形式。"""
    key = (text or "").strip().lower()
    for prefix in ("https://", "http://"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.rstrip("/").replace("/", "-")


def space_supports_multi(space: str | None) -> bool:
    """这个 Space 的 reference 入参是不是 list（即 gr.Gallery 输入）。"""
    if MULTI_ALL:
        return True
    key = _space_key(space)
    if not key:
        return False
    for item in MULTI_SPACES:
        want = _space_key(item)
        if want and (want in key or key in want):
            return True
    return False


def stitch_refs(paths: list[str], out_dir: str | Path, gap: int = 16, max_side: int = 2048) -> str:
    """单图 Space 的多图兜底：把多张参考图并排拼成一张。

    官方 Space 收到图后会 ``thumbnail((2048, 2048))``，所以拼图总宽压在 2048 内，
    免得它二次缩放把细节糊掉。prompt 里按「图1/图2…」指代，从左到右数。
    """
    ims = []
    for path in paths:
        with Image.open(path) as im:
            ims.append(im.convert("RGBA").copy())
    if not ims:
        raise ValidationError("Upload a reference image before editing.")

    height = min(i.height for i in ims)
    scaled = [
        i if i.height == height
        else i.resize((max(1, round(i.width * height / i.height)), height), Image.LANCZOS)
        for i in ims
    ]
    total_w = sum(i.width for i in scaled) + gap * (len(scaled) - 1)
    if total_w > max_side:
        k = max_side / total_w
        scaled = [
            i.resize((max(1, round(i.width * k)), max(1, round(i.height * k))), Image.LANCZOS)
            for i in scaled
        ]
        height = scaled[0].height
        total_w = sum(i.width for i in scaled) + gap * (len(scaled) - 1)

    canvas = Image.new("RGBA", (total_w, height), (255, 255, 255, 255))
    x = 0
    for i in scaled:
        canvas.paste(i, (x, 0), i)
        x += i.width + gap

    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"refs-stitched-{len(scaled)}-{int(time.time() * 1000)}.png"
    canvas.save(dst)
    return str(dst)


def validate_params(
    prompt: str,
    mode: str = DEFAULT_MODE,
    reference: Any = None,
    aspect_ratio: str = DEFAULT_ASPECT,
    steps: float = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    randomize_seed: bool = True,
) -> dict[str, Any]:
    """复刻官方 app.py generate() 开头的全部校验，返回归一化后的参数。"""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValidationError("Describe the image you want to create or the change to make.")
    if len(prompt) > PROMPT_MAX:
        raise ValidationError("Keep your prompt under 4,000 characters.")
    if mode not in MODES or aspect_ratio not in SIZES:
        raise ValidationError("Choose one of the available modes and aspect ratios.")
    if steps is None or int(steps) != steps or not STEPS_MIN <= steps <= STEPS_MAX:
        raise ValidationError("Steps must be a whole number between 4 and 40.")
    if not randomize_seed and (
        seed is None or int(seed) != seed or not 0 <= seed <= MAX_SEED
    ):
        raise ValidationError(f"Seed must be a whole number between 0 and {MAX_SEED}.")
    refs = norm_refs(reference)
    if len(refs) > MAX_REFS:
        raise ValidationError(
            f"最多支持 {MAX_REFS} 张参考图。 / Up to {MAX_REFS} input images are supported."
        )
    if mode == "Edit an image" and not refs:
        raise ValidationError("Upload a reference image before editing.")

    return {
        "prompt": prompt,
        "mode": mode,
        "reference": refs[0] if refs else None,   # 兼容只认单图的旧调用方
        "references": refs,
        "aspect_ratio": aspect_ratio,
        "steps": int(steps),
        "seed": int(seed) if seed is not None else DEFAULT_SEED,
        "randomize_seed": bool(randomize_seed),
    }


def _is_quota_error(text: str) -> bool:
    """判断是否是 ZeroGPU 配额耗尽（官方报错原文见下）。"""
    t = (text or "").lower()
    return (
        "zerogpu runs limit" in t
        or "quota exceeded" in t
        or "you have exceeded" in t
        or ("zerogpu" in t and "quota" in t)
    )


def _is_retryable(text: str) -> bool:
    """判断是否是值得换号/重试的临时故障。"""
    t = (text or "").lower()
    return any(
        k in t
        for k in (
            "gpu task aborted",
            "timed out",
            "timeout",
            "connection",
            "temporarily unavailable",
            "503",
            "502",
            "504",
            "service unavailable",
            "restarting",
            # ZeroGPU 代理令牌过期：换连接重试即可，账号本身没问题，别换号
            "expired zerogpu proxy token",
            "zerogpu proxy token",
        )
    )


#: 匿名（无 token）调用撞限额时的报错特征。ZeroGPU 对匿名用户报得比登录用户更杂，
#: 既有 quota 系文案，也有 429 / rate limit / 要求登录 的说法，全部归到「换代理」。
_ANON_QUOTA_KEYS = (
    "quota",
    "exceeded",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "429",
    "sign in",
    "log in",
    "login",
    "anonymous",
    "gpu task aborted",
)


def _is_anon_quota_error(text: str) -> bool:
    """匿名额耗尽 / 被限流 —— 该换代理，而不是换账号。"""
    t = (text or "").lower()
    return any(k in t for k in _ANON_QUOTA_KEYS)


#: 代理链路故障特征。命中就说明「换号没用」，只能换代理或换直连。
_PROXY_ERROR_KEYS = (
    "proxyerror",
    "proxy error",
    "proxy",
    "407",
    "tunnel connection failed",
    "socks",
    "connect error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "getaddrinfo",
    "name or service not known",
    "nodename nor servname",
    "all connection attempts failed",
    "ssl",
    "certificate verify failed",
    # Windows 上代理被 RST 时抛的是中文/系统码，别漏了
    "远程主机强迫关闭",
    "forcibly closed",
    "winerror",
    "10054",
)


def _is_proxy_error(text: str) -> bool:
    """代理本身挂了（连不上 / 认证失败 / SOCKS 握手失败 / TLS 被中间人掐）。"""
    t = (text or "").lower()
    return any(k in t for k in _PROXY_ERROR_KEYS)


#: Space 级别的故障 —— 换号没用，得换 Space。
#: 「排队满」也归这里：官方 max_size=16，满了就是直接被拒，原号重试只会一直撞墙。
_SPACE_ERROR_KEYS = (
    "queue is full",
    "queue full",
    "max_size",
    "too many requests",
    "429",
    "space is in the invalid state",
    "still building",
    "is building",
    "no such app",
    "not found",
    "404",
    "hostname mismatch",
    "certificate verify failed",
    "spaceunavailable",
    "cannot be reached",
)


def _is_space_error(text: str) -> bool:
    """这个 Space 本身不可用，应该换一个 Space 而不是换账号。"""
    t = (text or "").lower()
    return any(k in t for k in _SPACE_ERROR_KEYS)



# ==========================================================================
# 4. 账号池
# ==========================================================================


@dataclass
class Account:
    """一条调用路线：带 token 的 HF 账号，或「匿名 + 代理」的匿名额。

    - 有 token：配额按账号计，``proxy`` 只是可选出口
    - 无 token（``anonymous=True``）：配额按**出口 IP** 计，所以必须挂代理，
      否则所有匿名调用共用本机 IP 那一份额度
    """

    name: str
    token: str = ""
    note: str = ""
    proxy: str | None = None        # 这条路线走的代理 URL，None = 直连
    anonymous: bool = False         # True = 无 token，吃按 IP 的匿名额

    # 运行时状态
    exhausted_until: float = 0.0   # epoch 秒，早于此刻不可用
    success: int = 0
    failure: int = 0
    last_error: str = ""
    last_used: float = 0.0
    inflight: int = 0

    @property
    def key(self) -> str:
        """路线唯一标识。匿名路线的 key 绑在代理上，换代理 = 换一份匿名额。"""
        if self.token:
            return self.token
        return f"anon:{self.proxy or 'direct'}"

    def available(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return now >= self.exhausted_until

    def cooldown_left(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.exhausted_until - now)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "name": self.name,
            "masked": _mask_token(self.token),
            "note": self.note,
            "kind": "anon" if self.anonymous else "token",
            "anonymous": self.anonymous,
            "proxy": _mask_proxy(self.proxy) if self.proxy else None,
            "available": self.available(now),
            "cooldown_left": round(self.cooldown_left(now), 1),
            "exhausted_until": (
                datetime.fromtimestamp(self.exhausted_until, timezone.utc).isoformat()
                if self.exhausted_until > now
                else None
            ),
            "success": self.success,
            "failure": self.failure,
            "inflight": self.inflight,
            "last_used": (
                datetime.fromtimestamp(self.last_used, timezone.utc).isoformat()
                if self.last_used
                else None
            ),
            "last_error": self.last_error,
        }


def _mask_token(token: str) -> str:
    """脱敏展示：只留前缀和后 4 位。"""
    token = token or ""
    if not token:
        return "（未设置）"
    if len(token) <= 12:
        return token[:4] + "***"
    return f"{token[:7]}…{token[-4:]}"


# ==========================================================================
# 4. 代理池 —— 匿名额按 IP 计，换代理 = 换一份额度
# ==========================================================================

PROXY_SCHEMES = ("http", "https", "socks5", "socks5h", "socks4")


def normalize_proxy(raw: str) -> str:
    """把用户写的代理规范成 httpx 能吃的 URL。

    ``127.0.0.1:7897`` → ``http://127.0.0.1:7897``；已有 scheme 的原样保留。
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = f"http://{s}"
    return s


def _mask_proxy(url: str | None) -> str:
    """脱敏代理：把 user:pass 里的密码换成 ***。"""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.password:
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        user = parts.username or ""
        netloc = f"{user}:***@{host}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


def _proxy_hostport(url: str) -> str:
    """取 host:port 当默认显示名。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = parts.hostname or url
    return f"{host}:{parts.port}" if parts.port else host


def _redact_line(line: str) -> str:
    """把解析失败的行打码后再回显，避免密码跟着报错一起漏出去。"""
    s = (line or "").strip()
    if "@" in s:
        head, _, tail = s.partition("@")
        if "://" in head:
            scheme, _, cred = head.partition("://")
            return f"{scheme}://{cred.split(':')[0]}:***@{tail}"
        return f"{head.split(':')[0]}:***@{tail}"
    return s[:60]


def _parse_iso_ts(value: str | None) -> float | None:
    """ISO8601 → epoch 秒。解析不了返回 None（当成永久有效）。"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip())
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso_from_days(days: int | float | None) -> str | None:
    """从「还有几天」算出到期时间戳。None 或 <=0 表示不过期。"""
    if days is None:
        return None
    try:
        d = float(days)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=d)).isoformat()


def proxy_scheme_ok(url: str) -> bool:
    try:
        return urlsplit(url).scheme.lower() in PROXY_SCHEMES
    except ValueError:
        return False


def _valid_proxy_url(url: str) -> bool:
    """代理 URL 是否可用：scheme 认识 + 有 host + 有合法端口。"""
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    if p.scheme.lower() not in PROXY_SCHEMES or not p.hostname:
        return False
    try:
        return bool(p.port)
    except ValueError:      # 端口不是数字，或 host:port:xxx 这种粘错格式
        return False


def _parse_one_proxy(line: str) -> tuple[str, str, str, str] | None:
    """解析单行代理，兼容代理商常见的几种批量导出格式。

    认这些（大小写、前后空格都无所谓）::

        http://user:pass@1.2.3.4:8080        # 标准 URL
        socks5://user:pass@1.2.3.4:1080
        1.2.3.4:8080                         # host:port
        user:pass@1.2.3.4:8080               # 缺 scheme
        1.2.3.4:8080:user:pass               # 代理商最爱用的四段式
        1.2.3.4,8080,user,pass               # CSV 四列
        香港|http://user:pass@1.2.3.4:8080|住宅IP|2026-12-31   # 带名字、备注、到期
    """
    s = line.strip().lstrip("\ufeff")
    if not s:
        return None

    # 1) 名字|url|备注|到期
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        label = parts[0]
        url = parts[1] if len(parts) > 1 else ""
        note = parts[2] if len(parts) > 2 else ""
        expires = parts[3] if len(parts) > 3 else ""
        url = normalize_proxy(url)
        if not _valid_proxy_url(url):
            return None
        return (label or _proxy_hostport(url), url, note, expires)

    # 2) 带 scheme 的完整 URL
    if "://" in s:
        url = normalize_proxy(s)
        if not _valid_proxy_url(url):
            return None
        return (_proxy_hostport(url), url, "", "")

    # 3) user:pass@host:port（缺 scheme）
    if "@" in s:
        url = "http://" + s
        if not _valid_proxy_url(url):
            return None
        return (_proxy_hostport(url), url, "", "")

    # 4) CSV 四列：host,port,user,pass
    if s.count(",") == 3:
        host, port, user, pwd = (x.strip() for x in s.split(","))
        if host and port.isdigit():
            url = f"http://{user}:{pwd}@{host}:{port}"
            if _valid_proxy_url(url):
                return (f"{host}:{port}", url, "", "")

    # 5) 冒号分段：host:port 或 host:port:user:pass
    parts = [p.strip() for p in s.split(":")]
    if len(parts) == 2 and parts[1].isdigit():
        url = f"http://{parts[0]}:{parts[1]}"
        if _valid_proxy_url(url):
            return (f"{parts[0]}:{parts[1]}", url, "", "")
    if len(parts) == 4 and parts[1].isdigit():
        host, port, user, pwd = parts
        url = f"http://{user}:{pwd}@{host}:{port}"
        if _valid_proxy_url(url):
            return (f"{host}:{port}", url, "", "")

    return None


def parse_proxy_lines(text: str) -> list[tuple[str, str, str, str]]:
    """解析代理池文本，返回 [(label, url, note, expires), ...]。

    每行一个代理，``#`` 开头为注释，空行忽略。具体格式见 ``_parse_one_proxy``。
    """
    out: list[tuple[str, str, str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _parse_one_proxy(line)
        if parsed:
            out.append(parsed)
    return out


@dataclass
class Proxy:
    """一个出口代理及其运行时健康状态。"""

    url: str
    label: str = ""
    note: str = ""
    expires: str | None = None      # ISO8601，None = 永久有效

    ok: int = 0
    fail: int = 0
    inflight: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""
    last_used: float = 0.0
    egress_ip: str = ""       # 最近一次探测到的出口 IP
    latency_ms: float = 0.0   # 最近一次探测的往返毫秒

    @property
    def key(self) -> str:
        return self.url

    def expired(self, now: float | None = None) -> bool:
        """批次到期了就整批作废 —— 这是「这批代理都失效了」的判据。"""
        ts = _parse_iso_ts(self.expires)
        if ts is None:
            return False
        return (now if now is not None else time.time()) >= ts

    def available(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if self.expired(now):
            return False
        return now >= self.cooldown_until

    def cooldown_left(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.cooldown_until - now)

    def expires_in_days(self, now: float | None = None) -> int | None:
        ts = _parse_iso_ts(self.expires)
        if ts is None:
            return None
        now = now if now is not None else time.time()
        return max(0, int((ts - now) // 86400))

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        is_expired = self.expired(now)
        return {
            "label": self.label,
            "url": _mask_proxy(self.url),
            "note": self.note,
            "expires": self.expires,
            "expires_in": self.expires_in_days(now),
            "expired": is_expired,
            "available": self.available(now),
            "cooldown_left": round(self.cooldown_left(now), 1),
            "cooldown_until": (
                datetime.fromtimestamp(self.cooldown_until, timezone.utc).isoformat()
                if self.cooldown_until > now
                else None
            ),
            "ok": self.ok,
            "fail": self.fail,
            "inflight": self.inflight,
            "egress_ip": self.egress_ip,
            "latency_ms": self.latency_ms,
            "last_error": self.last_error,
        }


class ProxyPool:
    """代理池：轮询选代理、故障/限额自动冷却。

    匿名额是**按出口 IP** 算的，所以这里每个代理等价于一份独立的匿名配额。
    代理挂了冷却短（默认 300s），匿名额跑满冷却长（默认 1800s），
    两者都可用 ``QWEN_PROXY_COOLDOWN_H`` / ``QWEN_PROXY_ERROR_COOLDOWN_S`` 覆盖。
    """

    def __init__(
        self,
        proxies: Iterable[Proxy] = (),
        *,
        quota_cooldown: float | None = None,
        error_cooldown: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._rr = 0
        self.proxies: list[Proxy] = [p for p in proxies if p.url]
        self._quota_cooldown = quota_cooldown
        self._error_cooldown = error_cooldown

    # ---------- 构造 ----------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        quota_cooldown: float | None = None,
        error_cooldown: float | None = None,
    ) -> "ProxyPool":
        p = Path(path)
        text = p.read_text(encoding="utf-8-sig") if p.exists() else ""
        items = [
            Proxy(url=u, label=lbl, note=note, expires=exp or None)
            for lbl, u, note, exp in parse_proxy_lines(text)
        ]
        return cls(items, quota_cooldown=quota_cooldown, error_cooldown=error_cooldown)

    @classmethod
    def from_env(
        cls,
        *,
        proxy_file: str | Path | None = None,
        quota_cooldown: float | None = None,
        error_cooldown: float | None = None,
    ) -> "ProxyPool":
        """优先读 ``QWEN_PROXIES``（逗号或换行分隔），其次读代理文件。"""
        items: list[Proxy] = []
        raw = os.getenv("QWEN_PROXIES", "")
        if raw:
            text = raw.replace(",", "\n")
            items = [
                Proxy(url=u, label=lbl, note=note, expires=exp or None)
                for lbl, u, note, exp in parse_proxy_lines(text)
            ]
        if not items and proxy_file:
            p = Path(proxy_file)
            if p.exists():
                text = p.read_text(encoding="utf-8-sig")
                items = [
                    Proxy(url=u, label=lbl, note=note, expires=exp or None)
                    for lbl, u, note, exp in parse_proxy_lines(text)
                ]
        return cls(items, quota_cooldown=quota_cooldown, error_cooldown=error_cooldown)

    # ---------- 冷却 ----------

    def _quota_cooldown_seconds(self) -> float:
        if self._quota_cooldown is not None:
            return float(self._quota_cooldown)
        env = os.getenv("QWEN_PROXY_COOLDOWN_H")
        if env:
            try:
                return max(0.0, float(env) * 3600.0)
            except ValueError:
                pass
        return PROXY_QUOTA_COOLDOWN

    def _error_cooldown_seconds(self) -> float:
        if self._error_cooldown is not None:
            return float(self._error_cooldown)
        env = os.getenv("QWEN_PROXY_ERROR_COOLDOWN_S")
        if env:
            try:
                return max(0.0, float(env))
            except ValueError:
                pass
        return PROXY_ERROR_COOLDOWN

    # ---------- 选代理 ----------

    def acquire(
        self, *, exclude: set[str] | None = None, now: float | None = None
    ) -> Proxy | None:
        """挑一个健康代理：在途最少优先，并列时轮询。"""
        now = now if now is not None else time.time()
        exclude = exclude or set()
        with self._lock:
            cands = [p for p in self.proxies if p.url not in exclude and p.available(now)]
            if not cands:
                return None
            fewest = min(p.inflight for p in cands)
            least = [p for p in cands if p.inflight == fewest]
            least.sort(key=lambda p: (p.fail, p.label))
            p = least[self._rr % len(least)]
            self._rr += 1
            p.inflight += 1
            p.last_used = now
            return p

    def get(self, url: str) -> Proxy | None:
        with self._lock:
            for p in self.proxies:
                if p.url == url:
                    return p
        return None

    def release(
        self, proxy: Proxy | str, *, ok: bool, error: str = "", quota: bool = False
    ) -> None:
        """归还代理。``quota=True`` 表示匿名额跑满（冷却久），否则按故障冷却。"""
        target = self.get(proxy) if isinstance(proxy, str) else proxy
        if target is None:
            return
        with self._lock:
            target.inflight = max(0, target.inflight - 1)
            if ok:
                target.ok += 1
                target.last_error = ""
                return
            target.fail += 1
            target.last_error = (error or "")[:500]
            cool = self._quota_cooldown_seconds() if quota else self._error_cooldown_seconds()
            target.cooldown_until = time.time() + cool

    def mark_probe(
        self, proxy: Proxy | str, *, ok: bool, egress_ip: str = "", latency_ms: float = 0.0, error: str = ""
    ) -> None:
        """记录一次主动连通性探测的结果（不改变冷却，只更新展示字段）。"""
        target = self.get(proxy) if isinstance(proxy, str) else proxy
        if target is None:
            return
        with self._lock:
            if ok:
                target.ok += 1
                target.egress_ip = egress_ip or target.egress_ip
                target.latency_ms = round(latency_ms, 1)
                target.last_error = ""
            else:
                target.fail += 1
                target.last_error = (error or "")[:500]

    # ---------- 运维 ----------

    def add(
        self,
        url: str,
        label: str = "",
        note: str = "",
        *,
        days: int | float | None = None,
        expires: str | None = None,
    ) -> Proxy:
        u = normalize_proxy(url)
        with self._lock:
            p = Proxy(
                url=u,
                label=label or _proxy_hostport(u),
                note=note,
                expires=expires or _iso_from_days(days),
            )
            self.proxies.append(p)
            return p

    def add_many(
        self, text: str, *, days: int | float | None = None
    ) -> dict[str, Any]:
        """批量导入代理，返回明细：added / skipped（重复）/ invalid（解析不出来）。

        ``days`` 是**这批**的到期天数（None = 永久）。行内自带第 4 段到期时间的，
        以行内为准 —— 方便把不同批次的代理混在一个文件里。
        """
        added: list[str] = []
        skipped: list[str] = []
        invalid: list[dict[str, Any]] = []
        batch_expires = _iso_from_days(days)

        with self._lock:
            seen = {p.url for p in self.proxies}
            for lineno, raw in enumerate((text or "").splitlines(), start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parsed = _parse_one_proxy(line)
                if not parsed:
                    invalid.append({"line": lineno, "raw": _redact_line(line)})
                    continue
                label, url, note, line_expires = parsed
                if url in seen:
                    skipped.append(label)
                    continue
                self.proxies.append(
                    Proxy(
                        url=url,
                        label=label,
                        note=note,
                        expires=line_expires or batch_expires,
                    )
                )
                seen.add(url)
                added.append(label)

        return {"added": added, "skipped": skipped, "invalid": invalid}

    def purge_expired(self) -> dict[str, Any]:
        """一键清理已到期的代理，返回删掉的条数与标签。"""
        with self._lock:
            now = time.time()
            dead = [p for p in self.proxies if p.expired(now)]
            if dead:
                dead_urls = {p.url for p in dead}
                self.proxies = [p for p in self.proxies if p.url not in dead_urls]
            return {
                "removed": len(dead),
                "labels": [p.label for p in dead][:50],
                "remaining": len(self.proxies),
            }

    def remove(self, url: str) -> bool:
        u = normalize_proxy(url)
        with self._lock:
            for i, p in enumerate(self.proxies):
                if p.url == u or p.label == url:
                    self.proxies.pop(i)
                    return True
        return False

    def clear_cooldown(self, url: str | None = None) -> int:
        n = 0
        with self._lock:
            for p in self.proxies:
                if url is None or p.url == normalize_proxy(url) or p.label == url:
                    if p.cooldown_until > 0:
                        p.cooldown_until = 0.0
                        n += 1
        return n

    def save(self, path: str | Path) -> None:
        lines = [
            "# Qwen Studio 代理池 —— 每行一个代理",
            "# 写法： url  或  名字|url  或  名字|url|备注  或  名字|url|备注|到期时间",
            "# 例： 127.0.0.1:7897   /   socks5://user:pass@1.2.3.4:1080",
            "",
        ]
        for p in self.proxies:
            line = f"{p.label}|{p.url}"
            if p.note or p.expires:
                line += f"|{p.note}"
            if p.expires:
                line += f"|{p.expires}"
            lines.append(line)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def snapshot(
        self, *, limit: int | None = None, offset: int = 0, q: str = ""
    ) -> dict[str, Any]:
        """代理池快照。``limit=None`` 返回全部，否则分页 —— 几千个代理时前端不能一次吃下。

        ``q`` 按名称/URL/备注过滤，过滤后的总数就是 ``total``。
        """
        now = time.time()
        with self._lock:
            pool = self.proxies
            if q:
                needle = q.strip().lower()
                pool = [
                    p for p in pool
                    if needle in (p.label or "").lower()
                    or needle in (p.url or "").lower()
                    or needle in (p.note or "").lower()
                ]
            total = len(pool)
            usable = sum(1 for p in pool if p.available(now))
            expired = sum(1 for p in pool if p.expired(now))
            cooling = max(0, total - usable - expired)
            page = pool if limit is None else pool[offset : offset + limit]
            items = [p.snapshot() for p in page]
        return {
            "total": total,
            "usable": usable,
            "cooling": cooling,
            "expired": expired,
            "offset": offset,
            "limit": total if limit is None else limit,
            "q": q,
            "proxies": items,
        }

    def summary(self) -> dict[str, Any]:
        """只要计数，不要列表 —— 给 /api/pool 这类高频接口用。"""
        now = time.time()
        with self._lock:
            total = len(self.proxies)
            usable = sum(1 for p in self.proxies if p.available(now))
            expired = sum(1 for p in self.proxies if p.expired(now))
        return {
            "total": total,
            "usable": usable,
            "cooling": max(0, total - usable - expired),
            "expired": expired,
        }

    def __len__(self) -> int:
        return len(self.proxies)


def probe_proxy(url: str, *, timeout: float = 15.0, ip_service: str = "https://api.ipify.org") -> dict[str, Any]:
    """主动探测一个代理：能否出网 + 出口 IP 是什么 + 延迟多少。

    只有这个函数需要网络；它不影响调用逻辑，纯粹给界面上的「测试」按钮用。
    """
    import httpx

    started = time.perf_counter()
    try:
        with httpx.Client(proxy=url, timeout=timeout, follow_redirects=True) as c:
            r = c.get(ip_service)
            r.raise_for_status()
            ip = (r.text or "").strip()
        return {
            "ok": True,
            "egress_ip": ip,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "egress_ip": "",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


#: 构造 gradio_client 时临时把代理写进环境变量的互斥锁。
#: huggingface_hub 的 space_info / get_space_runtime 不吃 httpx_kwargs，
#: 只认 HTTP(S)_PROXY，所以建连那一下必须靠 env 兜住。
_ENV_PROXY_LOCK = threading.Lock()
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@contextmanager
def _env_proxy(url: str | None):
    """把代理临时注入进程环境变量，退出时精确还原。"""
    if not url:
        yield
        return
    with _ENV_PROXY_LOCK:
        prev = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS}
        try:
            for k in _PROXY_ENV_KEYS:
                os.environ[k] = url
            yield
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def parse_token_lines(text: str) -> list[tuple[str, str, str, str]]:
    """解析账号池文本，返回 [(name, token, note, proxy), ...]。

    支持四种写法（`#` 开头为注释）::

        hf_xxxxxxxxxxxxxxxxxxxx
        账号A|hf_xxxxxxxxxxxxxxxxxxxx
        账号A|hf_xxxxxxxxxxxxxxxxxxxx|备用小号
        账号A|hf_xxxxxxxxxxxxxxxxxxxx|备用小号|http://127.0.0.1:7897

    第四段是该账号的出口代理：本机不能直连 HF 时必须给，
    否则 token 路线建连就会失败（配额按账号算，走哪个 IP 无所谓）。
    """
    out: list[tuple[str, str, str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("\ufeff")  # 记事本存 UTF-8 会带 BOM，先剥掉
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            token, name, note, proxy = parts[0], "", "", ""
        else:
            name, token = parts[0], parts[1] if len(parts) > 1 else ""
            note = parts[2] if len(parts) > 2 else ""
            proxy = parts[3] if len(parts) > 3 else ""
        if not token:
            continue
        proxy = normalize_proxy(proxy)
        out.append((name or f"account{len(out) + 1}", token, note, proxy))
    return out


class AccountPool:
    """HF 账号池：轮询选号、配额耗尽自动冷却、失败自动切换。

    注意：ZeroGPU 配额是**按账号**计的，但 Space 本身的
    ``concurrency_limit=1`` 是**按 Space** 计的，所以账号池的作用是
    「一个号跑满配额后换下一个号继续」，而不是并发提速。
    """

    def __init__(
        self,
        accounts: Iterable[Account] = (),
        *,
        quota_cooldown: float | None = None,
        proxy_pool: "ProxyPool | None" = None,
        strategy: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._rr = 0
        # 没有 token 且不是匿名路线的条目直接丢掉：它既没有配额也没有出口
        self.accounts: list[Account] = [
            a for a in accounts if a.token or (a.anonymous and a.proxy)
        ]
        self._quota_cooldown_override = quota_cooldown
        self.proxy_pool: ProxyPool | None = proxy_pool
        self.strategy: str = strategy if strategy in STRATEGIES else DEFAULT_STRATEGY

    # ---------- 代理池挂载 ----------

    def attach_proxy_pool(self, proxy_pool: "ProxyPool | None", strategy: str | None = None) -> None:
        """挂上代理池并切换策略。已有 token 账号的运行态不受影响。"""
        with self._lock:
            self.proxy_pool = proxy_pool
            if strategy in STRATEGIES:
                self.strategy = strategy

    def replace_accounts(self, accounts: Iterable[Account]) -> None:
        """整体替换账号列表（reload 用）。带锁，避免和 acquire/release 抢。"""
        with self._lock:
            self.accounts = [a for a in accounts if a.token or (a.anonymous and a.proxy)]

    def _acquire_anon(
        self, exclude: set[str], now: float
    ) -> Account | None:
        """从代理池借一个代理，包装成一条匿名额路线。"""
        pp = self.proxy_pool
        if pp is None or len(pp) == 0:
            return None
        ex_urls = {k[len("anon:"):] for k in exclude if k.startswith("anon:")}
        p = pp.acquire(exclude=ex_urls, now=now)
        if p is None:
            return None
        acc = Account(
            name=f"匿名@{p.label or _proxy_hostport(p.url)}",
            token="",
            note="匿名额 · 按出口 IP 计",
            proxy=p.url,
            anonymous=True,
        )
        acc.inflight = 1     # 已由 ProxyPool.acquire 计数
        acc.last_used = now
        return acc

    # ---------- 构造 ----------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        quota_cooldown: float | None = None,
        proxy_pool: "ProxyPool | None" = None,
        strategy: str | None = None,
        default_proxy: str | None = None,
    ) -> "AccountPool":
        p = Path(path)
        text = p.read_text(encoding="utf-8-sig") if p.exists() else ""
        dflt = normalize_proxy(default_proxy or "")
        accts = [
            Account(name=n, token=t, note=note, proxy=px or dflt)
            for n, t, note, px in parse_token_lines(text)
        ]
        return cls(accts, quota_cooldown=quota_cooldown, proxy_pool=proxy_pool, strategy=strategy)

    @classmethod
    def from_env(
        cls,
        *,
        token_file: str | Path | None = None,
        quota_cooldown: float | None = None,
        proxy_pool: "ProxyPool | None" = None,
        strategy: str | None = None,
        default_proxy: str | None = None,
    ) -> "AccountPool":
        """优先读 QWEN_HF_TOKENS 环境变量，其次读 tokens.txt，最后读 HF_TOKEN。

        ``default_proxy`` / ``QWEN_DEFAULT_PROXY`` 给没写第四段的账号兜一个出口 ——
        本机不能直连 HF 时，token 路线也得走代理才建得起来。
        """
        dflt = normalize_proxy(default_proxy or os.getenv("QWEN_DEFAULT_PROXY", ""))
        accts: list[Account] = []

        raw = os.getenv("QWEN_HF_TOKENS", "")
        if raw:
            for name, token, note, px in parse_token_lines(raw.replace(",", "\n")):
                accts.append(Account(name=name, token=token, note=note, proxy=px or dflt))

        if not accts and token_file:
            p = Path(token_file)
            if p.exists():
                for name, token, note, px in parse_token_lines(p.read_text(encoding="utf-8-sig")):
                    accts.append(Account(name=name, token=token, note=note, proxy=px or dflt))

        if not accts:
            for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                tok = os.getenv(env_key)
                if tok:
                    accts.append(
                        Account(name=env_key.lower(), token=tok.strip(), note="来自环境变量", proxy=dflt)
                    )
                    break

        return cls(accts, quota_cooldown=quota_cooldown, proxy_pool=proxy_pool, strategy=strategy)

    # ---------- 配额冷却 ----------

    def _quota_cooldown_seconds(self) -> float:
        """配额耗尽后的冷却时长。默认到下一个 UTC 零点（ZeroGPU 每日重置）。"""
        if self._quota_cooldown_override is not None:
            return float(self._quota_cooldown_override)
        env = os.getenv("QWEN_QUOTA_COOLDOWN_H")
        if env:
            try:
                return max(0.0, float(env) * 3600.0)
            except ValueError:
                pass
        now = datetime.now(timezone.utc)
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (nxt - now).total_seconds()

    # ---------- 选号 ----------

    def acquire(self, *, exclude: set[str] | None = None, now: float | None = None) -> Account | None:
        """按当前策略挑一条调用路线。

        - ``proxy_first``（默认）：先借代理吃匿名额，代理全废才动 token 账号
        - ``token_first``：先花账号配额，配额没了再吃匿名额
        - ``proxy_only`` / ``token_only``：只用其中一边
        """
        now = now if now is not None else time.time()
        exclude = exclude or set()
        strat = self.strategy

        if strat == "proxy_first":
            acc = self._acquire_anon(exclude, now)
            return acc if acc is not None else self._acquire_token(exclude, now)
        if strat == "proxy_only":
            return self._acquire_anon(exclude, now)
        if strat == "token_first":
            acc = self._acquire_token(exclude, now)
            return acc if acc is not None else self._acquire_anon(exclude, now)
        return self._acquire_token(exclude, now)

    def _acquire_token(self, exclude: set[str], now: float) -> Account | None:
        """挑一个可用账号。跳过 exclude 与冷却中的账号。

        优先选**在途任务最少**的账号；并列时按轮询取。
        这样长期分布均匀，不会因为某个号用得少就一直被跳过。
        """
        with self._lock:
            cands = [a for a in self.accounts if a.key not in exclude and a.available(now)]
            if not cands:
                return None
            fewest = min(a.inflight for a in cands)
            least = [a for a in cands if a.inflight == fewest]
            least.sort(key=lambda a: (a.success, a.name))
            acc = least[self._rr % len(least)]
            self._rr += 1
            acc.inflight += 1
            acc.last_used = now
            return acc

    def routes_available(self, now: float | None = None) -> int:
        """当前可尝试的路线总数（token 账号 + 可用代理），用于算尝试上限。"""
        now = now if now is not None else time.time()
        with self._lock:
            n = sum(1 for a in self.accounts if a.available(now))
        if self.proxy_pool is not None:
            n += sum(1 for p in self.proxy_pool.proxies if p.available(now))
        return n

    def release(self, acc: Account, *, ok: bool, error: str = "", quota: bool = False) -> None:
        """归还路线。匿名路线的失败/限额冷却落在**代理**上，不落在账号上。"""
        if acc.anonymous:
            if self.proxy_pool is not None and acc.proxy:
                self.proxy_pool.release(acc.proxy, ok=ok, error=error, quota=quota)
            return
        with self._lock:
            acc.inflight = max(0, acc.inflight - 1)
            if ok:
                acc.success += 1
                acc.last_error = ""
                return
            acc.failure += 1
            acc.last_error = (error or "")[:500]
            if quota:
                acc.exhausted_until = time.time() + self._quota_cooldown_seconds()

    # ---------- 运维 ----------

    def add(self, name: str, token: str, note: str = "") -> Account:
        with self._lock:
            acc = Account(name=name or f"account{len(self.accounts) + 1}", token=token, note=note)
            self.accounts.append(acc)
            return acc

    def remove(self, name: str) -> bool:
        with self._lock:
            for i, a in enumerate(self.accounts):
                if a.name == name:
                    self.accounts.pop(i)
                    return True
        return False

    def clear_cooldown(self, name: str | None = None) -> int:
        with self._lock:
            n = 0
            for a in self.accounts:
                if name in (None, a.name) and a.exhausted_until > 0:
                    a.exhausted_until = 0.0
                    n += 1
            return n

    def save(self, path: str | Path) -> None:
        lines = [
            "# Qwen Studio 账号池 —— 每行一个 HF token",
            "# 写法： token  或  账号名|token  或  账号名|token|备注",
            "# 获取： https://huggingface.co/settings/tokens",
            "",
        ]
        for a in self.accounts:
            if not a.token:      # 匿名路线由代理池负责，不写进 tokens.txt
                continue
            lines.append(f"{a.name}|{a.token}" + (f"|{a.note}" if a.note else ""))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            accts = [a.snapshot() for a in self.accounts]
        usable = sum(1 for a in self.accounts if a.available(now))
        snap: dict[str, Any] = {
            "total": len(accts),
            "usable": usable,
            "cooling": sum(1 for a in self.accounts if not a.available(now)),
            "accounts": accts,
            "strategy": self.strategy,
            "proxy": self.proxy_pool.summary() if self.proxy_pool is not None else None,
            "routes_available": self.routes_available(now),
        }
        return snap

    def __len__(self) -> int:
        return len(self.accounts)


# ==========================================================================
# 5. 客户端
# ==========================================================================


def _image_size(path: str | Path) -> tuple[int, int] | None:
    """读图片的真实像素尺寸。读不出来返回 None，由调用方决定兜底。"""
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1]
    except Exception:  # noqa: BLE001
        return None


def _as_path(value: Any) -> str | None:
    """把 gradio_client 的返回值（str / dict / tuple）统一取成本地路径。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("url")
    if isinstance(value, (list, tuple)):
        for v in value:
            p = _as_path(v)
            if p:
                return p
        return None
    return getattr(value, "path", None)


class QwenStudio:
    """官方 Space 的客户端封装：校验 → 选号 → 提交 → 轮询 → 落地 PNG。"""

    def __init__(
        self,
        pool: AccountPool,
        *,
        out_dir: str | Path = "outputs",
        max_attempts: int | None = None,
        poll_interval: float = 0.6,
        job_timeout: float = 900.0,
        request_timeout: float = 600.0,
        same_account_retries: int = 2,
        retry_backoff: float = 2.0,
        spaces: list[str] | None = None,
    ) -> None:
        self.pool = pool
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gradio_tmp = self.out_dir / "_gradio"
        self.gradio_tmp.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts  # None = 每次调用时按当前池子大小动态取
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self.request_timeout = request_timeout
        self.same_account_retries = max(0, same_account_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        #: 可轮换的 Space。每个 Space 是**独立队列**，一个排满了换下一个。
        self.spaces: list[str] = list(spaces or SPACE_IDS)
        self._last_ok_space: str | None = None

        self._lock = threading.Lock()
        self._clients: dict[str, Client] = {}
        #: 每个 Space 的 aspect_ratio 可选值（不同 Space 的值格式可能不同）
        self._aspect_cache: dict[str, list[str]] = {}

    # ---------- Space 兼容层 ----------

    @staticmethod
    def _aspect_choices(client: Any) -> list[str]:
        """从 ``/config`` 里挖出 aspect_ratio 那个 Dropdown 的 choices。

        不同 Space 虽然参数名一样，**选项值可能不同** ——
        比如有的写 ``Square · 1:1``，有的写 ``Square · 1:1 (1024x1024)``。
        """
        cfg = getattr(client, "config", None) or {}
        comps = {c.get("id"): c for c in (cfg.get("components") or [])}
        for dep in cfg.get("dependencies") or []:
            if dep.get("api_name") not in ("generate", "/generate"):
                continue
            for cid in dep.get("inputs") or []:
                props = (comps.get(cid) or {}).get("props") or {}
                ch = props.get("choices")
                if not ch:
                    continue
                vals = [str(x[0]) if isinstance(x, (list, tuple)) else str(x) for x in ch]
                if any("\u00b7" in v for v in vals):   # 画幅键里的中黑点
                    return vals
        return []

    def _resolve_aspect(self, client: Any, want: str) -> str:
        """把本地键映射到该 Space 实际接受的选项值。匹配不上就原样发。"""
        src = getattr(client, "src", "") or ""
        if src not in self._aspect_cache:
            try:
                self._aspect_cache[src] = self._aspect_choices(client)
            except Exception:  # noqa: BLE001
                self._aspect_cache[src] = []
        choices = self._aspect_cache[src]
        if not choices or want in choices:
            return want
        for c in choices:
            if c.startswith(want):
                log.info("Space 的画幅值不同：%r → %r", want, c)
                return c
        return want

    def _space_order(self) -> list[str]:
        """上次成功的 Space 排最前 —— 冷启动很贵，别每次换着唤醒。"""
        order = list(self.spaces)
        last = self._last_ok_space
        if last and last in order:
            order.remove(last)
            order.insert(0, last)
        return order

    def set_spaces(self, spaces: list[str]) -> list[str]:
        """运行时替换渠道列表（控制台开关渠道时用），立即生效，不用重启。"""
        with self._lock:
            self.spaces = [s.strip() for s in spaces if s and s.strip()]
            if self._last_ok_space not in self.spaces:
                self._last_ok_space = None
            # 已关掉的渠道，缓存的连接一并丢掉
            for ckey in list(self._clients):
                if ckey.split("|", 1)[0] not in {space_src(s) for s in self.spaces}:
                    self._clients.pop(ckey, None)
            return list(self.spaces)

    # ---------- 连接管理 ----------

    def _client(self, acc: Account, space: str | None = None) -> Client:
        src = space_src(space)
        ckey = f"{src}|{acc.key}"
        with self._lock:
            c = self._clients.get(ckey)
            if c is not None:
                return c

            # 每条路线一套独立指纹：UA/语言、TLS cipher 顺序。
            # 同一路线恒定，不同路线不同 —— 同一出口 IP 每次换指纹反而更可疑。
            fp = fingerprint.for_route(acc.key)

            kwargs: dict[str, Any] = {"timeout": self.request_timeout}
            if acc.proxy:
                kwargs["proxy"] = acc.proxy
            # 注意：httpx_kwargs 会被 gradio_client 透传给 httpx.get/post/stream，
            # 那几个顶层函数只认 proxy/timeout/verify/headers/cookies/auth 这一小撮，
            # 塞 http2 / transport / limits 会直接 TypeError。
            # SSLContext 也不能走 verify 键 —— gradio_client 每个请求都显式传
            # verify=self.ssl_verify，两边同时给会 "multiple values"。
            # 它只是把 ssl_verify 原样转发给 httpx，所以直接传 SSLContext 进去即可。
            ctx = fingerprint.ssl_context(fp.cipher_set)

            log.info(
                "连接 Space（%s，路线=%s，出口=%s，指纹=%s）…",
                space or SPACE_ID,
                acc.name,
                _mask_proxy(acc.proxy) or "直连",
                fp.name,
            )
            # token=False 才是真匿名：token=None 会去读本机缓存的 HF token，
            # 那样「匿名」就白搭了。env 代理锁兜住 huggingface_hub 那几条元数据请求。
            with _env_proxy(acc.proxy):
                c = Client(
                    src,
                    token=acc.token or False,
                    verbose=False,
                    headers=fp.merged_headers() or None,
                    ssl_verify=ctx if ctx is not None else True,
                    download_files=str(self.gradio_tmp),
                    httpx_kwargs=kwargs,
                )
            self._clients[ckey] = c
            return c

    def _drop_client(self, acc: Account, space: str | None = None) -> None:
        ckey = f"{space_src(space)}|{acc.key}"
        with self._lock:
            self._clients.pop(ckey, None)

    # ---------- 调用 ----------

    def generate(
        self,
        prompt: str,
        mode: str = DEFAULT_MODE,
        reference: Any = None,
        aspect_ratio: str = DEFAULT_ASPECT,
        steps: float = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        randomize_seed: bool = True,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """与官方 generate() 参数顺序一致；返回带本地路径的 dict。

        多个 Space 时逐个试：一个排满了 / 没启动 / 域名错，换下一个。
        每个 Space 都是**独立队列**，所以这比死等一个快得多。
        """
        params = validate_params(prompt, mode, reference, aspect_ratio, steps, seed, randomize_seed)

        errors: list[str] = []
        for space in self._space_order():
            try:
                result = self._run_on_space(space, params, on_progress, should_cancel)
            except Cancelled:
                raise
            except (SpaceUnavailable, PoolExhausted) as exc:
                errors.append(f"{space}：{exc}")
                log.warning("Space %s 不可用，换下一个：%s", space, str(exc)[:140])
                continue
            except QwenError as exc:
                errors.append(f"{space}：{exc}")
                continue
            result["space"] = space
            self._last_ok_space = space
            return result

        pp = self.pool.proxy_pool
        has_proxy = pp is not None and len(pp) > 0
        if not self.pool.accounts and not has_proxy:
            raise PoolExhausted(
                "账号池和代理池都是空的，无法生成。任选其一："
                "把 Hugging Face token 写进 tokens.txt（每行一个，或 名字|token），"
                "或设 QWEN_HF_TOKENS 环境变量；"
                "代理写进 proxies.txt（每行一个，如 127.0.0.1:7897），或设 QWEN_PROXIES。"
            )
        cooling = [a.name for a in self.pool.accounts if not a.available()]
        cooling_px = [p.label for p in pp.proxies if not p.available()] if has_proxy else []
        extra = ""
        if cooling:
            extra += f" 冷却中的账号：{', '.join(cooling)}"
        if cooling_px:
            extra += f" 冷却中的代理：{', '.join(cooling_px)}"
        detail = "；".join(errors) if errors else "没有可用路线"
        raise PoolExhausted(
            f"全部路线已试完（{len(self.spaces)} 个 Space，策略 {self.pool.strategy}）。{detail}{extra}"
        )

    def _run_on_space(
        self,
        space: str,
        params: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        """在一个 Space 上试遍所有路线。成功返回结果；Space 级故障抛 SpaceUnavailable。"""
        tried: set[str] = set()
        last_err: Exception | None = None
        # 尝试上限按「当前可尝试的路线数」算：token 账号 + 可用代理，动态取，
        # 所以服务跑起来之后新加的号/新加的代理也能立刻用上。
        attempts = self.max_attempts or max(1, self.pool.routes_available())

        for _ in range(attempts):
            acc = self.pool.acquire(exclude=tried)
            if acc is None:
                break
            tried.add(acc.key)

            # 同一个账号上，瞬时故障（连接抖动、ZeroGPU 代理令牌过期等）允许原号重试；
            # 只有配额耗尽才立刻换号，换号要重新排队，代价很高。
            for retry in range(self.same_account_retries + 1):
                try:
                    result = self._call(acc, space, params, on_progress, should_cancel)
                except Cancelled:
                    self.pool.release(acc, ok=True)
                    raise
                except AnonQuotaExceeded as exc:
                    # 匿名额按 IP 计：这个代理的额度用完了，换下一个代理，别动 token 账号
                    log.warning("匿名额用尽（%s），换代理：%s", acc.name, str(exc)[:150])
                    self.pool.release(acc, ok=False, error=str(exc), quota=True)
                    self._drop_client(acc, space)
                    last_err = exc
                    break
                except ProxyError as exc:
                    # 代理链路本身坏了：原号重试没有意义，直接换下一条路线
                    log.warning("代理不可用（%s），换下一条路线：%s", acc.name, str(exc)[:150])
                    self.pool.release(acc, ok=False, error=str(exc), quota=False)
                    self._drop_client(acc, space)
                    last_err = exc
                    break
                except QuotaExceeded as exc:
                    log.warning("账号 %s 配额耗尽，切换下一个（%s）", acc.name, exc)
                    self.pool.release(acc, ok=False, error=str(exc), quota=True)
                    last_err = exc
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    # Space 级故障（排队满 / 没启动 / 域名错）：换号没用，直接换 Space
                    if _is_space_error(msg):
                        self.pool.release(acc, ok=False, error=msg, quota=False)
                        self._drop_client(acc, space)
                        raise SpaceUnavailable(msg) from exc
                    transient = _is_retryable(msg)
                    if transient and retry < self.same_account_retries:
                        log.warning(
                            "账号 %s 瞬时故障，原号重试 %d/%d：%s",
                            acc.name, retry + 1, self.same_account_retries, msg[:120],
                        )
                        self._drop_client(acc, space)  # 强制重建连接，重新拿 ZeroGPU 代理令牌
                        time.sleep(self.retry_backoff * (retry + 1))
                        continue
                    log.warning("账号 %s 调用失败：%s", acc.name, msg[:200])
                    self.pool.release(acc, ok=False, error=msg, quota=_is_quota_error(msg))
                    if transient:
                        self._drop_client(acc, space)
                    last_err = exc
                    break
                else:
                    self.pool.release(acc, ok=True)
                    result["account"] = acc.name
                    return result

        if last_err is not None:
            raise last_err
        raise PoolExhausted(f"{space} 上没有可用路线")

    # ---------- 内部：真正打一次接口 ----------

    @staticmethod
    def _classify(exc: BaseException, acc: Account) -> QwenError:
        """把底层异常归到「换代理」/「换账号」/「原号重试」三类。

        匿名路线上，配额和限流都意味着「这个出口 IP 不能用了」→ 换代理；
        代理链路故障同理。只有 token 账号才区分「配额耗尽」和「瞬时抖动」。
        """
        msg = str(exc) or type(exc).__name__
        if acc.anonymous:
            if _is_anon_quota_error(msg):
                return AnonQuotaExceeded(msg)
            if _is_proxy_error(msg):
                return ProxyError(msg)
            return QwenError(msg)
        if _is_quota_error(msg):
            return QuotaExceeded(msg)
        if _is_proxy_error(msg):
            return ProxyError(msg)
        return QwenError(msg)

    def _call(
        self,
        acc: Account,
        space: str | None,
        params: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        client = self._client(acc, space)

        # 官方只有 Edit 模式会读 reference；Create / Transparent 传了也是白传，
        # 还会把图白传一遍上去，这里直接不发。
        refs = params["references"] if params["mode"] == "Edit an image" else []
        native_multi = space_supports_multi(space)
        stitched: str | None = None
        if not refs:
            ref_arg = None
        elif len(refs) == 1:
            ref_arg = handle_file(str(refs[0]))
        elif native_multi:
            # Gallery 输入：整批按顺序传，pipeline 给每张图一个 <imageN> 块
            ref_arg = [handle_file(str(p)) for p in refs]
        else:
            # 单图 Space：本地拼成一张，prompt 里按「图1/图2…」指代
            stitched = stitch_refs(refs, out_dir=self.gradio_tmp)
            ref_arg = handle_file(stitched)

        # 有的 Space 把尺寸写进了选项值里，发之前先按它的 choices 校正
        aspect = self._resolve_aspect(client, params["aspect_ratio"])

        args = [
            params["prompt"],
            params["mode"],
            ref_arg,
            aspect,
            params["steps"],
            params["seed"],
            params["randomize_seed"],
        ]

        started = time.perf_counter()
        job = client.submit(*args, api_name=API_NAME)

        def emit(**kw: Any) -> None:
            if on_progress:
                try:
                    on_progress({"elapsed": round(time.perf_counter() - started, 1), **kw})
                except Exception:  # noqa: BLE001
                    pass

        emit(state="queued", rank=None, queue_size=None)

        while not job.done():
            if should_cancel and should_cancel():
                try:
                    job.cancel()
                except Exception:  # noqa: BLE001
                    pass
                raise Cancelled("任务已取消")
            if time.perf_counter() - started > self.job_timeout:
                raise QwenError(f"任务超时（>{self.job_timeout:.0f}s）")
            try:
                st = job.status()
                raw = getattr(st, "code", None)
                code = (getattr(raw, "name", None) or (str(raw) if raw else "")).lower()
                if code and code != "processing":
                    emit(state=code, rank=getattr(st, "rank", None),
                         queue_size=getattr(st, "queue_size", None),
                         eta=getattr(st, "eta", None))
            except Exception:  # noqa: BLE001
                pass
            # 固定间隔轮询是机器特征，打散成随机节奏
            time.sleep(fingerprint.jitter(self.poll_interval))

        exc = None
        try:
            exc = job.exception()
        except Exception as e:  # noqa: BLE001
            exc = e
        if exc is not None:
            raise self._classify(exc, acc) from exc

        try:
            outputs = job.outputs()
        except Exception as e:  # noqa: BLE001
            raise self._classify(e, acc) from e

        if not outputs:
            raise QwenError("接口未返回任何输出")

        first = outputs[0] if len(outputs) == 1 else outputs
        seq = list(first) if isinstance(first, (list, tuple)) else [first]
        while len(seq) < 4:
            seq.append(None)

        image_path = _as_path(seq[0])
        download_path = _as_path(seq[1]) or image_path
        actual_seed = seq[2]
        details = seq[3] if isinstance(seq[3], str) else ""

        if not image_path:
            raise QwenError("接口未返回图片路径")

        saved = self._persist(image_path, actual_seed)

        # 尺寸以**实际返回的像素**为准，而不是查本地那张表。
        # 官方 SIZES 是它自己的实现细节；哪天它改了，我们不该跟着报错尺寸。
        expected = SIZES[params["aspect_ratio"]]
        actual = _image_size(saved) or expected
        if actual != expected:
            log.warning(
                "返回尺寸 %dx%d 与官方 SIZES[%s]=%dx%d 不一致，以实际为准",
                actual[0], actual[1], params["aspect_ratio"], expected[0], expected[1],
            )
        return {
            "image_path": saved,
            "download_path": saved,
            "remote_path": image_path,
            "seed": int(actual_seed) if actual_seed is not None else params["seed"],
            "details": details,
            "elapsed": round(time.perf_counter() - started, 2),
            "width": actual[0],
            "height": actual[1],
            "params": {k: v for k, v in params.items() if k not in ("reference", "references")},
            "reference": refs[0] if refs else None,
            "references": list(refs),
            "reference_count": len(refs),
            "native_multi": bool(native_multi),
            "stitched": stitched,
            "mode": params["mode"],
            "prompt": params["prompt"],
            "account": acc.name,
            "route": "anon" if acc.anonymous else "token",
            "proxy": _mask_proxy(acc.proxy) if acc.proxy else None,
            "fingerprint": fingerprint.for_route(acc.key).name,
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def _persist(self, src: str, seed: Any) -> str:
        """HF 的产物文件 24h 后过期（delete_cache=(3600, 86400)），必须立刻落地。"""
        import shutil

        srcp = Path(src)
        if not srcp.exists():
            raise QwenError(f"下载结果不存在：{src}")
        name = OUTPUT_NAME.format(seed=seed if seed is not None else int(time.time()))
        dst = self.out_dir / name
        n = 1
        while dst.exists():
            dst = self.out_dir / f"{dst.stem}-{n}{dst.suffix}"
            n += 1
        if srcp.resolve() != dst.resolve():
            shutil.copy2(srcp, dst)
        return str(dst)


# ==========================================================================
# 6. 命令行
# ==========================================================================


def _build_pool(args: argparse.Namespace) -> AccountPool:
    token_file = args.tokens or (Path(__file__).parent / "tokens.txt")
    pool = AccountPool.from_env(
        token_file=token_file,
        default_proxy=getattr(args, "default_proxy", None),
    )
    proxy_file = getattr(args, "proxies", None) or (Path(__file__).parent / "proxies.txt")
    proxy_pool = ProxyPool.from_env(proxy_file=proxy_file)
    pool.attach_proxy_pool(proxy_pool, getattr(args, "strategy", None))
    return pool


def _cmd_pool(args: argparse.Namespace) -> int:
    pool = _build_pool(args)
    snap = pool.snapshot()
    print(
        f"策略 {snap['strategy']}  |  账号总数 {snap['total']}  |  可用 {snap['usable']}  |  "
        f"冷却中 {snap['cooling']}  |  可尝试路线 {snap['routes_available']}"
    )
    print("-" * 78)
    for a in snap["accounts"]:
        state = "可用" if a["available"] else f"冷却 {a['cooldown_left']:.0f}s"
        print(
            f"  {a['name']:<16} {a['masked']:<22} {state:<14} "
            f"ok={a['success']:<4} fail={a['failure']:<4} {a['last_error'][:40]}"
        )

    px = snap.get("proxy") or {}
    print("-" * 78)
    print(
        f"代理池：共 {px.get('total', 0)} 个，可用 {px.get('usable', 0)} 个，"
        f"冷却中 {px.get('cooling', 0)} 个，已过期 {px.get('expired', 0)} 个"
    )
    pp = pool.proxy_pool
    if pp is not None and len(pp):
        page = pp.snapshot(limit=20)
        for p in page["proxies"]:
            if p["expired"]:
                state = "已过期"
            elif p["available"]:
                state = "可用"
            else:
                state = f"冷却 {p['cooldown_left']:.0f}s"
            ip = p.get("egress_ip") or "-"
            exp = (
                "永久" if not p.get("expires")
                else f"{p['expires_in']}天后"
            )
            print(
                f"  {p['label']:<22} {p['url']:<38} {state:<12} 到期={exp:<10} "
                f"ok={p['ok']:<4} fail={p['fail']:<4} 出口={ip:<16} {p['last_error'][:24]}"
            )
        if page["total"] > 20:
            print(f"  … 还有 {page['total'] - 20} 个，完整列表在界面「代理」页签里看")

    if snap["routes_available"] == 0:
        print("\n提示：账号池和代理池里都没有可用路线，无法生成。")
        print("      账号：把 HF token 填进 tokens.txt；代理：填进 proxies.txt（如 127.0.0.1:7897）。")
    return 0


def _cmd_proxy_test(args: argparse.Namespace) -> int:
    pool = _build_pool(args)
    pp = pool.proxy_pool
    if pp is None or len(pp) == 0:
        print("代理池是空的，没什么可测的。")
        return 0
    limit = max(1, int(getattr(args, "proxy_limit", 20) or 20))
    live = [p for p in pp.proxies if not p.expired()]
    skipped = len(pp.proxies) - len(live)
    targets = live[:limit]
    print(
        f"探测 {len(targets)} / {len(live)} 个可用代理的出口 IP …（--proxy-limit 可调）"
        + (f" 已跳过 {skipped} 个过期代理" if skipped else "")
    )
    for p in targets:
        r = probe_proxy(p.url)
        pp.mark_probe(p, ok=r["ok"], egress_ip=r["egress_ip"], latency_ms=r["latency_ms"], error=r["error"])
        flag = "OK  " if r["ok"] else "FAIL"
        print(
            f"  {flag} {p.label:<22} {_mask_proxy(p.url):<38} "
            f"出口={r['egress_ip'] or '-':<16} {r['latency_ms']:>8.0f}ms {r['error'][:50]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Qwen Image 2.1 GGUF Studio 客户端（官方参数一致 + HF 账号池）"
    )
    ap.add_argument("--prompt", "-p", help="提示词，≤4000 字符")
    ap.add_argument("--mode", "-m", default=DEFAULT_MODE, choices=MODES)
    ap.add_argument("--ref", help="参考图路径（Edit an image 模式必填）")
    ap.add_argument(
        "--refs",
        nargs="+",
        help=f"多张参考图，最多 {MAX_REFS} 张，按顺序对应 prompt 里的「图1/图2…」",
    )
    ap.add_argument("--aspect", "-a", default=DEFAULT_ASPECT, choices=list(SIZES))
    ap.add_argument("--steps", "-s", type=int, default=DEFAULT_STEPS, help=f"{STEPS_MIN}-{STEPS_MAX}")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-randomize", action="store_true", help="固定 seed，便于复现")
    ap.add_argument("--out", "-o", default="outputs", help="输出目录")
    ap.add_argument("--tokens", help="账号池文件路径，默认 ./tokens.txt")
    ap.add_argument("--proxies", help="代理池文件路径，默认 ./proxies.txt")
    ap.add_argument(
        "--default-proxy",
        help="给没写第四段的 token 账号兜一个出口（本机不能直连 HF 时必需）",
    )
    ap.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=DEFAULT_STRATEGY,
        help="调用路线：proxy_first=先吃匿名额(默认) / token_first / proxy_only / token_only",
    )
    ap.add_argument("--pool", action="store_true", help="只打印账号池 + 代理池状态")
    ap.add_argument("--proxy-test", action="store_true", help="探测每个代理的出口 IP")
    ap.add_argument("--proxy-limit", type=int, default=20, help="--proxy-test 最多测几个（默认 20）")
    ap.add_argument("--list-sizes", action="store_true", help="打印全部画幅")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_sizes:
        for k, (w, h) in SIZES.items():
            print(f"  {k:<20} {w}×{h}")
        return 0

    if args.pool:
        return _cmd_pool(args)

    if args.proxy_test:
        return _cmd_proxy_test(args)

    if not args.prompt:
        ap.error("--prompt 必填（或用 --pool / --list-sizes）")

    pool = _build_pool(args)
    studio = QwenStudio(pool, out_dir=args.out)

    def prog(ev: dict[str, Any]) -> None:
        bits = [f"{ev.get('elapsed', 0):>6.1f}s"]
        if ev.get("state"):
            bits.append(f"状态={ev['state']}")
        if ev.get("rank") is not None:
            bits.append(f"队列第 {ev['rank']} 位/{ev.get('queue_size')}")
        print("  " + "  ".join(bits), end="\r", flush=True)

    try:
        res = studio.generate(
            prompt=args.prompt,
            mode=args.mode,
            reference=args.refs or args.ref,
            aspect_ratio=args.aspect,
            steps=args.steps,
            seed=args.seed,
            randomize_seed=not args.no_randomize,
            on_progress=prog,
        )
    except QwenError as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 2

    print(" " * 70, end="\r")
    print(f"完成 · {res['details']}")
    print(f"  文件   : {res['image_path']}")
    print(f"  seed   : {res['seed']}")
    print(f"  路线   : {res['route']}" + (f"（{res['proxy']}）" if res.get("proxy") else "（直连）"))
    print(f"  账号   : {res['account']}")
    print(f"  耗时   : {res['elapsed']}s")
    if res.get("reference_count"):
        how = "原生多图" if res.get("native_multi") else "本地拼图"
        print(f"  参考图 : {res['reference_count']} 张（{how}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
