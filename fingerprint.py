#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""请求指纹隔离
==============

目的：让「同一份服务、多个出口 IP」看起来不像同一个人在刷。

检测方能拿到的东西，按可靠性排序：

1. **Cookie / session** —— 最硬的关联证据。同一 session 换 IP = 铁证。
   本项目每个调用路线一个独立的 ``gradio_client.Client``，cookie jar 与
   ``session_hash`` 都是各自的，这一层天然隔离。
2. **TLS 指纹（JA3/JA4）** —— Python 的 ssl 默认 cipher 顺序固定，所有路线一模一样。
   这里按路线给每个 Client 配不同的 SSLContext（改 cipher 顺序），把 JA3 的
   cipher 段打散。**两处改不了**：TLS 1.3 的三个 cipher（Python 3.13 的 ssl 没有
   ``set_ciphersuites``）和 extensions 段顺序。所以 JA3 只是"彼此不同"，不是"像浏览器"。
3. **Header 组合** —— Accept-Language / Accept / Cache-Control 有细微差异。
4. **请求节奏** —— 固定 0.6s 轮询是机器特征，加抖动。

**做不到的两件事**（`gradio_client` 把 httpx 参数透传给了 `httpx.get()` 这类顶层函数，
而它们只接受 proxy/timeout/verify/headers/cookies/auth 这一小撮）：

- 自定义 ``http2`` / ``transport`` / ``limits`` —— 传进去直接 ``TypeError``
- 每条路线换 HTTP 版本 —— 同上，只能全部 HTTP/1.1

两种模式（``QWEN_FP_MODE``）::

    balanced（默认） UA 保持 gradio_client 自己的，只打散 TLS cipher 顺序 / 语言 / 节奏。
                     诚实：UA 与真实客户端一致，不制造"UA 说浏览器、JA3 说 Python"
                     这种更扎眼的不匹配。
    stealth          UA 换成真实浏览器（配套 sec-ch-ua），同时把 cipher 换成该
                     浏览器的顺序。比 balanced 更像人，但 TLS 扩展顺序仍与真浏览器
                     不同 —— 高精度检测仍能看出破绽，属于"提高门槛"而非"隐身"。
    off              完全不动（调试用）。

一个刻意的设计：**同一路线始终用同一套指纹**。如果同一个出口 IP 每次请求
换 UA / 换协议，那比固定指纹更可疑 —— 真实用户不会每刷新一次就换浏览器。
"""

from __future__ import annotations

import hashlib
import os
import random
import ssl
from dataclasses import dataclass, field
from typing import Any

MODES = ("balanced", "stealth", "off")
DEFAULT_MODE = os.getenv("QWEN_FP_MODE", "balanced").strip().lower()

#: TLS 1.2 的 cipher 顺序（会影响 JA3 的 cipher 段）。
#: 都保留 ECDHE + AEAD，不会削弱安全性，只是顺序与集合不同。
CIPHER_SETS: dict[str, str | None] = {
    "default": None,  # OpenSSL/Python 默认顺序
    "chrome": (
        "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
        "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:"
        "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305"
    ),
    "firefox": (
        "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
        "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:"
        "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"
    ),
    "safari": (
        "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:"
        "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256"
    ),
    "lean": "ECDHE+AESGCM:ECDHE+CHACHA20",
}


@dataclass(frozen=True)
class Profile:
    """一套请求指纹。``user_agent=None`` 表示沿用 gradio_client 自带的。"""

    name: str
    cipher_set: str = "default"
    user_agent: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def merged_headers(self) -> dict[str, str]:
        h = dict(self.headers)
        if self.user_agent:
            h["user-agent"] = self.user_agent
        return h


# ==========================================================================
# balanced：不伪装 UA，只打散能打散的
# ==========================================================================

_LANG_POOL = (
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,zh-CN;q=0.8",
    "en-CA,en;q=0.9,fr;q=0.7",
    "en-AU,en;q=0.9",
    "en-US,en;q=0.8,de;q=0.6",
)

BALANCED: tuple[Profile, ...] = tuple(
    Profile(
        name=f"balanced-{i + 1}",
        cipher_set=cipher,
        headers={"accept-language": lang},
    )
    for i, (cipher, lang) in enumerate(
        (
            ("default", _LANG_POOL[0]),
            ("chrome", _LANG_POOL[1]),
            ("firefox", _LANG_POOL[2]),
            ("safari", _LANG_POOL[3]),
            ("lean", _LANG_POOL[4]),
            ("chrome", _LANG_POOL[5]),
        )
    )
)

# ==========================================================================
# stealth：换成真实浏览器 UA，并让 cipher 顺序与之一致
# ==========================================================================

_CHROME_WIN = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_CHROME_MAC = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_EDGE_WIN = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)
_FIREFOX_WIN = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
_SAFARI_MAC = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.1 Safari/605.1.15"
)


def _chrome_headers(platform: str) -> dict[str, str]:
    return {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": f'"{platform}"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }


STEALTH: tuple[Profile, ...] = (
    Profile("stealth-chrome-win", "chrome", _CHROME_WIN, _chrome_headers("Windows")),
    Profile("stealth-chrome-mac", "chrome", _CHROME_MAC, _chrome_headers("macOS")),
    Profile(
        "stealth-edge-win", "chrome", _EDGE_WIN,
        {**_chrome_headers("Windows"),
         "sec-ch-ua": '"Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24"'},
    ),
    Profile(
        "stealth-firefox-win", "firefox", _FIREFOX_WIN,
        {"accept": "*/*", "accept-language": "en-US,en;q=0.5",
         "sec-fetch-dest": "empty", "sec-fetch-mode": "cors", "sec-fetch-site": "same-origin"},
    ),
    Profile(
        "stealth-safari-mac", "safari", _SAFARI_MAC,
        {"accept": "*/*", "accept-language": "en-US,en;q=0.9"},
    ),
)

OFF = (Profile("off"),)


def mode() -> str:
    m = os.getenv("QWEN_FP_MODE", DEFAULT_MODE).strip().lower()
    return m if m in MODES else "balanced"


def profiles() -> tuple[Profile, ...]:
    m = mode()
    if m == "off":
        return OFF
    return STEALTH if m == "stealth" else BALANCED


def for_route(key: str) -> Profile:
    """按路线 key 确定性地挑一套指纹 —— 同一路线永远同一套。"""
    pool = profiles()
    if len(pool) == 1:
        return pool[0]
    digest = hashlib.sha256((key or "default").encode("utf-8")).hexdigest()
    return pool[int(digest[:12], 16) % len(pool)]


# ==========================================================================
# SSLContext / 抖动
# ==========================================================================

_CTX_CACHE: dict[str, ssl.SSLContext | None] = {}


def ssl_context(cipher_set: str) -> ssl.SSLContext | None:
    """按 cipher 集造一个 SSLContext。造不出来就返回 None（回落到默认验证）。"""
    if cipher_set in _CTX_CACHE:
        return _CTX_CACHE[cipher_set]
    spec = CIPHER_SETS.get(cipher_set)
    ctx: ssl.SSLContext | None = None
    if spec:
        try:
            ctx = ssl.create_default_context()
            ctx.set_ciphers(spec)
        except (ssl.SSLError, ValueError):
            ctx = None
    _CTX_CACHE[cipher_set] = ctx
    return ctx


def jitter(base: float, *, lo: float = 0.62, hi: float = 1.55) -> float:
    """把固定轮询间隔打散成随机节奏。"""
    if base <= 0:
        return 0
    return base * random.uniform(lo, hi)


def describe(profile: Profile) -> dict[str, Any]:
    """给控制台/日志看的摘要，不含任何敏感值。"""
    return {
        "name": profile.name,
        "cipher_set": profile.cipher_set,
        "ua": "gradio_client（沿用）" if not profile.user_agent else profile.user_agent.split(")")[0] + ")",
        "header_keys": sorted(profile.merged_headers()),
    }


def summary() -> dict[str, Any]:
    """当前模式 + 所有 profile 的摘要，供 /api/overview 展示。"""
    pool = profiles()
    return {
        "mode": mode(),
        "count": len(pool),
        "profiles": [describe(p) for p in pool],
    }
