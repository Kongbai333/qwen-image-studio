#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探测 Hugging Face Space 的 /generate 签名，判断能不能直接替换当前 Space。

用法::

    python probe_spaces.py                                  # 跑内置候选列表
    python probe_spaces.py owner/name owner/name2           # 指定
    python probe_spaces.py --proxy http://127.0.0.1:7897 owner/name
    python probe_spaces.py --search qwen-image              # 先搜再探

判断标准是两件事，缺一不可：

1. **参数名与顺序**和本地一致（``prompt, mode, reference, aspect_ratio, steps, seed, randomize_seed``）
2. **选项值**能对上（有的 Space 把尺寸写进画幅值里，本项目会自动做前缀映射）

探测逻辑在 ``channels.probe_space``，控制台的「渠道」页签用的是同一套。
"""

from __future__ import annotations

import argparse
import concurrent.futures

import httpx

from channels import REF_PARAMS, probe_space, resolve_probe_proxy

CANDIDATES = [
    "assembledchaos/qwen-image-2-1-studio",
    "Valentinisjdidjis/qwen-image-2.1-uncensored-gguf",
    "hugging-apps/qwen-image-2-1",
    "Viggle/Qwen-Image-2.1-viggle-turbo",
    "BlueSkyXN/Qwen-Image-2.1-studio",
    "Qwen/Qwen-Image-2.1",
    "PrunaAI/Pruna-Qwen-Image-2.1",
    "arudradey/qwen-image-2.1-uncensored-gguf",
]


def search(term: str, *, proxy: str | None = None, limit: int = 30) -> list[str]:
    with httpx.Client(proxy=proxy, timeout=60, follow_redirects=True) as c:
        r = c.get(
            "https://huggingface.co/api/spaces",
            params={"search": term, "sort": "likes", "direction": -1, "limit": limit},
        )
        r.raise_for_status()
        return [s["id"] for s in r.json() if s.get("sdk") == "gradio"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="探测 Space 的 /generate 兼容性")
    ap.add_argument("spaces", nargs="*", help="owner/name，留空则跑内置候选")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，如 http://127.0.0.1:7897")
    ap.add_argument("--search", help="先搜 Space 再探测；多个关键词用逗号分隔")
    ap.add_argument("--limit", type=int, default=40, help="每个关键词最多取几条")
    args = ap.parse_args(argv)

    proxy = args.proxy or resolve_probe_proxy()
    spaces = list(args.spaces)

    if args.search:
        terms = [t.strip() for t in args.search.split(",") if t.strip()]
        found: dict[str, str] = {}
        for term in terms:
            try:
                hits = search(term, proxy=proxy, limit=args.limit)
                for sid in hits:
                    found.setdefault(sid, term)
                print(f"  [{term}] 命中 {len(hits)} 个")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{term}] 搜索失败：{type(exc).__name__}: {exc}")
        print(f"合并去重后 {len(found)} 个 gradio Space")
        spaces = list(found) + spaces

    if not spaces:
        spaces = CANDIDATES

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(lambda s: probe_space(s, proxy=proxy), spaces))

    print(f"\n{'space':<48} {'stage':<10} {'likes':<6} 结论")
    print("-" * 118)
    for r in rows:
        stage = str(r.get("stage", "?"))[:10]
        likes = str(r.get("likes", ""))
        ok = r.get("compatible") is True
        reason = r.get("reason") or r.get("error") or "?"
        mark = "★ " if ok else "  "
        print(f"{mark}{r['space']:<46} {stage:<10} {likes:<6} {reason}")
        if ok and r.get("aspects"):
            print(f"{'':<50}画幅值示例：{r['aspects'][0]}")

    usable = [r["space"] for r in rows if r.get("compatible") is True]
    if usable:
        print("\n可以直接配进 QWEN_SPACES（逗号分隔）：")
        print("  " + ",".join(usable))
    print("\n参照签名:", REF_PARAMS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
