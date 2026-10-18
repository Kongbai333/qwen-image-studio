---
title: Qwen Image 2.1 GGUF Studio · Multi
emoji: 🎨
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
short_description: Qwen Image 2.1 Q4_K_M with up to 10 reference images
startup_duration_timeout: 1h
models:
  - abenzerps/Qwen-Image-2.1-Uncensored-GGUF
  - Qwen/Qwen-Image-2.1
---

# Qwen Image 2.1 GGUF Studio · Multi

Fork of `assembledchaos/qwen-image-2-1-studio` that lifts the reference upload from a
single `gr.Image` to a `gr.Gallery` of up to **10** images. Nothing else about the
pipeline changes: `QwenImage21Pipeline` already takes a list.

- **Create an image:** describe a scene.
- **Edit an image:** upload **1–10** reference images, in the order you want to refer to them.
- **Transparent PNG:** Qwen's recommended transparency prompt is prepended.

## Why a list works

`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`:

```python
image = image if isinstance(image, list) else [image]      # line 610
```

then `_get_qwen_prompt_embeds` builds one `<image1>..<imageN>` block per entry and the
block-causal transformer lets later blocks attend to earlier ones. That is why **order
matters**: the prompt should say "the first image" / "the second image" following the
upload order. The model card documents up to 10 references.

## What this fork changes

| Original | This fork |
|---|---|
| `reference = gr.Image(...)` | `reference = gr.Gallery(type="pil", columns=5)` |
| `reference: Image.Image \| None` | `reference: list \| None` |
| `kwargs["image"] = reference` | `kwargs["image"] = [prepared, ...]` |
| `@spaces.GPU(duration=60)` | `@spaces.GPU(duration=gpu_duration)` — scales with image count |
| — | `MAX_INPUT_IMAGES = 10` validation, bilingual error |
| — | `use_kv_cache = kv_cache_fits(n)` — drops the prefix cache past 10 GB |

## Calling it

The `/generate` argument order is unchanged, so existing clients keep working — only
position 3 becomes a list:

```python
client.submit(prompt, "Edit an image", [f1, f2], "Square · 1:1", 40, 42, True,
              api_name="/generate")
```

## Hosting

Requires `zero-a10g` hardware (set it in the Space settings; README metadata does not
select hardware). No external inference API key is needed.
