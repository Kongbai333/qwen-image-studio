import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["GRADIO_EXAMPLES_CACHE"] = ".gradio/gguf-q4-k-m-multi-examples"

import spaces
import gradio as gr
import torch
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

from accelerate import init_empty_weights
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.models.model_loading_utils import load_gguf_checkpoint
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, PngImagePlugin

MODEL_ID = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
MODEL_REVISION = "e92386d3b86fd77ae1650534815ab3f6b85d46ba"
CHECKPOINT = "qwen-image-2.1-Q4_K_M.gguf"
CHECKPOINT_SHA256 = "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8"
COMPANION_ID = "Qwen/Qwen-Image-2.1"
COMPANION_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MODES = ["Create an image", "Edit an image", "Transparent PNG"]
SIZES = {
    "Square · 1:1": (1024, 1024),
    "Landscape · 16:9": (1344, 768),
    "Portrait · 9:16": (768, 1344),
    "Landscape · 4:3": (1152, 864),
    "Portrait · 3:4": (864, 1152),
}
MAX_SEED = 2**31 - 1

#: 模型原生支持的上限（Qwen-Image-2.1 模型卡：Support up to 10 reference images）
MAX_INPUT_IMAGES = 10
#: 前缀 KV cache 每 1K 条件图约 2GB；超过预算就关掉 cache 换显存
KV_CACHE_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
KV_CACHE_BUDGET_GB = 10.0
NUM_INFERENCE_STEPS = 40

print(f"Loading {MODEL_ID} at {MODEL_REVISION}", flush=True)
checkpoint_path = hf_hub_download(MODEL_ID, CHECKPOINT, revision=MODEL_REVISION)
with open(checkpoint_path, "rb") as checkpoint_file:
    checksum = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
if checksum != CHECKPOINT_SHA256:
    raise RuntimeError("The GGUF checkpoint checksum does not match the published SHA256SUMS.")

# Expand the requested GGUF once for ZeroGPU's eager tensor packing. This
# preserves the quantized checkpoint's values without per-step dequantization.
weights = load_gguf_checkpoint(checkpoint_path)
for name in weights:
    weights[name] = dequantize_gguf_tensor(weights[name]).to(torch.bfloat16)
config = QwenImage21Transformer2DModel.load_config(
    COMPANION_ID, subfolder="transformer", revision=COMPANION_REVISION
)
with init_empty_weights():
    transformer = QwenImage21Transformer2DModel.from_config(config)
transformer.load_state_dict(weights, strict=True, assign=True)
transformer.eval().requires_grad_(False)
print(f"Verified {CHECKPOINT}: loaded all {len(weights)} tensors; sha256={checksum}", flush=True)
del weights

# Supplying the transformer prevents the upstream diffusion weights from
# being downloaded. Only its compatible text encoder, VAE and config are used.
pipe = QwenImage21Pipeline.from_pretrained(
    COMPANION_ID, revision=COMPANION_REVISION, transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")
print(f"{MODEL_ID} / {CHECKPOINT} loaded on CUDA via ZeroGPU.", flush=True)


def kv_cache_fits(n_images: int) -> bool:
    """条件图的 prefix KV cache 装不装得下。装不下就退回重算，省显存但更慢。"""
    return n_images * (1024 // 16) ** 2 * KV_CACHE_BYTES_PER_TOKEN <= KV_CACHE_BUDGET_GB * 1e9


def gpu_duration(
    prompt: str | None = None,
    mode: str | None = None,
    reference: list | None = None,
    aspect_ratio: str | None = None,
    steps: int | float | None = NUM_INFERENCE_STEPS,
    seed: int | None = None,
    randomize_seed: bool | None = None,
    **_ignored,
) -> int:
    """ZeroGPU 申请多少秒。

    每多一张参考图，prefix 就多一份 latent token；不按张数加时长，
    多图请求会被 ZeroGPU 在跑到一半时掐掉（单图能跑、三张必挂）。
    """
    n_images = len(reference or [])
    width, height = SIZES.get(aspect_ratio or "", (1024, 1024))
    pixels = width * height
    prefix = n_images * (1024 // 16) ** 2
    cached = kv_cache_fits(n_images)
    tokens = pixels / 256 + (0.2 * prefix if cached else prefix)
    per_step = 3.4e-6 * tokens ** 1.363 * 1.3
    fixed = 4 + 2 * n_images + 1.5e-6 * pixels
    return int(min(300, (fixed + (steps or NUM_INFERENCE_STEPS) * per_step) * 1.25))


@spaces.GPU(duration=gpu_duration)
def generate(
    prompt: str,
    mode: str = "Create an image",
    reference: list | None = None,
    aspect_ratio: str = "Square · 1:1",
    steps: int = 40,
    seed: int = 42,
    randomize_seed: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, int, str]:
    """Generate or edit an image with abenzerps' Qwen-Image-2.1 Q4_K_M GGUF.

    Edit an image accepts up to 10 reference images; every entry becomes its
    own block in the joint sequence, so the order matters: the prompt should
    say "the first image" / "the second image" following the upload order.
    """
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Describe the image you want to create or the change to make.")
    if len(prompt) > 4000:
        raise gr.Error("Keep your prompt under 4,000 characters.")
    if mode not in MODES or aspect_ratio not in SIZES:
        raise gr.Error("Choose one of the available modes and aspect ratios.")
    if steps is None or int(steps) != steps or not 4 <= steps <= 40:
        raise gr.Error("Steps must be a whole number between 4 and 40.")
    if not randomize_seed and (
        seed is None or int(seed) != seed or not 0 <= seed <= MAX_SEED
    ):
        raise gr.Error(f"Seed must be a whole number between 0 and {MAX_SEED}.")

    refs = [r for r in (reference or []) if r is not None]
    if len(refs) > MAX_INPUT_IMAGES:
        raise gr.Error(
            f"最多支持 {MAX_INPUT_IMAGES} 张输入图片。 / "
            f"Up to {MAX_INPUT_IMAGES} input images are supported."
        )
    if mode == "Edit an image" and not refs:
        raise gr.Error("Upload a reference image before editing.")

    actual_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    width, height = SIZES[aspect_ratio]
    effective_prompt = prompt
    if mode == "Transparent PNG":
        effective_prompt = (
            "This is an RGBA image with transparency. " + prompt
            + " The image has alpha channel and the background is transparent."
        )

    kwargs = {}
    if mode == "Edit an image":
        prepared = []
        for ref in refs:
            ref = ImageOps.exif_transpose(ref).convert("RGBA")
            ref.thumbnail((2048, 2048))
            prepared.append(ref)
        # list 直接进 pipeline：QwenImage21Pipeline 会按顺序给每张图一个
        # <image1>..<imageN> 块（pipeline_qwenimage21.py:610 / 258-266）
        kwargs["image"] = prepared
        kwargs["use_kv_cache"] = kv_cache_fits(len(prepared))

    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=effective_prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            generator=torch.Generator("cuda").manual_seed(actual_seed),
            **kwargs,
        ).images[0]
    elapsed = time.perf_counter() - start
    metadata = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_dtype": "bfloat16",
        "prompt": prompt,
        "effective_prompt": effective_prompt,
        "mode": mode,
        "seed": actual_seed,
        "steps": int(steps),
        "width": result.width,
        "height": result.height,
        "seconds": round(elapsed, 2),
        "reference_count": len(refs) if mode == "Edit an image" else 0,
    }
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("generation", json.dumps(metadata, ensure_ascii=False))
    # Gradio owns its output cache and expires files after 24 hours.
    output_dir = Path(tempfile.mkdtemp(prefix="qwen21-gguf-", dir=gr.utils.get_upload_folder()))
    path = output_dir / f"qwen-image-2.1-gguf-{actual_seed}.png"
    result.save(path, pnginfo=png_info)
    details = (
        f"{result.width} × {result.height} · {int(steps)} steps · "
        f"{elapsed:.1f}s · seed {actual_seed} · {result.mode} PNG · Q4_K_M"
    )
    if mode == "Edit an image" and len(refs) > 1:
        details += f" · {len(refs)} refs"
    if mode == "Transparent PNG" and (
        "A" not in result.getbands() or result.getchannel("A").getextrema()[0] == 255
    ):
        details += " · The model returned an opaque image; try another seed or prompt."
    print(f"Completed {mode}: {details}", flush=True)
    return str(path), str(path), actual_seed, details


def show_reference(mode: str) -> gr.Gallery:
    """Show the reference upload for image editing."""
    return gr.Gallery(visible=mode == "Edit an image")


CSS = """
.gradio-container { max-width: 1180px !important; margin: auto !important; }
.dark .gradio-container { color: var(--body-text-color); }
#intro { padding: 40px 0 12px; }
#intro h1 { font-size: clamp(28px, 4vw, 44px); letter-spacing: -1.5px; }
#result { min-height: 420px; }
#generate { min-height: 48px; }
"""

with gr.Blocks(title="Qwen Image 2.1 GGUF Studio · Multi", delete_cache=(3600, 86400)) as demo:
    gr.Markdown(
        "# Qwen Image 2.1 GGUF Studio · Multi\n"
        "Turn an idea into an image. Reimagine a photo with up to 10 references. Create a transparent asset.\n\n"
        "[abenzerps / Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) · **Q4_K_M** · "
        "[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)",
        elem_id="intro",
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            mode = gr.Radio(MODES, value=MODES[0], label="What would you like to do?")
            prompt = gr.Textbox(
                label="Your prompt",
                placeholder="A tiny observatory on a floating island, warm evening light, detailed illustration…",
                lines=5,
                max_lines=10,
            )
            reference = gr.Gallery(
                label="Reference images (up to 10, order matters)",
                type="pil",
                interactive=True,
                columns=5,
                height="auto",
                visible=False,
            )
            aspect_ratio = gr.Dropdown(list(SIZES), value="Square · 1:1", label="Aspect ratio")
            with gr.Accordion("Generation settings", open=False):
                steps = gr.Slider(4, 40, value=40, step=1, label="Steps", info="40 is Qwen's recommended setting. Lower values give quicker previews.")
                randomize_seed = gr.Checkbox(True, label="Use a new seed each time")
                seed = gr.Number(value=42, precision=0, minimum=0, maximum=MAX_SEED, label="Seed", info="Turn off new seeds to reproduce a result.")
            run = gr.Button("Generate image", variant="primary", elem_id="generate")
            gr.Markdown("Shared ZeroGPU compute. Queue times and daily usage limits apply.")
        with gr.Column(scale=6):
            result = gr.Image(label="Your image", type="filepath", image_mode="RGBA", format="png", interactive=False, elem_id="result")
            download = gr.File(label="Download original PNG", interactive=False)
            details = gr.Textbox(label="Generation details", interactive=False)

    mode.change(show_reference, inputs=mode, outputs=reference, api_visibility="private", queue=False)
    run.click(
        generate,
        inputs=[prompt, mode, reference, aspect_ratio, steps, seed, randomize_seed],
        outputs=[result, download, seed, details],
        api_name="generate", concurrency_limit=1, concurrency_id="qwen-pipeline",
    )
    gr.Examples(
        examples=[
            ['A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement', MODES[0]],
            ["A tiny observatory on a floating island above clouds, warm sunset, intricate storybook illustration", MODES[0]],
            ["A cute cartoon dragon sticker, jade green scales, a friendly expression, clean edges", MODES[2]],
        ],
        inputs=[prompt, mode], outputs=[result, download, seed, details],
        fn=generate, cache_examples=True, cache_mode="lazy", label="Try an idea",
    )
    gr.Markdown(
        "For editing, upload up to 10 images and describe what should change. "
        "Every image becomes its own block, so refer to them in upload order "
        "(\"the flowers from the first image\", \"the scene in the second image\"). "
        "Downloaded PNGs include the prompt and settings."
    )

if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=CSS, mcp_server=True, max_file_size="10mb",
    )
