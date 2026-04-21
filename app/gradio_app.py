"""
ClearShot Phase 6 Gradio application.

This app reuses the existing ClearShotPipeline for both single-image and batch
enhancement. It is designed to be deployment-ready locally and on platforms
such as Hugging Face Spaces, while still handling limited environments
gracefully (missing LoRA path, CPU-only execution, runtime download errors).
"""

from __future__ import annotations

import io
import os
import shutil
import time
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "inference_config.yaml")
DEFAULT_LORA_PATH = str(PROJECT_ROOT / "ClearShot_checkpoints" / "final")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "app_outputs"
DEFAULT_CONFIG_DISPLAY = "configs/inference_config.yaml"
DEFAULT_LORA_DISPLAY = "ClearShot_checkpoints/final"


def _to_abs_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _ensure_src_on_path() -> None:
    import sys

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_src_on_path()
from src.pipeline.enhancement_pipeline import ClearShotPipeline, EnhancementResult  # noqa: E402


@dataclass
class RuntimeOptions:
    config_path: str
    lora_path: Optional[str]
    device: Optional[str]


class PipelineManager:
    """Caches pipeline instances by runtime settings."""

    def __init__(self) -> None:
        self._pipelines: Dict[Tuple[str, Optional[str], Optional[str]], ClearShotPipeline] = {}
        self._lock = Lock()

    def get_pipeline(self, options: RuntimeOptions) -> ClearShotPipeline:
        key = (options.config_path, options.lora_path, options.device)
        with self._lock:
            if key not in self._pipelines:
                self._pipelines[key] = ClearShotPipeline(
                    config_path=options.config_path,
                    lora_weights_path=options.lora_path,
                    device=options.device,
                )
            return self._pipelines[key]


PIPELINES = PipelineManager()


def _runtime_summary(pipeline: ClearShotPipeline, lora_path: Optional[str]) -> str:
    lora_label = lora_path if lora_path else "disabled"
    return (
        f"Device: `{pipeline.device}`  \n"
        f"Config: `{pipeline.config and 'loaded' or 'defaults'}`  \n"
        f"LoRA path: `{lora_label}`"
    )


def _validate_runtime_paths(config_path: str, lora_path: str, enable_lora: bool) -> Tuple[str, Optional[str], List[str]]:
    notes: List[str] = []
    config_abs = _to_abs_path(config_path)
    if not Path(config_abs).exists():
        raise FileNotFoundError(f"Inference config not found: {config_abs}")

    lora_abs: Optional[str] = None
    if enable_lora:
        lora_abs = _to_abs_path(lora_path)
        if not Path(lora_abs).exists():
            notes.append(
                f"LoRA path not found at `{lora_abs}`. Falling back to base SD + ControlNet."
            )
            lora_abs = None
    else:
        notes.append("LoRA disabled by user. Running the base SD + ControlNet pipeline.")

    return config_abs, lora_abs, notes


def _prepare_pipeline(
    config_path: str,
    lora_path: str,
    device_choice: str,
    enable_lora: bool,
) -> Tuple[ClearShotPipeline, List[str]]:
    config_abs, lora_abs, notes = _validate_runtime_paths(config_path, lora_path, enable_lora)
    device = None if device_choice == "auto" else device_choice
    options = RuntimeOptions(config_path=config_abs, lora_path=lora_abs, device=device)
    pipeline = PIPELINES.get_pipeline(options)
    if pipeline.device == "cpu":
        notes.append(
            "Running on CPU. Enhancement may be slow because diffusion and super-resolution are computationally heavy."
        )
    return pipeline, notes


def _set_request_overrides(
    pipeline: ClearShotPipeline,
    inference_steps: int,
    guidance_scale: float,
    strength: float,
) -> Dict[str, float]:
    pipe_cfg = pipeline.config.setdefault("pipeline", {})
    previous = {
        "num_inference_steps": pipe_cfg.get("num_inference_steps", 30),
        "guidance_scale": pipe_cfg.get("guidance_scale", 7.5),
        "strength": pipe_cfg.get("strength", 0.45),
    }
    pipe_cfg["num_inference_steps"] = int(inference_steps)
    pipe_cfg["guidance_scale"] = float(guidance_scale)
    pipe_cfg["strength"] = float(strength)
    return previous


def _restore_request_overrides(pipeline: ClearShotPipeline, previous: Dict[str, float]) -> None:
    pipe_cfg = pipeline.config.setdefault("pipeline", {})
    pipe_cfg.update(previous)


def _save_image_to_temp(image: Image.Image, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="clearshot_")
    os.close(fd)
    image.save(path)
    return path


def _emit_progress(
    progress_fn: Optional[Callable[[float, str], None]],
    fraction: float,
    desc: str,
) -> None:
    if progress_fn is None:
        return
    progress_fn(max(0.0, min(1.0, fraction)), desc=desc)


def _make_progress_mapper(
    progress_fn: Optional[Callable[[float, str], None]],
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    span = max(0.0, end - start)

    def emit(local_fraction: float, desc: str) -> None:
        _emit_progress(progress_fn, start + span * max(0.0, min(1.0, local_fraction)), desc)

    return emit


def _stage_gallery(result) -> List[Tuple[Image.Image, str]]:
    entries = [
        (result.mask, "Mask"),
        (result.edge_map, "Edges"),
        (result.diffusion_output, "Diffusion Output"),
        (result.with_background, "Refined Background"),
        (result.final, "Final Output"),
    ]
    return [(img, label) for img, label in entries if img is not None]


def _status_block(notes: List[str], result=None) -> str:
    lines = ["### Runtime Status"]
    for note in notes:
        lines.append(f"- {note}")
    if result is not None:
        meta = result.metadata
        timings = meta.get("timings", {})
        lines.append("")
        lines.append("### Inference Summary")
        lines.append(f"- Device: `{meta.get('device', 'unknown')}`")
        lines.append(f"- LoRA loaded: `{meta.get('lora_loaded', False)}`")
        lines.append(f"- Background: `{meta.get('bg_type', 'white')}`")
        lines.append(f"- Super-resolution: `{meta.get('sr_enabled', False)}`")
        lines.append(f"- Total time: `{meta.get('total_time_seconds', 0):.2f}s`")
        for stage_name, stage_time in timings.items():
            lines.append(f"- {stage_name}: `{stage_time:.2f}s`")
    return "\n".join(lines)


def _run_pipeline_with_progress(
    pipeline: ClearShotPipeline,
    image,
    bg_type: str,
    enable_sr: bool,
    seed: Optional[int],
    progress_fn: Optional[Callable[[float, str], None]] = None,
):
    import torch

    t_start = time.time()
    timings: Dict[str, float] = {}

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    result = EnhancementResult(original=image.copy())

    pipeline_res = 512
    if image.size != (pipeline_res, pipeline_res):
        image = image.resize((pipeline_res, pipeline_res), Image.BILINEAR)

    pipe_cfg = pipeline.config.get("pipeline", {})
    prompt_cfg = pipeline.config.get("prompt", {})
    bg_cfg = pipeline.config.get("background", {})

    prompt = prompt_cfg.get(
        "template",
        "professional product photography, studio lighting, clean white background, high quality, 4k, detailed",
    )
    negative_prompt = prompt_cfg.get(
        "negative",
        "blurry, noisy, low quality, distorted, deformed, watermark, text",
    )
    bg_type = bg_type or bg_cfg.get("mode", "white")

    stage1 = _make_progress_mapper(progress_fn, 0.05, 0.15)
    stage2 = _make_progress_mapper(progress_fn, 0.15, 0.25)
    stage3 = _make_progress_mapper(progress_fn, 0.25, 0.85)
    stage4 = _make_progress_mapper(progress_fn, 0.85, 0.93)
    stage5 = _make_progress_mapper(progress_fn, 0.93, 0.99)

    stage1(0.0, "Stage 1/5: Background removal")
    t0 = time.time()
    product_rgba, mask = pipeline.bg_remover.remove_background(image)
    result.product_rgba = product_rgba
    result.mask = mask
    timings["bg_removal"] = time.time() - t0
    stage1(1.0, "Stage 1/5: Background removal complete")

    stage2(0.0, "Stage 2/5: Edge extraction")
    t0 = time.time()
    edge_map = pipeline.edge_extractor.extract_for_controlnet(image)
    result.edge_map = edge_map
    timings["edge_extraction"] = time.time() - t0
    stage2(1.0, "Stage 2/5: Edge extraction complete")

    stage3(0.0, "Stage 3/5: Diffusion running")
    t0 = time.time()
    diff_enhancer = pipeline.diffusion_enhancer
    generator = None
    if seed is not None:
        generator = torch.Generator(device=diff_enhancer.device).manual_seed(seed)

    num_inference_steps = int(pipe_cfg.get("num_inference_steps", 30))
    diffusion_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=edge_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=pipe_cfg.get("guidance_scale", 7.5),
        controlnet_conditioning_scale=pipe_cfg.get("controlnet_conditioning_scale", 0.8),
        strength=pipe_cfg.get("strength", 0.45),
        generator=generator,
    )

    def step_callback(_pipe, step_index, _timestep, callback_kwargs):
        total_steps = max(1, num_inference_steps)
        step_num = min(total_steps, int(step_index) + 1)
        stage3(step_num / total_steps, f"Stage 3/5: Diffusion running, step {step_num}/{total_steps}")
        return callback_kwargs

    try:
        pipe_result = diff_enhancer.pipe(
            **diffusion_kwargs,
            callback_on_step_end=step_callback,
        )
    except TypeError:
        stage3(0.1, "Stage 3/5: Diffusion running")
        pipe_result = diff_enhancer.pipe(**diffusion_kwargs)

    diffusion_output = pipe_result.images[0]
    result.diffusion_output = diffusion_output
    timings["diffusion"] = time.time() - t0
    stage3(1.0, "Stage 3/5: Diffusion complete")

    stage4(0.0, "Stage 4/5: Background refinement")
    t0 = time.time()
    diff_rgba = pipeline._apply_mask_to_image(diffusion_output, mask)
    with_background = pipeline.bg_remover.apply_studio_background(
        diff_rgba, mask, bg_type=bg_type, shadow=True
    )
    result.with_background = with_background
    timings["bg_refinement"] = time.time() - t0
    stage4(1.0, "Stage 4/5: Background refinement complete")

    if enable_sr:
        stage5(0.0, "Stage 5/5: Super-resolution")
        t0 = time.time()
        final = pipeline.super_resolver.upscale(with_background)
        result.final = final
        timings["super_resolution"] = time.time() - t0
        stage5(1.0, "Stage 5/5: Super-resolution complete")
    else:
        result.final = with_background
        stage5(1.0, "Stage 5/5: Super-resolution skipped")

    total_time = time.time() - t_start
    result.metadata = {
        "timings": timings,
        "total_time_seconds": total_time,
        "input_size": image.size,
        "output_size": result.final.size,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "bg_type": bg_type,
        "sr_enabled": enable_sr,
        "seed": seed,
        "lora_loaded": diff_enhancer.lora_loaded,
        "device": pipeline.device,
    }
    _emit_progress(progress_fn, 1.0, "Completed")
    return result


def enhance_single(
    image: Optional[Image.Image],
    config_path: str,
    lora_path: str,
    enable_lora: bool,
    device_choice: str,
    background_choice: str,
    inference_steps: int,
    guidance_scale: float,
    strength: float,
    enable_sr: bool,
    seed: int,
    progress=gr.Progress(track_tqdm=False),
):
    if image is None:
        raise gr.Error("Please upload an image before running enhancement.")

    _emit_progress(progress, 0.02, "Preparing pipeline")
    pipeline, notes = _prepare_pipeline(config_path, lora_path, device_choice, enable_lora)
    previous = _set_request_overrides(pipeline, inference_steps, guidance_scale, strength)

    try:
        actual_seed = None if seed < 0 else int(seed)
        result = _run_pipeline_with_progress(
            pipeline,
            image=image,
            bg_type=background_choice,
            enable_sr=enable_sr,
            seed=actual_seed,
            progress_fn=progress,
        )
        _emit_progress(progress, 0.98, "Preparing outputs")
    except Exception as exc:  # noqa: BLE001
        debug_hint = traceback.format_exc(limit=2)
        message = (
            f"Enhancement failed: {type(exc).__name__}: {exc}\n\n"
            f"Tips:\n"
            f"- Verify model weights and internet access for first-time model downloads.\n"
            f"- Try CPU mode only for debugging, not for large runs.\n"
            f"- Confirm the LoRA path points to `adapter_config.json` and `adapter_model.safetensors`.\n\n"
            f"Debug snippet:\n{debug_hint}"
        )
        raise gr.Error(message) from exc
    finally:
        _restore_request_overrides(pipeline, previous)

    final_path = _save_image_to_temp(result.final)
    status_md = _status_block(notes, result)
    _emit_progress(progress, 1.0, "Done")
    return result.original, result.final, _stage_gallery(result), final_path, status_md


def _collect_batch_images(files: List[gr.File]) -> List[Path]:
    paths: List[Path] = []
    for file_obj in files or []:
        if not file_obj:
            continue
        file_path = getattr(file_obj, "name", None) or str(file_obj)
        p = Path(file_path)
        if p.is_file():
            paths.append(p)
    return paths


def enhance_batch(
    files,
    config_path: str,
    lora_path: str,
    enable_lora: bool,
    device_choice: str,
    background_choice: str,
    inference_steps: int,
    guidance_scale: float,
    strength: float,
    enable_sr: bool,
    seed: int,
    progress=gr.Progress(track_tqdm=False),
):
    image_paths = _collect_batch_images(files)
    if not image_paths:
        raise gr.Error("Please upload one or more images for batch processing.")

    pipeline, notes = _prepare_pipeline(config_path, lora_path, device_choice, enable_lora)
    previous = _set_request_overrides(pipeline, inference_steps, guidance_scale, strength)

    output_dir = Path(tempfile.mkdtemp(prefix="clearshot_batch_"))
    preview_items: List[Tuple[Image.Image, str]] = []
    failures: List[str] = []

    try:
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, start=1):
            image_start = (idx - 1) / max(total, 1)
            image_end = idx / max(total, 1)
            image_progress = _make_progress_mapper(progress, image_start, image_end)
            image_progress(0.0, f"Batch {idx}/{total}: preparing {img_path.name}")
            try:
                actual_seed = None if seed < 0 else int(seed)
                result = _run_pipeline_with_progress(
                    pipeline,
                    image=img_path,
                    bg_type=background_choice,
                    enable_sr=enable_sr,
                    seed=actual_seed,
                    progress_fn=lambda frac, desc, p=image_progress, i=idx, t=total, n=img_path.name: p(
                        frac, f"Batch {i}/{t}: {n} | {desc}"
                    ),
                )
                result.final.save(output_dir / f"{img_path.stem}_final.png")
                if len(preview_items) < 8:
                    preview_items.append((result.final, img_path.name))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{img_path.name}: {type(exc).__name__}: {exc}")
    finally:
        _restore_request_overrides(pipeline, previous)

    archive_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_dir.glob("*")):
            zf.write(file_path, arcname=file_path.name)

    notes.append(f"Processed `{len(image_paths) - len(failures)}` of `{len(image_paths)}` images.")
    if failures:
        notes.append("Some files failed during batch processing:")
        notes.extend([f"`{line}`" for line in failures[:8]])
    _emit_progress(progress, 1.0, "Batch processing complete")
    return preview_items, str(archive_path), _status_block(notes)


def build_app() -> gr.Blocks:
    default_device = os.environ.get("CLEARSHOT_DEVICE", "auto")
    default_config = os.environ.get("CLEARSHOT_CONFIG_PATH", DEFAULT_CONFIG_DISPLAY)
    default_lora = os.environ.get("CLEARSHOT_LORA_PATH", DEFAULT_LORA_DISPLAY)

    with gr.Blocks(title="ClearShot") as demo:
        gr.Markdown(
            """
            # ClearShot
            AI-powered product photo enhancement for e-commerce.

            This app reuses the existing ClearShot pipeline. It is optimized for
            GPU execution, but can still start in a limited environment for local testing.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                enable_lora = gr.Checkbox(label="Enable LoRA", value=True)
                device_choice = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda", "cpu"],
                    value=default_device if default_device in {"auto", "cuda", "cpu"} else "auto",
                )
            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    Use the main controls below for normal enhancement. Runtime path overrides
                    are available under **Advanced Settings** for local debugging or deployment setup.
                    """
                )

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    config_path = gr.Textbox(
                        label="Inference Config Path",
                        value=default_config,
                        placeholder=DEFAULT_CONFIG_DISPLAY,
                        info="Optional override. Repo-relative paths are supported.",
                    )
                    lora_path = gr.Textbox(
                        label="LoRA Weights Path",
                        value=default_lora,
                        placeholder=DEFAULT_LORA_DISPLAY,
                        info="Optional override. Repo-relative paths are supported.",
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        **Defaults**

                        - Config: `configs/inference_config.yaml`
                        - LoRA: `ClearShot_checkpoints/final`
                        """
                    )

        with gr.Tabs():
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Input Image")
                        background_choice = gr.Dropdown(
                            label="Background",
                            choices=["white", "gradient", "studio"],
                            value="white",
                        )
                        inference_steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                        guidance_scale = gr.Slider(1.0, 12.0, value=7.5, step=0.1, label="Guidance Scale")
                        strength = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Enhancement Strength")
                        enable_sr = gr.Checkbox(label="Enable Super-Resolution", value=True)
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        run_button = gr.Button("Enhance Image", variant="primary")
                    with gr.Column(scale=1):
                        before_image = gr.Image(type="pil", label="Before")
                        after_image = gr.Image(type="pil", label="After")
                        download_file = gr.File(label="Download Enhanced Image")
                        status_md = gr.Markdown(_status_block(["Waiting for input."]))

                stage_gallery = gr.Gallery(
                    label="Intermediate Stages",
                    columns=5,
                    height="auto",
                    object_fit="contain",
                )

                run_button.click(
                    fn=enhance_single,
                    inputs=[
                        input_image,
                        config_path,
                        lora_path,
                        enable_lora,
                        device_choice,
                        background_choice,
                        inference_steps,
                        guidance_scale,
                        strength,
                        enable_sr,
                        seed,
                    ],
                    outputs=[before_image, after_image, stage_gallery, download_file, status_md],
                )

            with gr.Tab("Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        batch_background = gr.Dropdown(
                            label="Background",
                            choices=["white", "gradient", "studio"],
                            value="white",
                        )
                        batch_steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                        batch_guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.1, label="Guidance Scale")
                        batch_strength = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Enhancement Strength")
                        batch_sr = gr.Checkbox(label="Enable Super-Resolution", value=True)
                        batch_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        batch_run = gr.Button("Run Batch", variant="primary")
                    with gr.Column(scale=1):
                        batch_gallery = gr.Gallery(
                            label="Batch Preview",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        batch_zip = gr.File(label="Download Batch ZIP")
                        batch_status = gr.Markdown(_status_block(["Waiting for batch input."]))

                batch_run.click(
                    fn=enhance_batch,
                    inputs=[
                        batch_files,
                        config_path,
                        lora_path,
                        enable_lora,
                        device_choice,
                        batch_background,
                        batch_steps,
                        batch_guidance,
                        batch_strength,
                        batch_sr,
                        batch_seed,
                    ],
                    outputs=[batch_gallery, batch_zip, batch_status],
                )

        gr.Markdown(
            """
            **Notes**

            - First-time model use may download diffusion, ControlNet, rembg, or Real-ESRGAN weights.
            - CPU mode is supported for debugging, but real enhancement is intended for GPU execution.
            - For deployment preparation details, see `docs/PHASE6_APP.md`.
            """
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
