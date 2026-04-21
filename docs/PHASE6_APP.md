# Phase 6 - Gradio App and Deployment Prep

This document describes the Phase 6 application work for ClearShot.

## What Phase 6 Implements

Phase 6 adds a Gradio-based application layer on top of the existing
`ClearShotPipeline` and prepares the project for local containerization and
future Hugging Face Spaces deployment.

Implemented now:

- `app/gradio_app.py` as the main Gradio UI
- `app.py` as a Spaces-friendly entry point
- single-image enhancement workflow
- batch processing workflow with ZIP export
- configurable runtime paths for inference config and LoRA weights
- user controls for:
  - enhancement strength
  - guidance scale
  - inference steps
  - background choice
  - super-resolution toggle
  - device choice
- before/after display
- intermediate stage display
- downloadable outputs
- Dockerfile and `docker-compose.yml` for local deployment preparation

Prepared but not claimed as fully deployed:

- Hugging Face Spaces-ready entry structure
- deployment guidance for hosted execution

Not claimed here:

- a verified hosted Spaces deployment URL
- confirmed production deployment success on Hugging Face infrastructure

## Local Run

From the repository root:

```bash
python app.py
```

Or:

```bash
python app/gradio_app.py
```

Default app URL:

```text
http://127.0.0.1:7860
```

## Runtime Environment Variables

The app supports these optional environment variables:

```bash
CLEARSHOT_CONFIG_PATH=configs/inference_config.yaml
CLEARSHOT_LORA_PATH=ClearShot_checkpoints/final
CLEARSHOT_DEVICE=auto
PORT=7860
```

`CLEARSHOT_DEVICE` may be `auto`, `cuda`, or `cpu`.

## Required Assets

For full ClearShot behavior, the following should be available:

```text
configs/inference_config.yaml
ClearShot_checkpoints/final/adapter_config.json
ClearShot_checkpoints/final/adapter_model.safetensors
```

The app can still start if LoRA weights are missing; in that case it falls back
to the base SD + ControlNet path and surfaces a user-facing note.

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t clearshot-app .
docker run -p 7860:7860 clearshot-app
```

## Hugging Face Spaces Preparation

The repository now includes a root-level `app.py`, which is the conventional
entry point for a Gradio Space.

To prepare a real Spaces deployment later, the following still need to be done
in the actual Hugging Face environment:

1. Create the Space.
2. Upload the repository or selected deployment files.
3. Confirm dependency installation in the hosted environment.
4. Verify model downloads, mounted assets, and runtime memory behavior.
5. Test the UI end to end in the live Space.

Until that real hosted test happens, the project should be described as
**deployment-ready** rather than **fully deployed**.

## Practical Notes

- GPU is strongly recommended for actual enhancement workloads.
- CPU mode is acceptable for debugging but will be slow.
- First use may trigger downloads for diffusion, ControlNet, rembg, or
  Real-ESRGAN weights.
- The app reuses the existing `ClearShotPipeline`; it does not duplicate model
  logic.
