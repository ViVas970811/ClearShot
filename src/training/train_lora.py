"""
LoRA fine-tuning script for Stable Diffusion 1.5 UNet with ControlNet conditioning.
Trains only LoRA adapter weights (~2-4M params) while keeping the base model frozen.
"""

import os
import math
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.dataset import ProductEnhancementDataset


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_models(config: dict, device: str, dtype: torch.dtype):
    """Load and prepare all model components."""
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    # VAE
    vae = AutoencoderKL.from_pretrained(
        model_cfg["base_model"], subfolder="vae", torch_dtype=dtype
    )
    vae.requires_grad_(False)
    vae.to(device)

    # Text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg["base_model"], subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg["base_model"], subfolder="tokenizer"
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_cfg["base_model"], subfolder="scheduler"
    )

    # ControlNet (frozen)
    controlnet = ControlNetModel.from_pretrained(
        model_cfg["controlnet"], torch_dtype=dtype
    )
    controlnet.requires_grad_(False)
    controlnet.to(device)

    # UNet with LoRA
    unet = UNet2DConditionModel.from_pretrained(
        model_cfg["base_model"], subfolder="unet", torch_dtype=dtype
    )
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.to(device)

    if is_xformers_available() and device == "cuda":
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    return vae, text_encoder, tokenizer, noise_scheduler, controlnet, unet


def encode_prompt(tokenizer, text_encoder, prompt: str, device: str):
    """Tokenize and encode a text prompt."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids)[0]
    return encoder_hidden_states


def train(
    config_path: str = "configs/train_config.yaml",
    output_dir: str = "checkpoints",
    resume_from: str = None,
):
    config = load_config(config_path)
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    prompt_cfg = config["prompt"]

    set_seed(train_cfg["seed"])

    accelerator = Accelerator(
        mixed_precision=train_cfg["mixed_precision"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )

    device = accelerator.device
    dtype = torch.float16 if train_cfg["mixed_precision"] == "fp16" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading models...")

    vae, text_encoder, tokenizer, noise_scheduler, controlnet, unet = prepare_models(
        config, str(device), dtype
    )

    # Encode the fixed prompt once
    prompt_embeds = encode_prompt(
        tokenizer, text_encoder, prompt_cfg["template"], str(device)
    )

    # Dataset and dataloader
    train_dataset = ProductEnhancementDataset(
        manifest_path=config["data"]["train_manifest"],
        resolution=train_cfg["resolution"],
        prompt=prompt_cfg["template"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer (only LoRA parameters)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lora_cfg["learning_rate"],
        weight_decay=lora_cfg.get("weight_decay", 0.01),
    )

    # Learning rate scheduler
    num_training_steps = min(
        train_cfg.get("max_train_steps", float("inf")),
        len(train_loader) * train_cfg["num_epochs"] // train_cfg["gradient_accumulation_steps"],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=1e-6
    )

    # Prepare with accelerator
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    loss_history = []

    print(f"Training for {train_cfg['num_epochs']} epochs, {len(train_loader)} steps/epoch")
    print(f"Effective batch size: {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"Max training steps: {num_training_steps}")
    print(f"Saving checkpoints every {train_cfg['save_every_n_steps']} steps to {output_dir}")

    for epoch in range(train_cfg["num_epochs"]):
        unet.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{train_cfg['num_epochs']}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(unet):
                clean = batch["clean"].to(dtype=dtype)
                edge_map = batch["edge_map"].to(dtype=dtype)

                # Encode clean image to latent space
                latents = vae.encode(clean).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long,
                )

                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Expand prompt embeddings to batch size
                encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)

                # ControlNet conditioning
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=edge_map,
                    return_dict=False,
                )

                # UNet prediction with ControlNet residuals
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # MSE loss on predicted noise
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                loss_val = loss.detach().item()
                epoch_loss += loss_val
                num_batches += 1
                loss_history.append(loss_val)

                progress_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                # Log periodically
                if global_step % config["logging"]["log_every_n_steps"] == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Step {global_step}: loss={loss_val:.4f}, avg_loss={avg_loss:.4f}")

                # Save checkpoint
                if global_step % train_cfg["save_every_n_steps"] == 0:
                    ckpt_path = output_path / f"checkpoint-{global_step}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_pretrained(str(ckpt_path))
                    print(f"  Checkpoint saved: {ckpt_path}")

                # Check max steps
                if global_step >= num_training_steps:
                    break

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

        if global_step >= num_training_steps:
            break

    # Save final weights
    final_path = output_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    unwrapped.save_pretrained(str(final_path))
    print(f"Final LoRA weights saved to {final_path}")

    # Save loss history
    import json
    with open(output_path / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    return loss_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    
