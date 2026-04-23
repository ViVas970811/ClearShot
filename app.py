"""Hugging Face Spaces-friendly entry point for ClearShot."""

import sys
import subprocess

# HACK: Install SR dependencies at runtime to bypass HF Spaces build isolation bug
try:
    import basicsr
except ImportError:
    print("Installing basicsr and realesrgan dynamically to bypass torch dependency collision...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools<70"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "basicsr", "facexlib", "gfpgan", "realesrgan"])

from app.gradio_app import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
