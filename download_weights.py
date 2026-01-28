# download_weights.py
from diffusers import AutoPipelineForText2Image
import torch
import os

MODEL_DIR = "/weights"
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Downloading SDXL Turbo model...")

try:
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=MODEL_DIR,   # <- save here
        local_files_only=False
    )
    print(f"✅ Model downloaded successfully to {MODEL_DIR}")
except Exception as e:
    print(f"❌ Error downloading model: {str(e)}")
    raise
