import os
import base64
from io import BytesIO

import torch
import runpod
from diffusers import FluxPipeline
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# ---- Load model ONCE at container startup ----
print("🚀 Loading FLUX model...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=os.environ["HF_TOKEN"]  # REQUIRED
).to("cuda")

print("✅ FLUX model loaded and ready!")

# ---- RunPod handler ----
def handler(event):
    """
    Expected input:
    {
        "prompt": "A cinematic cyberpunk city at night",
        "steps": 30,
        "seed": 42
    }
    """

    input_data = event.get("input", {})

    prompt = input_data.get("prompt", "A futuristic city")
    steps = input_data.get("steps", 30)
    seed = input_data.get("seed", None)

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

    # Convert image → base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": img_base64
    }

# ---- Start RunPod serverless ----
runpod.serverless.start({"handler": handler})
