mkdir -p weights
pip install --upgrade diffusers torch safetensors transformers
python -c "
from diffusers import AutoPipelineForText2Image
AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    cache_dir='./weights',
    torch_dtype='auto'
)
"
