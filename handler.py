def load_model(self):
    """Load the SDXL Turbo model from local weights folder."""
    print("🚀 Loading SDXL Turbo model...")

    MODEL_DIR = "/weights"  # pre-downloaded model location

    try:
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_DIR,             # <- load from local folder
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True   # <- no network download
        )

        if torch.cuda.is_available():
            self.pipe.to("cuda")
            print("✅ Model loaded successfully on GPU!")
        else:
            print("⚠️ GPU not available, running on CPU")

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load SDXL Turbo model: {str(e)}")
