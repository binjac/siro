from diffusers import StableDiffusionXLPipeline
import torch

# Path to your model
model_path = "models/epicrealism/epicrealismXL_vxviLastfameRealism.safetensors"

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# Load the pipeline from safetensors
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=dtype,
)
pipe = pipe.to(device)

# Prompt
prompt = input("Enter your prompt: ")

# Generate image
image = pipe(
    prompt,
    height=1024,     # SDXL default, can use 768 or 512 for less memory
    width=1024,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

out_path = "epicrealism_xl_out.png"
image.save(out_path)
print(f"Image saved to {out_path}")