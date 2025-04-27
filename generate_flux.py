import torch
from diffusers import FluxPipeline

# Use MPS for Apple Silicon if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16 if device=="mps" else torch.bfloat16
).to(device)

prompt = input("Enter your prompt: ")

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512
).images[0]

image.save("flux-dev.png")
print("Image saved to flux-dev.png")