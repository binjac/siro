import torch
from diffusers import StableDiffusionPipeline

# Use MPS for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

def main():
    prompt = input("Enter your prompt: ")
    print("Generating image, please wait...")

    with torch.autocast("mps"):
        image = pipe(prompt, guidance_scale=7.5, num_inference_steps=25).images[0]
    
    output_path = "output.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()