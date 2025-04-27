from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

model_path = "models/epicrealism/epicrealismXL_vxviLastfameRealism.safetensors"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=dtype,
)
pipe = pipe.to(device)

# Set scheduler to DPM++ with Karras config if available
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

prompt = (
    "35mm film sample image, Close up shot of a 33 years old woman face, "
    "harsh low-key lighting, blinds shadows on her face, high contrast, Grainy, Lomography, "
    "lomoChrome, Analog Film, Kodak Ektar 100, Cinematic composition, Rule of thirds"
)

negative_prompt = (
    "Camera, lens, digital photography, low quality , bad quality , film frame, frame, painting, "
    "drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, ugly, deformed, "
    "noisy, blurry, distorted, camera, lens, black and white, b&w, watermark, epiCPhoto-neg"
)

seed = 785777314
generator = torch.Generator(device=device).manual_seed(seed)

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    height=1216,
    width=832,
    num_inference_steps=20,
    guidance_scale=5,
    generator=generator
).images[0]

out_path = "epicrealism_xl_film_sample.png"
image.save(out_path)
print(f"Image saved to {out_path}")