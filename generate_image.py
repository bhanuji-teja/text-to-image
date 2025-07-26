import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from datetime import datetime
import os

# Paths
base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "lora_training/lora_peft_output"  # PEFT output folder
output_dir = "generated_images_peft"
os.makedirs(output_dir, exist_ok=True)

# Load base model
print("Loading Stable Diffusion model...")
#pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32).to("cpu")

# Apply LoRA using PEFT
print("Applying LoRA adapter...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

# Enable attention optimization if available
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Warning: xformers not enabled: {e}")

# User input prompt
prompt = input("Enter your prompt: ")
negative_prompt = (
    "blurry, low quality, extra limbs, disfigured, text, watermark, oversaturated, cropped, cartoon, stylized, clothes, wristbands, extreme red lighting, full-body muscles, unrealistic anatomy, bodybuilding"
)

# Settings
num_images = 3
guidance_scale = 8.5
num_inference_steps = 50
height = 768
width = 512

# Generate images
print(f"Generating {num_images} images with LoRA applied...")
for i in range(num_images):
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    image = result.images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"peft_image_{i+1}_{timestamp}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print("✅ All images generated successfully!")