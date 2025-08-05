import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import PeftModel
from diffusers.utils import load_image

# === Ask user for prompt ===
prompt = input("📝 Enter your prompt: ")

# === CONFIG ===
ref_img_dir = "lora_training/Images"
base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "lora_training/lora_peft_output"
controlnet_model = "lllyasviel/sd-controlnet-canny"
output_dir = "generated_test_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Muscle area coordinates (you can refine)
MUSCLE_AREAS = {
    "quadriceps": (220, 350, 300, 470),
    "hamstrings": (220, 470, 300, 580),
    "biceps": (180, 200, 230, 250),
    "triceps": (230, 200, 280, 250),
    "calves": (220, 580, 300, 670),
    "deltoids": (180, 140, 250, 180),
    "glutes": (200, 480, 300, 550)
}

# === Keyword to Image Matching ===
def match_reference_image(prompt):
    keywords = ["knee", "spine", "arm", "leg", "brain", "lungs", "neck", "shoulder", "foot", "pelvis"]
    for kw in keywords:
        if kw in prompt.lower():
            for file in os.listdir(ref_img_dir):
                if kw in file.lower():
                    return os.path.join(ref_img_dir, file)
    return None

# === Extract target muscle from prompt ===
def extract_target_muscle(prompt):
    prompt_lower = prompt.lower()
    for muscle in MUSCLE_AREAS:
        if muscle in prompt_lower:
            return muscle
    # fallback: check for common patterns
    muscle_synonyms = {
        "thigh": "quadriceps",
        "quads": "quadriceps",
        "calf": "calves",
        "shoulder": "deltoids",
        "butt": "glutes",
        "buttocks": "glutes",
        "upper arm": "biceps",
        "lower arm": "triceps"
    }
    for word, mapped_muscle in muscle_synonyms.items():
        if word in prompt_lower:
            return mapped_muscle
    # fallback for generic mentions
    if "highlight" in prompt_lower or "red muscle" in prompt_lower or "glow" in prompt_lower:
        return "quadriceps"  # default if nothing specific found
    return None


# === Add red mask to reference image ===
def add_red_mask(image_path, muscle_name):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    box = MUSCLE_AREAS[muscle_name]
    draw.rectangle(box, fill=(255, 0, 0, 120))  # semi-transparent red overlay
    temp_path = "temp_masked_image.png"
    image.save(temp_path)
    return temp_path

# === Canny Edge from Image ===
def get_canny_edges(image_path):
    image = load_image(image_path)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edge_rgb)

# === Load pipeline ===
print("🔄 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float32)

print("📦 Loading base model with ControlNet...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
).to("cpu")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

print("🎯 Applying LoRA...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

# === Get reference image ===
ref_image_path = match_reference_image(prompt)
if not ref_image_path:
    raise ValueError("❌ No reference image matched the prompt.")
print(f"📸 Matched reference image: {ref_image_path}")

# === Check and apply muscle mask ===
target_muscle = extract_target_muscle(prompt)
if target_muscle:
    print(f"🔴 Applying red overlay for: {target_muscle}")
    ref_image_path = add_red_mask(ref_image_path, target_muscle)
else:
    print("ℹ️ No muscle highlight found in prompt — using plain reference image.")

# === Generate structure + final image
structure_image = get_canny_edges(ref_image_path)

print("🎨 Generating image...")
result = pipe(
    prompt=prompt,
    negative_prompt="blurry, disfigured, low quality, unrealistic anatomy, text",
    image=structure_image,
    guidance_scale=8.5,
    num_inference_steps=40,
    height=768,
    width=512
)

image = result.images[0]
filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
save_path = os.path.join(output_dir, filename)
image.save(save_path)
print(f"\n✅ Saved generated image: {save_path}")
