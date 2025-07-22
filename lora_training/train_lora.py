import os
import torch
from torchvision import transforms
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# ------------------------------
# Config
# ------------------------------
model_id = "runwayml/stable-diffusion-v1-5"
image_folder = "Images"
output_dir = "lora_peft_output"
prompt_text = "A glowing 3D medical-style human anatomical scan"
rank = 4
num_epochs = 10
batch_size = 1
learning_rate = 1e-4

accelerator = Accelerator()
device = accelerator.device

# ------------------------------
# Image Transform
# ------------------------------
image_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ------------------------------
# Dataset
# ------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, folder, prompt):
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.prompt = prompt
        print(f"Found {len(self.image_paths)} images in {folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": image_transforms(image),
            "prompt": self.prompt
        }

# ------------------------------
# Load Pipeline
# ------------------------------
print("Loading SD v1.5 pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

unet = pipe.unet
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

# ------------------------------
# Apply LoRA using PEFT
# ------------------------------
print("Applying PEFT LoRA...")
lora_config = LoraConfig(
    r=rank,
    lora_alpha=rank,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Attention projections
    lora_dropout=0.0,
    bias="none",
    task_type="UNET"
)

unet = get_peft_model(unet, lora_config)

# Collect LoRA params
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print(f"Trainable LoRA params: {sum(p.numel() for p in trainable_params)}")

optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

# ------------------------------
# Data Loader
# ------------------------------
dataset = ImageOnlyDataset(image_folder, prompt_text)
if len(dataset) == 0:
    raise ValueError("No images found in dataset path!")

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
vae.to(device)
text_encoder.to(device)

# ------------------------------
# Training Loop
# ------------------------------
unet.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    pbar = tqdm(train_dataloader, desc="Training", disable=not accelerator.is_local_main_process)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        prompt = batch["prompt"]

        # Encode images to latents
        latents = vae.encode(pixel_values * 2 - 1).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Encode text
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({"loss": loss.item()})

# ------------------------------
# Save LoRA Weights
# ------------------------------
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"✅ LoRA weights saved in PEFT format at: {output_dir}")
