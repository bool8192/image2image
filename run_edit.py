import torch
import json
import uuid
import os
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from PIL import Image

torch.set_num_threads(24)  #это у меня столько. Надо поменять на ваше число потоков процессора.

INPUT_DIR = "photos"
OUTPUT_DIR = "photos_edited"
JSON_PATH = "results.json"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_id = "diffusers/sdxl-instructpix2pix-768"
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    model_id, 
    use_safetensors=True
)

pipe.to("cpu") # Модель не влезает в мои 6 гб видеопамяти, если перекидывать, то будет просто все черное. 

results_metadata = []
prompt = "Increase color saturation and brightness moderately without overexposure. Add a small yellow five-pointed star in the bottom-left corner with a slight margin from the edges."

for i, filename in enumerate(os.listdir(INPUT_DIR)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        source_id = filename
        final_id = f"res_{uuid.uuid4().hex[:8]}"
        
        input_path = os.path.join(INPUT_DIR, filename)
        image = Image.open(input_path).convert("RGB")

        print(f"Старт {i}: {filename}")
        edited_image = pipe(
           prompt,
           image=image,
           num_inference_steps=20,
           image_guidance_scale=1.5,
           guidance_scale=7.5,
           height=768,                  
           width=768
        ).images[0]

        out_name = f"{final_id}.jpg"
        edited_image.save(os.path.join(OUTPUT_DIR, out_name))
        
        results_metadata.append({
            "id_исходного_изображения": source_id,
            "промпт_редактирования": prompt,
            "id_финального_изображения": final_id
        })
        print(f"Готово: {out_name}")

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results_metadata, f, indent=4, ensure_ascii=False)