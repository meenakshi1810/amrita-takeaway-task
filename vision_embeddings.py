import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_image_embeddings(image_paths, model_name="openai/clip-vit-base-patch32", batch_size=8):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        raise ValueError("No embeddings generated. Check that your dataset has valid filepaths.")

    return np.vstack(all_embeddings)

if __name__ == "__main__":
    image_paths = []
    with open("data/mimic_cxr_dataset/multimodal_chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "filepath" in obj and obj["filepath"].strip():
                image_paths.append(obj["filepath"])

    print(f"Loaded {len(image_paths)} image paths")
    image_embeds = get_image_embeddings(image_paths)
    print("Image embeddings shape:", image_embeds.shape)

    np.save("image_embeddings.npy", image_embeds)
