import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def get_text_embeddings(texts, model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        raise ValueError("No embeddings generated. Check that your dataset has non-empty text_chunk fields.")

    return np.vstack(all_embeddings)

if __name__ == "__main__":
    texts = []
    with open("data/mimic_cxr_dataset/multimodal_chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text_chunk" in obj and obj["text_chunk"].strip():
                texts.append(obj["text_chunk"])

    print(f"Loaded {len(texts)} text chunks")
    text_embeds = get_text_embeddings(texts)
    print("Text embeddings shape:", text_embeds.shape)

    np.save("text_embeddings.npy", text_embeds)