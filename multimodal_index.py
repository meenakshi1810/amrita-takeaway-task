import numpy as np
import faiss

# Load embeddings
text_embeds = np.load("text_embeddings.npy").astype("float32")
image_embeds = np.load("vision_embeddings.npy").astype("float32")

# Normalize
def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

text_embeds = normalize(text_embeds)
image_embeds = normalize(image_embeds)

# Build indices
text_index = faiss.IndexFlatIP(text_embeds.shape[1])
image_index = faiss.IndexFlatIP(image_embeds.shape[1])

text_index.add(text_embeds)
image_index.add(image_embeds)

# Multimodal fusion
min_len = min(len(text_embeds), len(image_embeds))
multimodal_embeds = np.hstack([text_embeds[:min_len], image_embeds[:min_len]])
multimodal_index = faiss.IndexFlatIP(multimodal_embeds.shape[1])
multimodal_index.add(multimodal_embeds)

# Save indices
faiss.write_index(text_index, "text.index")
faiss.write_index(image_index, "vision.index")
faiss.write_index(multimodal_index, "multimodal.index")

print("Indices saved: text.index, image.index, multimodal.index")

