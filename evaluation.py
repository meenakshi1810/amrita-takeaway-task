# eval.py
import json
import time
import numpy as np
import faiss

TEXT_EMB_PATH = "text_embeddings.npy"
IMAGE_EMB_PATH = "vision_embeddings.npy"
JSONL_PATH = "data/mimic_cxr_dataset/multimodal_chunks.jsonl"

TEXT_INDEX_PATH = "text.index"
IMAGE_INDEX_PATH = "image.index"
MULTIMODAL_INDEX_PATH = "multimodal.index"

TOP_K = 5

# -------------------------------
# Load data
# -------------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

rows = load_jsonl(JSONL_PATH)
text_embeds = np.load(TEXT_EMB_PATH).astype("float32")
image_embeds = np.load(IMAGE_EMB_PATH).astype("float32")

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

text_embeds = normalize(text_embeds)
image_embeds = normalize(image_embeds)

# Build stores
text_store = [r for r in rows if "text_chunk" in r and r["text_chunk"].strip()]
seen = set()
image_store = []
for r in rows:
    fp = r.get("filepath")
    if fp and fp not in seen:
        seen.add(fp)
        image_store.append({"filepath": fp})

# -------------------------------
# Load indices (already built)
# -------------------------------
text_index = faiss.read_index(TEXT_INDEX_PATH)
image_index = faiss.read_index(IMAGE_INDEX_PATH)
multimodal_index = faiss.read_index(MULTIMODAL_INDEX_PATH)


# Recreate fused embeddings for queries (aligned by min_len)
min_len = min(len(text_embeds), len(image_embeds))
multimodal_embeds = np.hstack([text_embeds[:min_len], image_embeds[:min_len]])

# -------------------------------
# Grounding helpers
# -------------------------------
def ground_text(idx, score):
    meta = text_store[idx]
    return {
        "section": meta.get("section"),
        "text_chunk": meta.get("text_chunk", "")[:140] + "...",
        "filepath": meta.get("filepath"),
        "score": float(score),
    }

def ground_image(idx, score):
    meta = image_store[idx]
    return {
        "filepath": meta.get("filepath"),
        "score": float(score),
    }

# -------------------------------
# Metrics
# -------------------------------
def precision_at_k_text_section(index, embeddings, k=5):
    labels = [t.get("section") for t in text_store]
    D, I = index.search(embeddings, k)
    correct = 0
    total = embeddings.shape[0] * k
    for qi in range(I.shape[0]):
        qlab = labels[qi]
        for j in range(k):
            if labels[I[qi, j]] == qlab:
                correct += 1
    return correct / total

def avg_latency(index, embeddings, k=5, batch=256):
    start = time.time()
    index.search(embeddings[:batch], k)
    end = time.time()
    return (end - start) / batch

def index_memory_estimate(num_vecs, dim, dtype_bytes=4):
    # rough footprint: vectors only
    return num_vecs * dim * dtype_bytes / (1024**2)

# -------------------------------
# Run evaluation
# -------------------------------
if __name__ == "__main__":
    print("=== Qualitative dumps ===")
    q_idx = 0

    # Text→Text
    D_tt, I_tt = text_index.search(text_embeds[q_idx:q_idx+1], TOP_K)
    print("\nText→Text:")
    for idx, score in zip(I_tt[0], D_tt[0]):
        print(ground_text(idx, score))

    # Text→Image (via linked filepaths from text hits)
    print("\nText→Image (via linked reports):")
    for idx, score in zip(I_tt[0], D_tt[0]):
        print(ground_image(idx, score))

    # Image→Text (direct image index, then ground text by closest text chunks)
    D_it, I_it = image_index.search(image_embeds[q_idx:q_idx+1], TOP_K)
    print("\nImage→Text (via linked metadata):")
    # We ground by nearest text chunks for each image neighbor:
    D_t_for_img, I_t_for_img = text_index.search(text_embeds[I_tt[0][:TOP_K]], TOP_K)  # alt: use a mapping if available
    # Simpler: just show text chunks for the same filepaths where available
    for idx, score in zip(I_it[0], D_it[0]):
        # find any matching text_store rows for this filepath
        fp = image_store[idx]["filepath"]
        candidates = [t for t in text_store if t.get("filepath") == fp]
        if candidates:
            tmeta = candidates[0]
            print({
                "section": tmeta.get("section"),
                "text_chunk": tmeta.get("text_chunk", "")[:140] + "...",
                "filepath": fp,
                "score": float(score),
            })
        else:
            print({"filepath": fp, "score": float(score)})

    # Multimodal→Multimodal
    D_mm, I_mm = multimodal_index.search(multimodal_embeds[q_idx:q_idx+1], TOP_K)
    print("\nMultimodal→Multimodal:")
    for idx, score in zip(I_mm[0], D_mm[0]):
        print(ground_image(idx, score))

    print("\n=== Quantitative metrics ===")
    p5 = precision_at_k_text_section(text_index, text_embeds, k=5)
    print(f"Precision@5 (Text→Text by section): {p5:.4f}")

    lat_text = avg_latency(text_index, text_embeds, k=TOP_K)
    lat_image = avg_latency(image_index, image_embeds, k=TOP_K)
    lat_mm = avg_latency(multimodal_index, multimodal_embeds, k=TOP_K)
    print(f"Avg latency/query: Text {lat_text:.6f}s | Image {lat_image:.6f}s | Multimodal {lat_mm:.6f}s")

    print("\n=== Memory (rough vector footprint) ===")
    print(f"Text index ntotal={text_index.ntotal}, Image ntotal={image_index.ntotal}, Multimodal ntotal={multimodal_index.ntotal}")
    print(f"Text ~{index_memory_estimate(text_index.ntotal, text_embeds.shape[1]):.2f} MB, "
          f"Image ~{index_memory_estimate(image_index.ntotal, image_embeds.shape[1]):.2f} MB, "
          f"Multimodal ~{index_memory_estimate(multimodal_index.ntotal, multimodal_embeds.shape[1])::.2f} MB")
