import streamlit as st
import numpy as np
import faiss
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Load FAISS indexes ---
text_index = faiss.read_index("text.index")
image_index = faiss.read_index("image.index")
multimodal_index = faiss.read_index("multimodal.index")

# --- Load metadata ---
rows = [json.loads(line) for line in open(
    "data/mimic_cxr_dataset/multimodal_chunks.jsonl", "r", encoding="utf-8"
)]

# --- Load encoders ---
# MiniLM for text-only retrieval
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# CLIP for both text→image and image→image retrieval
clip_encoder = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def encode_text(query: str) -> np.ndarray:
    vec = text_encoder.encode([query], normalize_embeddings=True)
    return vec.astype("float32")

def encode_clip_text(query: str) -> np.ndarray:
    vec = clip_encoder.encode([query], normalize_embeddings=True)
    return vec.astype("float32")

def encode_image(img: Image.Image) -> np.ndarray:
    vec = clip_encoder.encode([img], normalize_embeddings=True)
    return vec.astype("float32")

# --- Local Flan-T5 model (CPU-friendly) ---
local_dir = "models/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir=local_dir)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

def search(index, query_vec, k=5):
    qvec = np.array(query_vec).astype("float32")
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)
    assert qvec.shape[1] == index.d, f"Query dim {qvec.shape[1]} != index dim {index.d}"
    D, I = index.search(qvec, k)
    return sorted(zip(I[0], D[0]), key=lambda x: -x[1])

def build_prompt(query, evidence_lines, mode):
    if mode == "Image→Image":
        return f"""Question: {query}

Evidence (ordered by relevance):
{chr(10).join(evidence_lines)}

Write a cohesive summary describing similarities between the retrieved images.
If evidence is insufficient, reply exactly: "Unanswerable".
Answer:"""
    else:
        return f"""Question: {query}

Evidence (ordered by relevance):
{chr(10).join(evidence_lines)}

Write a cohesive clinical summary that integrates the evidence into a single answer.
If evidence is insufficient, reply exactly: "Unanswerable".
Answer:"""

# --- Streamlit UI ---
st.title("Multimodal Retrieval Demo")

mode = st.selectbox(
    "Choose retrieval mode",
    ["Text→Text", "Text→Image", "Image→Text", "Image→Image", "Multimodal→Multimodal"]
)
query_text = st.text_input("Enter query text")
uploaded_image = st.file_uploader("Upload query image", type=["jpg","png"])
k = st.slider("Top-k results", 1, 10, 5)

if st.button("Run Retrieval"):
    text_hits, image_hits, multimodal_hits = [], [], []

    if mode == "Text→Text" and query_text:
        qvec = encode_text(query_text)
        text_hits = search(text_index, qvec, k)

    elif mode == "Text→Image" and query_text:
        qvec = encode_clip_text(query_text)
        image_hits = search(image_index, qvec, k)

    elif mode == "Image→Text" and uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        qvec = encode_image(img)
        text_hits = search(text_index, qvec, k)

    elif mode == "Image→Image" and uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        qvec = encode_image(img)
        image_hits = search(image_index, qvec, k)

    elif mode == "Multimodal→Multimodal" and query_text and uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        text_vec = encode_text(query_text)
        image_vec = encode_image(img)
        multimodal_vec = np.hstack([text_vec, image_vec])
        multimodal_vec = normalize(multimodal_vec)
        multimodal_hits = search(multimodal_index, multimodal_vec, k)

    # --- Answer generation ---
    evidence_lines = []
    for idx, score in text_hits:
        evidence_lines.append(f"[doc_id={idx}] {rows[idx]['text_chunk'][:120]} (score={score:.4f})")
    for idx, score in image_hits:
        evidence_lines.append(f"[image_id={idx}] {rows[idx]['filepath']} (score={score:.4f})")
    for idx, score in multimodal_hits:
        evidence_lines.append(f"[mm_id={idx}] {rows[idx]['text_chunk'][:120]} (score={score:.4f})")

    if evidence_lines:
        prompt = build_prompt(query_text or "Image query", evidence_lines, mode)
        out = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        answer = out.strip()

        st.subheader("Generated Answer")
        st.write(answer)

        with st.expander("Show retrieved evidence"):
            for line in evidence_lines:
                st.write(line)
    else:
        st.warning("No hits retrieved. Answer: Unanswerable")
