# amrita-takeaway-task

# 🩺 Multimodal Retrieval and Evaluation Toolkit

This project implements a multimodal retrieval system for clinical data, combining text and image embeddings to support cross-modal search and answer generation. It includes ingestion, indexing, retrieval, and evaluation components built around FAISS, SentenceTransformers, and Flan-T5.

## 📁 File Overview (Execution Order)

| Step | File | Description |
|------|------|-------------|
| 1️ | data sampling.py | Samples 500 random instances from the MIMIC-CXR dataset for downstream embedding and indexing. |
| 2️ | multimodal_ingest.py | Prepares metadata and chunked text-image pairs for embedding. |
| 3️ | text_embeddings.py | Encodes text chunks using MiniLM and saves them for FAISS indexing. |
| 4️ | vision_embeddings.py | Encodes images using CLIP and saves image embeddings for indexing. |
| 5️ | multimodal_index.py | Builds FAISS indexes for text, image, and multimodal embeddings. |
| 6️ | retrieve.py | Streamlit app for querying across five modes: Text→Text, Text→Image, Image→Text, Image→Image, and Multimodal→Multimodal. |
| 7 | embedding evaluation | contains python files which were used to evaluate 3 text embedding and 2 image embedding models |
| 8 | evaluation.py | contains evaluation suite for retreival system

##  Retrieval Modes Supported

- **Text→Text**: Retrieve relevant report chunks from a text query.
- **Text→Image**: Retrieve similar CXRs using CLIP text encoder.
- **Image→Text**: Retrieve relevant report chunks from an uploaded image.
- **Image→Image**: Retrieve visually similar CXRs.
- **Multimodal→Multimodal**: Retrieve joint text-image chunks using concatenated embeddings.

##  Models Used

- **Text Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
- **Image Encoder**: `sentence-transformers/clip-ViT-B-32`
- **Answer Generator**: `google/flan-t5-base` (instruction-tuned, CPU-friendly)
