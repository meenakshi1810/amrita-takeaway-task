import pandas as pd
import io
from PIL import Image
import os
import json

def chunk_text(text, max_len=128):
    if not isinstance(text, str) or text.strip() == "":
        return []
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current.split()) + len(s.split()) <= max_len:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def ingest_parquet(parquet_path, base_dir, sample_size=500):
    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Sample rows
    sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Prepare folders
    images_dir = os.path.join(base_dir, "mimic_cxr_dataset/images")
    os.makedirs(images_dir, exist_ok=True)

    filepaths, findings_list, impressions_list = [], [], []

    # Save images + collect metadata
    for idx, row in sample_df.iterrows():
        img_bytes = row["image"]
        img = Image.open(io.BytesIO(img_bytes))
        filepath = os.path.join(images_dir, f"img_{idx:05d}.png")
        img.save(filepath)

        filepaths.append(filepath)
        findings_list.append(row["findings"])
        impressions_list.append(row["impression"] if pd.notnull(row["impression"]) else "")

    # Build CSV
    csv_path = os.path.join(base_dir, "mimic_cxr_dataset/dataset.csv")
    out_df = pd.DataFrame({
        "filepath": filepaths,
        "findings": findings_list,
        "impression": impressions_list
    })
    out_df.to_csv(csv_path, index=False)

    # Build chunked records
    records = []
    for idx, row in out_df.iterrows():
        for chunk in chunk_text(row["findings"]):
            records.append({
                "id": f"{idx}_findings_{hash(chunk)%10000}",
                "filepath": row["filepath"],
                "section": "findings",
                "text_chunk": chunk
            })
        for chunk in chunk_text(row["impression"]):
            records.append({
                "id": f"{idx}_impression_{hash(chunk)%10000}",
                "filepath": row["filepath"],
                "section": "impression",
                "text_chunk": chunk
            })

    # Save chunked dataset
    jsonl_path = os.path.join(base_dir, "mimic_cxr_dataset/multimodal_chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Saved {sample_size} images to '{images_dir}'")
    print(f"💾 Metadata CSV: {csv_path}")
    print(f"💾 Chunked JSONL: {jsonl_path}")

if __name__ == "__main__":
    base_dir = "data"
    parquet_path = "data/mimic_cxr_dataset.parquet"
    ingest_parquet(parquet_path, base_dir, sample_size=500)
