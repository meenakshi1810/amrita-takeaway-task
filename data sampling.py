import pandas as pd
import io
from PIL import Image
import os

# Path to parquet inside data/
parquet_path = "data/mimic_cxr_dataset.parquet"

# Load parquet file
df = pd.read_parquet(parquet_path)

# Randomly sample 500 rows
sample_df = df.sample(n=500, random_state=42).reset_index(drop=True)

# Define base directory for outputs
base_dir = "data/mimic_cxr_dataset"
images_dir = os.path.join(base_dir, "images")

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)

filepaths = []
findings_list = []
impressions_list = []

# Iterate and save images
for idx, row in sample_df.iterrows():
    img_bytes = row["image"]
    img = Image.open(io.BytesIO(img_bytes))

    # Save image with unique filename inside base_dir/images
    filepath = os.path.join(images_dir, f"img_{idx:05d}.png")
    img.save(filepath)

    filepaths.append(filepath)
    findings_list.append(row["findings"])
    impressions_list.append(row["impression"])

# Build CSV inside base_dir
out_df = pd.DataFrame({
    "filepath": filepaths,
    "findings": findings_list,
    "impression": impressions_list
})

csv_path = os.path.join(base_dir, "dataset.csv")
out_df.to_csv(csv_path, index=False)

print(f"✅ Saved 500 images to '{images_dir}' and metadata to '{csv_path}'")
