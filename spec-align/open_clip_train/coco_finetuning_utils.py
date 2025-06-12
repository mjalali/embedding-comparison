


# class COCODataset(Dataset):
#     def __init__(self, img_dir, ann_file, transform=None, tokenizer=None):
#         # Load COCO annotations
#         self.coco = COCO(ann_file)
#         self.img_dir = img_dir
#         self.ids = list(self.coco.imgs.keys())
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
#         self.tokenizer = tokenizer
#
#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#         caption = anns[0]['caption']
#
#         # Load image
#         img_data = self.coco.loadImgs(img_id)[0]
#         img_path = os.path.join(self.img_dir, img_data['file_name'])
#         image = Image.open(img_path).convert('RGB')
#
#         if self.transform is not None:
#             image = self.transform(image)
#
#         # Tokenize caption if tokenizer provided
#         if self.tokenizer is not None:
#             caption = self.tokenizer([caption])[0]
#
#         return image, caption
#
#     def __len__(self):
#         return len(self.ids)

import os
import pandas as pd
from pycocotools.coco import COCO

# Define paths to images and annotations
img_dir = "/home/student/Documents/evaluation/t2i/COCO/dataset_2017/val2017"
ann_file = "/home/student/Documents/evaluation/t2i/COCO/dataset_2017/annotations/captions_val2017.json"

# # Load COCO annotation file
# coco = COCO(ann_file)
#
# # Get all image IDs
# img_ids = coco.getImgIds()
#
# # Initialize dictionary to store file paths and captions
# future_df = {"filepath": [], "title": []}
#
# # Iterate through images and gather captions
# for img_id in img_ids:
#     # Load image information
#     img_info = coco.loadImgs(img_id)[0]
#     img_path = os.path.join(img_dir, img_info["file_name"])
#
#     # Load all annotations (captions) for the given image
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)
#
#     for ann in anns:
#         future_df["filepath"].append(img_path)
#         future_df["title"].append(ann["caption"])
#
# # Convert to DataFrame and save as CSV
# df = pd.DataFrame.from_dict(future_df)
# output_csv_path = os.path.join(img_dir.replace('val2017', ''), "val2017.csv")
# df.to_csv(output_csv_path, index=False, sep="\t")
#
# print(f"CSV file saved at {output_csv_path}")

python train.py \
  --train-data /home/student/Documents/evaluation/t2i/COCO/dataset_2017/train2017.csv \
  --val-data /home/student/Documents/evaluation/t2i/COCO/dataset_2017/val2017.csv \
  --dataset-type csv \
  --csv-img-key filepath \
  --csv-caption-key title \
  --csv-separator "\t" \
  --model ViT-B/32 \
  --pretrained openai \
  --batch-size 64 \
  --epochs 10 \
  --lr 5e-5 \
  --wd 0.2 \
  --warmup 1000 \
  --device cuda \
  --workers 8 \
  --precision amp \
  --logs ./logs/ \
  --save-frequency 1 \
  --val-frequency 1 \
  --epochs 1


torchrun --nproc_per_node 2 -m open_clip_train.main
--train-data="/home/student/Documents/evaluation/t2i/COCO/dataset_2017/train2017_fixed.csv"
--val-data="/home/student/Documents/evaluation/t2i/COCO/dataset_2017/val2017_fixed.csv"
--dataset-type
csv
--csv-img-key
filepath
--csv-caption-key
title
--csv-separator
","
--model
ViT-B-32
--pretrained
openai
--batch-size
64
--epochs
2
--lr
5e-5
--wd
0.2
--warmup
1000

import os
import pandas as pd

# Define the path to your CSV file
csv_path = "/home/student/Documents/evaluation/t2i/COCO/dataset_2017/val2017.csv"

# Load the CSV file
try:
    df = pd.read_csv(csv_path, sep="\t")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}.")
    exit()

# Initialize lists to keep track of errors
missing_files = []
invalid_char_files = []
invalid_format_files = []

# Supported image extensions
supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Check each file path in the CSV
for idx, filepath in enumerate(df["filepath"]):
    # Strip leading and trailing whitespaces or quotation marks
    stripped_path = filepath.strip().strip('"').strip("'")
    
    # Check if there were invalid leading/trailing characters
    if stripped_path != filepath:
        invalid_char_files.append((idx, filepath))

    # Check if the file exists
    if not os.path.exists(stripped_path):
        missing_files.append((idx, stripped_path))

    # Check if the file format is supported
    _, ext = os.path.splitext(stripped_path)
    if ext.lower() not in supported_extensions:
        invalid_format_files.append((idx, stripped_path))

# Print results
if missing_files:
    print(f"Missing Files ({len(missing_files)}):")
    for idx, path in missing_files[:10]:  # Display up to 10 missing files
        print(f"  Row {idx}: {path}")
else:
    print("All files exist.")

if invalid_char_files:
    print(f"\nFiles with Invalid Leading/Trailing Characters ({len(invalid_char_files)}):")
    for idx, path in invalid_char_files[:10]:  # Display up to 10 invalid character files
        print(f"  Row {idx}: {path}")
else:
    print("No files with invalid leading/trailing characters.")

if invalid_format_files:
    print(f"\nFiles with Unsupported Formats ({len(invalid_format_files)}):")
    for idx, path in invalid_format_files[:10]:  # Display up to 10 invalid format files
        print(f"  Row {idx}: {path}")
else:
    print("All files have supported formats.")

# Summary
print("\nSummary:")
print(f"Total Missing Files: {len(missing_files)}")
print(f"Total Files with Invalid Characters: {len(invalid_char_files)}")
print(f"Total Files with Unsupported Formats: {len(invalid_format_files)}")

# Optionally, save problematic rows to a CSV file for further inspection
if missing_files or invalid_char_files or invalid_format_files:
    error_df = pd.DataFrame({
        "Row Index": [idx for idx, _ in (missing_files + invalid_char_files + invalid_format_files)],
        "Filepath": [path for _, path in (missing_files + invalid_char_files + invalid_format_files)]
    })
    error_df.to_csv("invalid_file_paths.csv", index=False)
    print("\nSaved details of problematic file paths to 'invalid_file_paths.csv'.")



import pandas as pd

# Load CSV file
csv_path = "/home/student/Documents/evaluation/t2i/COCO/dataset_2017/train2017.csv"
df = pd.read_csv(csv_path, sep="\t")

# Look for invalid characters in the file paths
invalid_rows = df[df["filepath"].str.contains(r'["\'\\]')]

if not invalid_rows.empty:
    print("Found invalid file paths:")
    print(invalid_rows)
else:
    print("No invalid file paths found.")
