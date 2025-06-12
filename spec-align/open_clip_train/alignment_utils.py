import os

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO


def cosine_kernel(x, y=None, batchsize=256, normalize=True, device="cuda"):
    '''
    Calculate the cosine similarity kernel matrix. The shape of x and y should be equal except for the batch dimension.

    x:
        Input tensor, dim: [batch, dims]
    y:
        Input tensor, dim: [batch, dims]. If y is `None`, then y = x and it will compute cosine similarity k(x, x).
    batchsize:
        Batchify the formation of the kernel matrix, trade time for memory.
        batchsize should be smaller than the length of data.

    return:
        Scalar: Mean of cosine similarity values
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        y = x if y is None else torch.from_numpy(y).to(device)
    else:
        x = x.to(device)
        y = x if y is None else y.to(device)

    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx * batchsize:min((batchidx + 1) * batchsize, y.shape[0])]

        # Normalize x and y_slice
        x_norm = x / x.norm(dim=1, keepdim=True)
        y_norm = y_slice / y_slice.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        res = torch.mm(x_norm, y_norm.T)

        total_res = torch.hstack([total_res, res])

        del res, y_slice

    if normalize is True:
        total_res = total_res / (x.shape[0] * y.shape[0])

    return total_res




# Define paths to images and annotations
img_dir = "/home/student/Documents/evaluation/t2i/COCO/SDXL/coco_fake_images"
ann_file = "/home/student/Documents/evaluation/t2i/COCO/dataset_2017/annotations/captions_train2017.json"
captions_file = "/home/student/Documents/evaluation/t2i/COCO/captions.txt"

# Load COCO annotation file
coco = COCO(ann_file)

# Get all image IDs
img_ids = coco.getImgIds()

# Initialize dictionary to store file paths and captions
future_df = {"filepath": [], "title": []}

# Iterate through images and gather captions
for img_id in img_ids:
    # Load image information
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info["file_name"])

    # Verify if the image file exists
    if not os.path.exists(img_path):
        print(f"Warning: Image file does not exist - {img_path}")
        continue

    # Load all annotations (captions) for the given image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        future_df["filepath"].append(img_path)
        future_df["title"].append(ann["caption"])

# Convert to DataFrame and save as CSV in the parent directory of img_dir
parent_dir = os.path.dirname(img_dir)  # Get the parent directory of img_dir
output_csv_path = os.path.join(parent_dir, "train2017_fixed_comma.csv")
df = pd.DataFrame.from_dict(future_df)
df.to_csv(output_csv_path, index=False, sep=",")

print(f"CSV file saved at {output_csv_path}")