import argparse
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from torch.utils.data import DataLoader

# --- Import your feature extractor classes ---
from features.CLIPFeatureExtractor import CLIPImageFeatureExtractor, OpenCLIPImageFeatureExtractor
from features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from features.InceptionFeatureExtractor import InceptionFeatureExtractor
from features.SWAVFeatureExtractor import SWAVFeatureExtractor
from datasets.ImageFilesDataset import ImageFilesDataset


def get_feature_extractor(name, pretrained=None):
    if name == 'dino':
        return DINOv2FeatureExtractor(save_path='.')
    if name == 'clip':
        return CLIPImageFeatureExtractor(save_path='.')
    if name == 'inception':
        return InceptionFeatureExtractor(save_path='.')
    if name == 'swav':
        return SWAVFeatureExtractor(save_path='.')
    if name == 'finetuned_clip':
        return OpenCLIPImageFeatureExtractor(save_path='.', pretrained=pretrained)
    raise ValueError(f"Unknown feature extractor: {name}")


def main():
    parser = argparse.ArgumentParser(description="Extract image features and save to .npz")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--caption_file", type=str, required=False, help="Path to captions text file")
    parser.add_argument("--feat_extractor", type=str, default="dino",
                        choices=["dino", "clip", "inception", "swav", "finetuned_clip"])
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint (for finetuned_clip)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="./features.npz",
                        help="Output .npz file")
    parser.add_argument("--extension", type=str, default="JPEG",
                        help="Image extension (e.g., JPEG, png, jpg)")

    args = parser.parse_args()

    # Collect image file paths
    path_files = natsorted(glob(f"{args.img_path}/*.{args.extension}"), key=str)
    if len(path_files) == 0:
        raise FileNotFoundError(f"No {args.extension} files found in {args.img_path}")

    # Load captions if provided
    captions = []
    if args.caption_file:
        with open(args.caption_file) as f:
            captions = [line.strip() for line in f]

    # Dataset & dataloader
    dataset = ImageFilesDataset(path='.', path_files=path_files,
                                name="experiment", extension=args.extension,
                                conditional=False, sort_files=True)
    print(f"Loaded dataset with {len(dataset)} images.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda x: x)
    img_data = iter(dataloader)

    # Feature extractor
    ImageFeats = get_feature_extractor(name=args.feat_extractor, pretrained=args.pretrained)

    # Storage
    img_feats_np = np.zeros((len(dataset), ImageFeats.features_size))
    index = 0

    for imgs in tqdm(img_data, desc="Extracting features"):
        # Preprocess and batch
        preprocessed_imgs = [ImageFeats.preprocess(img[0]).cuda() for img in imgs]
        img_feats = ImageFeats.get_feature_batch(torch.stack(preprocessed_imgs).cuda())
        img_feats_np[index: index + args.batch_size] = img_feats.detach().cpu().numpy()

        index += args.batch_size

        # Cleanup
        del img_feats, preprocessed_imgs
        torch.cuda.empty_cache()

    # Save
    np.savez(args.output, features=img_feats_np)
    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
