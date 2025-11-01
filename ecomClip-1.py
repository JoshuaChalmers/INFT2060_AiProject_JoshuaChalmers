## CLIP Test #1 - open-clip
## Run from INFT2060_AiProject_JoshuaChalmers
## Dataset 100 images - Fashion Product Images (Small) https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

# Config
dataSize = 100
modelName = "ViT-B-32"
preTrained = "laion2b_s34b_b79k"
device = "cpu" # AMD Card
seed = 42

csvPath = Path(__file__).resolve().parent / "ecommerce" / "products.csv"

# Manually setting seed so if the test is re run, we get the same results for consistency
def setSeed(seedVal: int):
    torch.manual_seed(seedVal)
    np.random.seed(seedVal)

# Load csv data
def loadData(csvFile: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(csvFile, dtype=str).fillna("")
    df["exists"] = df["imagePath"].apply(lambda p: os.path.isfile(p))
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("Check imagePath values in products.csv")
    if len(df) > n:
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df

# Normalising image for CLIP processing
def loadImage(path: str, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img)

# Normalising vectors
def normVector(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True)

# Calculating vector similarity
def simVector(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = normVector(A)
    B = normVector(B)
    return A @ B.T

# Checking accuracy
def topKAccuracy(simMat: torch.Tensor, k: int = 1) -> float:
    topk = simMat.topk(k, dim=1).indices
    gt = torch.arange(simMat.size(0), device=simMat.device).unsqueeze(1)
    correct = (topk == gt).any(dim=1).float().mean().item()
    return correct

# Main CLIP function
def main():
    print(f"Device: {device}")
    setSeed(seed)

    df = loadData(csvPath, dataSize)
    print(f"Loaded data from: {csvPath}")
    print(f"Using {len(df)} correct rows")

    print(f"Clip Model: {modelName} ({preTrained})")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        modelName, pretrained=preTrained, device=device
    )
    preprocess = preprocess_val
    tokenizer = open_clip.get_tokenizer(modelName)
    model.eval()

    # Encoding images
    print("Encoding images...")
    imageTensors = [loadImage(p, preprocess) for p in tqdm(df["imagePath"].tolist())]
    imageBatch = torch.stack(imageTensors).to(device)
    with torch.no_grad():
        imageEmb = model.encode_image(imageBatch).float()
    del imageBatch

    # Encoding text
    print("Encoding text...")
    titles = df["title"].tolist()
    with torch.no_grad():
        textTokens = tokenizer(titles).to(device)
        textEmb = model.encode_text(textTokens).float()
    print("Complete!")

    # Compares text and image
    sim = simVector(textEmb, imageEmb)
    acc1 = topKAccuracy(sim, k=1)
    acc5 = topKAccuracy(sim, k=5)

    # Prints results
    print("\nResults")
    print(f"Accuracy@1: {acc1:.3f}")
    print(f"Accuracy@5: {acc5:.3f}")

main()