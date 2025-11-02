## Healthcare CLIP Test #2 - open-clip
## Method - changed to zero-shot classifaction rather than 1:1
## Run from INFT2060_AiProject_JoshuaChalmers
## Dataset 400 images - Chest X-Ray Images (Pneumonia) https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

# Config
dataSize = 400
modelName = "ViT-B-32"
preTrained = "laion2b_s34b_b79k"
device = "cpu" # AMD Card
seed = 42

csvPath = Path(__file__).resolve().parent / "healthcare" / "records-1.csv"

# Manually setting seed so if the test is re run, we get the same results for consistency
def setSeed(seedVal: int):
    torch.manual_seed(seedVal)
    np.random.seed(seedVal)

def loadData(csvFile: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(csvFile, dtype=str).fillna("")
    df["exists"] = df["imagePath"].apply(lambda p: os.path.isfile(p))
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("Check imagePath values in records-1.csv")
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

# Creating labels from csv
# Normal - 0
# Pneumonia - 1
def createLabels(labelsSeries: pd.Series) -> np.ndarray:
    """
    Expect df['labels'] like: 'Mode=XRAY; BodyPart=Chest; Disease=Normal; Year=2025'
    Map: Normal -> 0, Pneumonia -> 1
    """
    lab = labelsSeries.fillna("").str.lower()
    return np.where(lab.str.contains("disease=pneumonia"), 1, 0)

# Zero shot classify
def classifyPrompt(model, tokenizer, imageEmb: torch.Tensor, device="cpu") -> np.ndarray:
    prompts = [
        "chest x-ray showing normal lungs",
        "chest x-ray showing pneumonia",
    ]
    with torch.no_grad():
        textTokens = tokenizer(prompts).to(device)
        textEmb = model.encode_text(textTokens).float()

    img_n = normVector(imageEmb)
    txt_n = normVector(textEmb)
    logits = img_n @ txt_n.T
    preds = logits.argmax(dim=1).cpu().numpy()
    return preds

# Zero shot classify group
def classifyPromptGroup(model, tokenizer, imageEmb: torch.Tensor, device="cpu") -> np.ndarray:
    normalPrompts = [
        "chest x-ray with no focal consolidation",
        "frontal chest radiograph without pneumonia",
        "normal chest radiograph",
        "clear lungs on chest x-ray",
    ]
    pneumoniaPrompts = [
        "chest x-ray with airspace opacity",
        "frontal chest radiograph with lobar consolidation",
        "chest x-ray showing pneumonia",
        "infective consolidation on chest radiograph",
    ]
    prompts = normalPrompts + pneumoniaPrompts

    with torch.no_grad():
        textTokens = tokenizer(prompts).to(device)
        textEmb = model.encode_text(textTokens).float()

    img_n = normVector(imageEmb)
    txt_n = normVector(textEmb)
    logits = img_n @ txt_n.T

    n = len(normalPrompts)
    normalLogit = logits[:, :n].mean(dim=1, keepdim=True)
    pneuLogit   = logits[:, n:].mean(dim=1, keepdim=True)
    clsLogits   = torch.cat([normalLogit, pneuLogit], dim=1)

    preds = clsLogits.argmax(dim=1).cpu().numpy()
    return preds

# Checking accuracy
def accuracy(preds: np.ndarray, gt: np.ndarray) -> float:
    return (preds == gt).mean().item()

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

    # Create Labels
    ct =  createLabels(df["labels"])

    # Classify
    print("Classifying zero shot...")
    predsSingle = classifyPrompt(model, tokenizer, imageEmb, device=device)
    accSingle = accuracy(predsSingle, ct)

    predsGroup = classifyPromptGroup(model, tokenizer, imageEmb, device=device)
    accGroup = accuracy(predsGroup, ct)

    # Prints results
    print("\nResults")
    print(f"Accuracy (Single Prompt): {accSingle:.3f}")
    print(f"Accuracy (Prompt Group): {accGroup:.3f}")

main()