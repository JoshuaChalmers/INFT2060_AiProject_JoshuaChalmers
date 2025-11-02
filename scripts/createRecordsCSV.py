# Dataset - Chest X-Ray Images (Pneumonia) https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
# Script to turn folder names from dataset into workable format in records.csv

import pandas as pd
import os

outputCSV = "../healthcare/records-1.csv"
imageFolder = r"D:/Code/INFT2060_AiProject_JoshuaChalmers/healthcare/images"
imageFormats = (".jpeg", ".jpg")

perClassLimit = 200 # Limit sample size per folder (200 = 400 images total) NORMAL has 234 and PNEUMONIA has 390

# Check for file (stops errors if image doesnt exist once loaded)
def fileExists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False
    
# title column
def addTitle(disease: str) -> str:
    d = (disease or "").strip().capitalize()
    return f"Chest XRAY = {d}" if d else "Chest XRAY = NORMAL"

# caption column
def addCaption(disease: str) -> str:
    d = (disease or "").strip().lower()
    if d == "pneumonia":
        return "Chest XRAY showing pneumonia infection."
    if d == "normal":
        return "Chest XRAY showing normal lungs."
    return "Chest XRAY medical image."

# labels column (minor update to be a bit more readable prompt)
def addLabels(disease: str) -> str:
    d = (disease or "").strip().capitalize() or "Unknown"
    parts = [
        "Mode=XRAY",
        "BodyPart=Chest",
        f"Disease={d}",
        "Year=2025"
    ]
    return "; ".join(parts)

# Build dataframe from folders
def buildData(baseDir: str) -> pd.DataFrame:
    rows = []

    for label in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(baseDir, label)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(imageFormats)]

        files.sort()
        if perClassLimit is not None:
            files = files[:min(len(files), perClassLimit)]

        for fileName in files:
            imagePath = os.path.join(folder, fileName)
            if not fileExists(imagePath):
                continue

            disease = "Normal" if label == "NORMAL" else "Pneumonia"
            
            # Build output
            rows.append({
                "imagePath": imagePath,
                "title": addTitle(disease),
                "caption": addCaption(disease),
                "labels": addLabels(disease),                
            })

    return pd.DataFrame(rows)

df = buildData(imageFolder).reset_index(drop=True)

if df.empty:
    raise SystemExit("No images found")

df.to_csv(outputCSV, index=False, encoding="utf-8")

print("Done!")