# Dataset - Fashion Product Images (Small) https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
# Script to turn styles.csv from dataset into workable format in products.csv

import pandas as pd
import os

inputCSV = "../ecommerce/styles.csv"
outputCSV = "../ecommerce/products-4.csv"
totalRows = 1000 # Fourth iteration of CLIP training to use 1000 images
imageFolder = r"D:/Code/INFT2060_AiProject_JoshuaChalmers/ecommerce/images"
imageFormat = ".jpg"

# Had to add checking for bad csv data, original line was df = pd.read_csv(inputCSV, dtype=str).fillna("")
colNames = [
    "id","gender","masterCategory","subCategory","articleType",
    "baseColour","season","year","usage","productDisplayName"
]

df = pd.read_csv(
    inputCSV,
    dtype=str,
    usecols=colNames,
    engine="python",
    on_bad_lines="skip",
)
df = df.fillna("")

print(f"Done after skipping bad lines: {len(df)}")

# imagePath column
# Had to use os.path as had issues coding between home / study devices
def addImagePath(idStr: str) -> str:
    return os.path.join(imageFolder, f"{idStr}{imageFormat}")

# title column
def addTitle(row: pd.Series) -> str:
    return row["productDisplayName"] or "Untitled Product"

# caption column
def addCaption(row: pd.Series) -> str:
    parts = [
        row["baseColour"],
        row["articleType"],
        f"for {row['gender']}" if row["gender"] else "",
        row["usage"],
        row["season"],
        str(row["year"]) if row["year"] else "",
    ]
    parts = [p.strip() for p in parts if p and str(p).strip()]
    return ", ".join(parts) or addTitle(row)

# labels column
def addLabels(row: pd.Series) -> str:
    parts = []
    if row["articleType"]:
        parts.append(f"ArticleType: {row['articleType']}")
    if row["baseColour"]:
        parts.append(f"Colour: {row['baseColour']}")
    if row["gender"]:
        parts.append(f"Gender: {row['gender']}")
    if row["season"]:
        parts.append(f"Season: {row['season']}")
    if row["usage"]:
        parts.append(f"Usage: {row['usage']}")
    if row["year"]:
        parts.append(f"Year: {row['year']}")
    return "; ".join(parts) if parts else "No Label"

# Check for file (stops errors if image doesnt exist once loaded)
def fileExists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False
    
# Fill out imagePath column
df["imagePath"] = df["id"].apply(addImagePath)
dfExisting = df[df["imagePath"].apply(fileExists)].reset_index(drop=True)

if dfExisting.empty:
    raise SystemExit(
        "No matching images found (PATH TYPO?)"
    )

# CSV Row count (there will always be enough rows, however keeping this for future reference)
if len(dfExisting) >= totalRows:
    work = dfExisting.sample(totalRows, random_state=42).reset_index(drop=True)
else:
    print(f"Only {len(dfExisting)} rows found, adding all to csv")
    work = dfExisting.copy()

# Build output and save to csv
out = pd.DataFrame({
    "imagePath": work["imagePath"],
    "title": work.apply(addTitle, axis=1),
    "caption": work.apply(addCaption, axis=1),
    "labels": work.apply(addLabels, axis=1),
})

out.to_csv(outputCSV, index=False, encoding="utf-8")

print("Done!")