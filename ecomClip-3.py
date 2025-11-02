## CLIP Test #3 - open-clip
## Run from INFT2060_AiProject_JoshuaChalmers
## Dataset 1000 images - Fashion Product Images (Small) https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

# Config
dataSize = 1000
modelName = "ViT-B-32"
preTrained = "laion2b_s34b_b79k"
device = "cpu" # AMD Card
seed = 42

csvPath = Path(__file__).resolve().parent / "ecommerce" / "products-3.csv"