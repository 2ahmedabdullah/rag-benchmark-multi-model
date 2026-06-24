# download.py

import pathlib
import os
from beir import util


# Choose the dataset (e.g., 'scifact', 'fiqa', 'nfcorpus', 'hotpotqa')
dataset = "scifact"

# Saves to a folder named "datasets" in current directory
out_dir = os.path.join(pathlib.Path.cwd(), "datasets")
data_path = util.download_and_unzip(url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip", out_dir=out_dir)

print(f"Dataset downloaded and extracted to: {data_path}")

