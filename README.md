## INFT2060_AiProject
### Setting up Environment

Creating ENV
conda create -n clipenv python=3.10

Activating ENV
conda activate clipenv

Install Packages
conda install pandas pillow tqdm
conda install "numpy<2"
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install open_clip_torch