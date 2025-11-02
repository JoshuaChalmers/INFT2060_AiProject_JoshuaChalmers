## INFT2060_AiProject
### Setting up Environment (MiniConda)


#### Creating ENV
`conda create -n clipenv python=3.10`

#### Activating ENV
`conda activate clipenv`

#### Install Packages
`conda install pandas pillow tqdm`

`conda install "numpy<2"`

`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
##### If you have Nvidia, run:
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

`pip install open_clip_torch`

#### Verify
`python -c "import torch, pandas; print(torch.__version__, pandas.__version__)`

#### Make sure to point IDE at clipenv Python: Select Interpreter