

## Installation instructions
This repository is a Fork from multiperson. Here I report the installation instructions for using it with CUDA 11.8
```bash
python -m venv crmhenv
source crmhenv/bin/activate
python -m pip install --upgrade pip==19.3.1
pip install -r multiperson/requirements_first.txt
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0"
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
python -c "import torch; print(torch.cuda.is_available())"

pip install -r multiperson/requirements_second.txt

pip install pika pyyaml==6.0


module load ninja

export FORCE_CUDA="1"
export PYTORCH_CUDA_DEBUG=1

cd multiperson/neural_renderer

python setup.py install

cd ../mmcv

python setup.py install

cd ../mmdetection

bash compile.sh

python setup.py develop

cd ../sdf

python setup.py install

export TORCH_HOME="/leonardo/home/userexternal/lsantos0/multiperson/mmdetection/cache"

cd ../mmdetection

python installations.py --config=configs/smpl/tune.py --ckpt data/checkpoint.pt

pip install loguru

pip install git+https://github.com/mattloper/chumpy
```

## Run demo code

We provide code to evaluate our pretrained model on a folder of images by running:

```bash
cd mmdetection
python3 tools/demo.py --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
```

