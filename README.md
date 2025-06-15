# FGSTP

Paper: [arXiv](https://www.arxiv.org/pdf/2505.00295)

This work is accepted by [IEEE ICIP 2025](https://2025.ieeeicip.org/)

## Start

conda create -n fgstp python=3.8

conda activate fgstp

install [2.4.1 pytorch, torchvision, and torchaudio](https://pytorch.org/get-started/previous-versions/) 

install [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 

pip install mmcv-full
pip install mmsegmentation


The backbone is pre-trained on the COD10K dataset.   

[Dataset & Pretrained Backbone Link](https://drive.google.com/drive/folders/1UCw2AOAyZCqRYkpwapcw2kBQIG9_rsUy?usp=sharing)

Please put the pretrain model into the ./pretrain folder, and please change the dataset_path.py to your dataset path.

## Training 
   python train.py
## Testing 
  python test.py


## Citing 

If you find this code useful, please consider citing our work.

@inproceedings{zhao2025fgstp,
  title={Fine-grained Spatial-Temporal Perception for Gas Leak Segmentation},
  author={Zhao, Xinlong and Du, Shan},
  booktitle={IEEE ICIP},
  year={2025}
}
