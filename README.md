# FGSTP: Fine-grained Spatial-Temporal Perception for Gas Leak Segmentation

Paper: [arXiv](https://www.arxiv.org/abs/2505.00295)

This work is accepted by [IEEE ICIP 2025](https://2025.ieeeicip.org/)

## Abstract
Gas leaks pose significant risks to human health and the environment. Despite long-standing concerns, there are limited methods that can efficiently and accurately detect and segment leaks due to their concealed appearance and random shapes. In this paper, we propose a Fine-Grained Spatial-Temporal Perception (FGSTP) algorithm for gas leak segmentation. FGSTP captures critical motion clues across frames and integrates them with refined object features in an end-to-end network. Specifically, we first construct a correlation volume to capture motion information between consecutive frames. Then, the fine-grained perception progressively refines the object-level features using previous outputs. Finally, a decoder is employed to optimize boundary segmentation. Because there is no highly precise labeled dataset for gas leak segmentation, we manually label a gas leak video dataset, GasVid. Experimental results on GasVid demonstrate that our model excels in segmenting non-rigid objects such as gas leaks, generating the most accurate mask compared to other state-of-the-art (SOTA) models.

## Method
![fgstp](https://github.com/user-attachments/assets/ddc013d2-86d0-4975-a2db-a393e3bcf790)
The architecture of FGSTP. The model processes every current frame along with an adjacent frame each time. Each current frame needs to be processed twice with two adjacent frames. The encoder denoises the input frame and extracts the multi-scale features, while the decoder optimizes boundaries through the GRA and NCD modules.  $f_{i}^j$ denotes the feature of the ith frame and jth scale. The Consecutive Temporal Correlation (CTC) block captures motion information. $f^j$ denotes the CTC output in the jth scale. The Fine-grained Spatial Perception (FSP) module refines spatial features.

## Result
| Models             | **S<sub>α</sub>↑** | **F<sub>β</sub><sup>ω</sup>↑** | **M↓** | **E<sub>φ</sub>↑** | **mIOU↑** | **mDice↑** |
|--------------------|-------------------:|-------------------------------:|-------:|-------------------:|----------:|-----------:|
| MG                 | –                   | –                             | –          | –                  | –         | –          |
| SINet-V2           | 0.684               | 0.472                         | **0.022**  | 0.743              | 0.361     | 0.465      |
| XMem++             | 0.675               | 0.424                         | 0.036      | 0.675              | 0.361     | 0.459      |
| RMem               | 0.685               | 0.431                         | 0.023      | 0.711              | 0.361     | 0.451      |
| Zoomnext           | 0.691               | 0.450                         | 0.025      | 0.729              | 0.378     | 0.472      |
| SLT-Net            | 0.702               | 0.500                         | 0.025      | **0.806**          | 0.392     | 0.505      |
| **FGSTP (Ours)**   |**0.705**            | **0.507**                     | **0.022**  | 0.797              | **0.399**     | **0.509**      |

## Visualization
![屏幕截图 2025-06-16 121938](https://github.com/user-attachments/assets/1e29865b-89b6-470d-a38b-9fa93f7afc72)
Visualization results on GasVid Dataset. Our model prediction is the most accurate in different situations, $i.e.,$ close (camera) distance with complex background (cloud or birds interference, 1467), long distance with complex background (1476), medium distance with clear background (2559), long distance with clear background (2563), close distance with clear background (2566)



## Start
```
conda create -n fgstp python=3.8
conda activate fgstp
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
install [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 
```
pip install mmcv-full
pip install segmentation
pip install -r requirements.txt
```

The backbone is pre-trained on the COD10K dataset.   

[Dataset & Pretrained Backbone Link](https://drive.google.com/drive/folders/1UCw2AOAyZCqRYkpwapcw2kBQIG9_rsUy?usp=sharing)

Please put the pretrain model into the ./pretrain folder, and please change the dataset_path.py to your dataset path.

## Training 
```
python train.py
```
## Testing 
```
python test.py
```

## Citing 

If you find this code useful, please consider citing our work.
```
@INPROCEEDINGS{11084304,
  author={Zhao, Xinlong and Du, Shan},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Fine-Grained Spatial-Temporal Perception for Gas Leak Segmentation}, 
  year={2025},
  pages={869-874},
  doi={10.1109/ICIP55913.2025.11084304}}
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.00295}, 
}
```
