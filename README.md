# PAMR: Physics-informed meta neural representation for high-fidelity, aberration-corrected, sparse-view Fourier ptychographic tomography
This repository is developed based on the paper *Physics-informed meta neural representation for high-fidelity, aberration-corrected, sparse-view Fourier ptychographic tomography*, which provides 【】

# Contents
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Citation](#Citation)
- [Contact](#Contact)


# Requirements
- System requirements
```
 - Linux and windows 10/11 should be able to run the code, but the code has been only teseted on Linux (Ubuntu 24.04.1 LTS)
 - Python 3.10
 - CUDA 11.8 and cuDNN 8.7
 - Graphics: NVDIA RTX 3090, A6000 or better 
```

- Running environment requirements 
```
 - python=3.10.14
 - torch>=2.3.0+cu118
 - torchvision>=0.18.0+cu118
 - PyWavelets=1.7.0
 - timm=1.0.15
 - einops=0.8.0
 - opencv-python=4.10.0.84
 - yaml=6.0.1
 - numpy=1.26.3
 - pathlib2=2.3.7
 ```

***Note: To quickly set up the runtime environment, you can use the provided `PAMR_environment.yaml` file for one-click installation following the Setup Guide below***

### Quick Environment Setup Guide

1. Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. In the project root directory, run the following commands to create and activate a new environment:
   ```bash
   conda env create -f PAMR_environment.yaml
   conda activate PAMR
   ```
3. After installation, you can proceed with 3D-RI reconstruction as described below.

## Structure

```
PAMR_OpenSourced/
├── bio_dataset/                   # dataset used (.tif)
├── chkpt/                         # Saved model checkpoints for multiple tasks
├── config/                        # Model definitions using yaml
├── loss/                          # defined loss funcitons
├── model/                         # defined model structure
├── util/                          # utility functions
├── logs/                          # logs folder
├── trainer.py                     # Main script for reconstruction
├── AIMS_enviroment.yaml           # Python dependencies for rapid environment deployment
└── README.md                      # Project documentation
```

# Usage
## 3D Refractive Index (RI) reconstruction for U2OS cells

1. Environment installation and activation
2. Download the images of U2OS cells under multiple illumiination angles (**[Google driven](https://drive.google.com/file/d/1XfFNHBVSbp-IelKVBiZDXt7CszrD53e0/view?usp=sharing)**) and place in `bio_dataset/`.
3. Run the following commands for quick reconstruction, the corresponding prediction results will be saved in `Bio_sample_exp/`.

   ```bash
   python trainer.py
   ```

# Citation
If you use the repository or relevant data, please cite the corresponding paper:
```
Sun, M., Zhong, F., Mao, S., Liu, Y., Zhang, Z., Li, D., Liu, B., Fei, P., Physics-informed meta neural representation for high-fidelity, aberration-corrected, sparse-view Fourier ptychographic tomography. 
```

# Contact
For questions or suggestions regarding this code, please open an issue in this repository or contact the authors at [minglusun@hust.edu.cn](mailto:minglusun@hust.edu.cn).


