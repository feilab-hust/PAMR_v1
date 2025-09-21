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
├── chkpt_INR/               # Saved model checkpoints for multiple tasks
├── config/                  # Model definitions using yaml
├── dataset/                 # dataset used
├── loss/                    # defined loss funcitons
├── model/                   # defined model structure
├── util/                    # utility functions
├── Test_AC.py               # Main script
├── AIMS_enviroment.yaml     # Python dependencies for rapid environment deployment
└── README.md                # Project documentation
```
# Usage
## Model inference for quick validation

1. Environment installation and activation
2. Download or prepare the trained model checkpoints (**【Google driven】**) and place in `Checkpoints/`.
3. Download the dataset (**【Google driven】**) and place it in `datatest/`.
4. Run the following commands for quick model inference, the corresponding prediction results will be output on the terminal.

   #### $\blacksquare$ Example 1: Binary classification between Ctrl and etoposide-induced (Eto) senescence using U2OS cell
      ```bash
      python Test_AC.py --Net_config 'config/Net_AC_mito_Ctrl&Eto_U2OS.yaml'
      ```

   #### $\blacksquare$ Example 2: Binary classification between Ctrl and doxorubicin-induced (Doxo) senescence using U2OS cell
      ```bash
      python Test_AC.py --Net_config 'Net_AC_mito_Ctrl&Doxo_U2OS.yaml'
      ```

   #### $\blacksquare$ Example 3: Binary classification between Ctrl and antimycin A-induced (Anti) senescence using U2OS cell
      ```bash
      python Test_AC.py --Net_config 'Net_AC_mito_Ctrl&Anti_U2OS.yaml'
      ```

   #### $\blacksquare$ Example 4: Binary classification between Ctrl and hydrogen peroxide-induced (H2O2) senescence using U2OS cell
      ```bash
      python Test_AC.py --Net_config 'Net_AC_mito_Ctrl&OS_U2OS.yaml'
      ```

   #### $\blacksquare$ Example 5: Binary classification between Ctrl and etoposide-induced senescence for cross multiple cell types (BHK21, HeLa, ARPE-19)
      ```bash
      python Test_AC.py --Net_config 'Net_AC_mito_Ctrl&Eto_crossCell.yaml'
      ```

   #### $\blacksquare$ Example 6: multi-senescent classification between Ctrl and senescent cells induced by four distinct senescence mechanisms (Doxo, Eto, H2O2, Anti)
      ```bash
      python Test_AC.py --Net_config 'Net_AC_mito_MultiSeneClass.yaml'
      ```


# Citation
If you use the repository or relevant data, please cite the corresponding paper:
```
Mao, S., Sun, M., Liu, Y., Li, D., Fei, P., Artificial Intelligence for Mitochondria-based Senescence detection (AIMS) across diverse cellular contexts. 
```

# Contact
For questions or suggestions regarding this code, please open an issue in this repository or contact the authors at [minglusun@hust.edu.cn](mailto:minglusun@hust.edu.cn).


