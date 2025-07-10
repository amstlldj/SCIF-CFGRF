# SCIF-CFGRF(Initial title of the paper:"Spatiotemporal-Conditional Information Fusion Classifier-FreeGuidance Rectified Flow Generative Model for Bearing Fault Diagnosis")

## 1.Copyright statement and update plan
### 1.1 Copyright statement and update plan
This code repository is the official code repository of paper "Data Augmentation for Bearing Fault Diagnosis Using a Novel Flow-Based Generative Model", and the account author is the original author of the paper and source code. The project's file usage instructions will be supplemented later.
The code files of this repository are still being uploaded recently, and the weight files will be considered to be uploaded to Google Cloud Drive due to their large size. If you reference this project, please inform us in the issue, thank you!(2025-3-23)
### 1.2 update
We have uploaded the weight files, which can be downloaded and used at the link. At the same time, each folder has a brief introduction, which can be seen in the Readme in each folder.(2025-3-25)

## 2.Weight file download link and instructions
###
To ensure the credibility of our paper, we decided to provide the pre-trained weight files of our model, and also to facilitate readers to directly complete image data synthesis by loading our weight files into the generator file.
Our model pre-trained weight files can be obtained from this link:
https://drive.google.com/drive/folders/1-nT8oyWBRETIntGGdyILk6AqfBFtFGXD?usp=drive_link
Replace the downloaded checkpoint folder with the downloaded checkpoint folder, and then load the corresponding .pth file in the folder in the generator to run and synthesize data.

## 3.Brief Instructions
### 3.1 dl.yaml
dl.yaml: This file contains all the environment configurations required for this project. You can use this file in conda to quickly deploy a virtual environment that can run our project.
### 3.2 config.yaml
config.yaml: This file contains the training and generation parameter configurations of the train.ipynb and generator.ipynb files, which can be adjusted according to your needs.
### 3.3 train.ipynb and generator.ipynb
train.ipynb: This file can be used to train your model, but the data needs to be adjusted to a level acceptable to the model.py and dataset folders. The specific resolution should be adjusted according to the individual device, and 32x32 pixel images are recommended.
generator.ipynb: This file is used to load and synthesize pre-trained files.

## 4.Abstracts of the papers to which this repository belongs
###
Bearings are critical components in industrial machinery, making fault diagnosis essential for ensuring operational
reliability. Due to the rarity of fault conditions, the resulting
signal data are extremely scarce compared to normal signal
data, resulting in severe class imbalance and biasing traditional
diagnostic methods toward the normal states. Researchers have
turned to data augmentation to address the aforementioned challenge. However, existing approaches often struggle to capture the
correlations between spatiotemporal and conditional information
and suffer from inefficiency. To address these limitations, we
propose a novel Spatiotemporal-Conditional information fusion
Classifier-Free Guidance Rectified Flow generative model (SCIFCFGRF). Specifically, the Continuous Wavelet Transform-based
Feature Extraction Network block and the Residual Seaformer
block, both carefully designed and integrated into the UNet
architecture, enable high-fidelity modeling of high-dimensional
Rectified Flow (RF) representations in 2D time–frequency domain signal feature heatmaps, while simultaneously embedding
spatiotemporal-conditional information fusion. Furthermore, a
dedicated loss function tailored to the characteristics of RF
ensures fast and stable convergence during training. Finally,
a novel spatiotemporal-conditional information fusion classifierfree guidance inference sampler and Quality Enhancer block
complete the efficient and high-quality synthesis of samples.
Experiments on two real-world bearing fault datasets and extensive ablation studies demonstrate that SCIF-CFGRF significantly outperforms mainstream methods in terms of synthesis
quality and inference speed. Under a 1:5 class balance ratio,
SCIF-CFGRF achieves a Cosine Similarity exceeding 0.94 and
a diagnostic accuracy of 99.50% on synthetic samples while
requiring only 6.97% of the inference time compared to the
DDPM-based model.
###
The key contributions of this study are as follows:

(1) We propose SCIF-CFGRF, a model that fuses
spatiotemporal-conditional information via a CFEN-block and
RS-block within a UNet backbone built upon RF. The CFENblock combines Continuous Wavelet Transform, RF, and a
feature extractor to model RF in the 2D time–frequency
domain, generating feature heatmaps and high-dimensional
representations. The RS-block further fuses these features and
spatiotemporal-conditional information to enable high-fidelity 
signal modeling.

(2) We design an optimization loss tailored to the path
and velocity characteristics of RF, facilitating faster and more
stable training of the SCIF-CFGRF backbone. This enhances
both synthesis quality and training efficiency, supporting rapid
industrial deployment.

(3) We propose a novel Classifier-Free Guidance inference
sampler and a QE-block based on spatiotemporal-conditional
RF information fusion, which eliminates the need for explicit classifiers while enhancing both generation quality and
efficiency. The integration of the RF-based Euler synthesis
method further improves performance and accelerates deployment.

(4) Experiments on 14 balanced datasets from two realworld bearing datasets show that SCIF-CFGRF surpasses
mainstream methods in synthesis quality. Under 1:400 imbalance, it achieves a cosine similarity of 0.94 and a downstream
accuracy of 80%, with inference time reduced to 5% of
DDPM.

###
In the future, we plan to test our method on additional industrial equipment, such as aircraft engines, chillers, and gearboxes. Moreover, we aim to explore its application in the financial sector for related research.

## 5.Recommended hardware equipment
CPU recommendation: i7-14650HX(Laptop)

GPU recommendation: RTX-4060(8GB)(Laptop)

Memory：32GB

Solid State Drive：1TB

It is recommended to use a higher configuration than this. The author's device uses shared memory to prevent out of GPU memory. It is recommended that readers use devices with more than 12GB of GPU memory to prevent program crashes or reduced model performance.

## 6.Schematic diagram of the structure of SCIF-CFGRF and the framework of the entire method

![Example Image](Fig/SCQ-CFGRF-blocks.jpg)
Fig.1 SCIF-CFGRF model and each block structure diagram.

![Example Image](Fig/SCQ-CFGRF.jpg)
Fig.2 Proposed methodological framework.

## 7.Results Visualization

![Example Image](Fig/cwru_result.jpg)
Fig.3 Result on CWRU dataset.

![Example Image](Fig/seu_result.jpg)
Fig.4 Result on SEU dataset.

![Example Image](Fig/Gif/br1_400_cwru/batch_0_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_1_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_2_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_3_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_4_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_5_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_6_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_7_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_8_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_cwru/batch_9_image_0_diffusion_process.gif)

Fig.5 GIF of the inference synthesis process on the CWRU dataset at BR1:400.

![Example Image](Fig/Gif/br1_5_cwru/batch_0_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_1_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_2_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_3_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_4_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_5_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_6_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_7_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_8_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_cwru/batch_9_image_0_diffusion_process.gif)

Fig.6 GIF of the inference synthesis process on the CWRU dataset at BR1:5.

![Example Image](Fig/Gif/br1_400_seu/batch_0_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_seu/batch_1_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_seu/batch_2_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_seu/batch_3_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_400_seu/batch_4_image_0_diffusion_process.gif)

Fig.7 GIF of the inference synthesis process on the SEU dataset at BR1:400.

![Example Image](Fig/Gif/br1_5_seu/batch_0_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_seu/batch_1_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_seu/batch_2_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_seu/batch_3_image_0_diffusion_process.gif)
![Example Image](Fig/Gif/br1_5_seu/batch_4_image_0_diffusion_process.gif)

Fig.8 GIF of the inference synthesis process on the SEU dataset at BR1:5.

![Example Image](Fig/time-3.bmp)

Fig.8 Inference process efficiency comparison. (From top to bottom: SCIF-CFGRF, DDPM, DDIM.)

## 8.Citation method of this paper repository
###
If your paper, research or project uses our research, please use this latex citation format:

@misc{SCIF-CFGRF,
  author = {Hongliang Dai, Dongjie Lin, Junpu He, Xinyu Fang, Siting Huang},
  title = {SCIF-CFGRF},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15083611},
  url = {https://github.com/amstlldj/SCIF-CFGRF}

or

@software{amstlldj_2025_15083611,
  author       = {Hongliang Dai, Dongjie Lin, Junpu He, Xinyu Fang, Siting Huang},
  title        = {amstlldj/SCIF-CFGRF: Data Augmentation for Bearing
                   Fault Diagnosis Using a Novel Flow-Based
                   Generative Model
                  },
  month        = mar,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15083611},
  url          = {https://doi.org/10.5281/zenodo.15083611},
  swhid        = {swh:1:dir:f69d22d49556a2f0b882c1d34b8a1000e92fd8e4
                   ;origin=https://doi.org/10.5281/zenodo.15083610;vi
                   sit=swh:1:snp:09a80b47e4c679e71b90b4020f498667ee45
                   bdbd;anchor=swh:1:rel:5fe1544ec3cdd61acd4dd183cbca
                   44643218d6c5;path=amstlldj-SCIF-CFGRF-76a8de6
                  },
}

If you plagiarize our research, we will pursue legal action.

## 9.Subsequent plans for this code repository
###
We will continue to improve the code comments of the project later.

## 10.Our other projects
### 10.1 DSEA-SMOTE
You can also follow our other fault data synthesis project DSEA-SMOTE (https://github.com/amstlldj/DSEA-SMOTE). The project will also be updated and maintained in the future.

## 11.Contact us for cooperation or consultation
###
If you have any questions, please contact the author's work email or leave a message in issues. The author will try his best to answer them at his convenience.If you are interested in seeking cooperation, you can also consult this email.
Author contact email: 2112464120@e.gzhu.edu.cn
