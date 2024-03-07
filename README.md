# Multimodal Bayesian Networks - Python Implementation

**Multimodal Bayesian Networks for Automatic Skin Disease Diagnosis - Submission to MICCAI 2024**
Implementation was carried out using PyTorch deep learning framework and Pyro deep probabilistic programming framework on a single NVIDIA RTX A6000 GPU (48 GB).

# 1. Download Datasets
* The PAD-UFES-20 dataset is available for download at: https://data.mendeley.com/datasets/zr7vgbcyr2/1.
* The SkinCon dataset is available for download at: https://skincon-dataset.github.io/.

# 2. Train, Evaluate, and Use the Models
`0_DatasetSplit` Folder:
* `PAD-UFES-20_Split.csv` is the file containing the 5-fold cross-validation splits peformed on the PAD-UFES-20 dataset.
* `SkinCon_Split.csv` is the file containing the 4:1 train-val splits peformed on the SkinCon dataset.

`1_ImageOnlyDNN` Folder:
* `PreprocessImage.ipynb` is the notebook used to resize the images, and subsequently perform gamma correction and Shades of Gray color constancy transformation.
* `ImageOnlyDNN.ipynb` is the notebook used to train and evaluate the *Image-Only Deep Neural Network*.
* `checkpoints` is the folder containing the parameters of the trained *Image-Only Deep Neural Networks*.

`2_MetadataOnlyBN` Folder:
* `MetadataOnlyBN.ipynb` is the notebook used to train and evaluate the *Metadata-Only Bayesian Network*.
* `metadata` is the folder containing the data used for training, validating, and testing the *Metadata-Only Bayesian Network*.
* `checkpoints` is the folder containing the parameters of the trained *Metadata-Only Bayesian Network*s.

`3_MultimodalBN` Folder: 
* `MultimodalBN.ipynb` is the notebook used to train and evaluate the *Multimodal Bayesian Network*.
* `metadata` is the folder containing the data used for training, validating, and testing the *Multimodal Bayesian Network*.
* `checkpoints` is the folder containing the parameters of the trained *Multimodal Bayesian Network*s.

`4_ConceptBN` Folder:
* `ConceptBN.ipynb` is the notebook used to train and evaluate the *Concept Bayesian Network*.
* `2ClassBN.ipynb`, `4ClassBN.ipynb`, and `FusedConceptBN.ipynb` are the notebooks used to perform the follow-up extensibility experiment, which is the *Fused Concept Bayesian Network*.
* `metadata` is the folder containing the data used for training, validating, and testing the *Concept Bayesian Network* as well as the *Fused Concept Bayesian Network*.
* `checkpoints` is the folder containing the parameters of the trained *Concept Bayesian Network*s and the *Fused Concept Bayesian Network*s.
* `backbones` is the folder containing the parameters of the trained EfficientNetB3s used to extract high-level clinical concepts from the skin lesion images.

`5_Demo` Folder:
* `Demo.ipynb` is a notebook demo of our proposed *Multimodal Bayesian Network* that first reads in parameters of the EfficientNetB3 deep neural network backbone `CNN_params.pt` and parameters of the Bayesian network `BN_params.pt`, then constructs the multimodal Bayesian network.
* **In order to run the demo, just execute the notebook cell by cell.** Sample images and metadata are provided in `img_sample` and `metadata_sample` respectively. You are welcome to test on your own samples.
