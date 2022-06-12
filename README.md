# AutoSMOTE
This is the implementation for the submission **Towards Automated Over-Sampling for Imbalanced Classification**

# Installation
Make sure that you have Python 3.6+ installed. Install with
```
pip3 install -r requirements.txt
pip3 install -e .
```

# Datasets
You don't need to mannually download datasets. Just pass the dataset name, and it will be automatically downloaded.

# Quick Start
Train on the Mozilla4 dataset with undersampler ratio of 100 and SVM as the base classifier:
```
python3 train.py
```

# Important Arguments
You can run AutoSMOTE under different configurations. Some important arguments are listed below.
*   `--dataset`: which dataset to use
*   `--clf`: which base classifeir to use
*   `--metric`: which metric to use
*   `--device`: by default it trains with GPU. Train with CPU by passing `cpu`
*   `--total_steps`: search budget
