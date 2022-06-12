# Machine Learning for Acknowledgement Analysis
<!-- This is the implementation for the submission **Towards Automated Over-Sampling for Imbalanced Classification** -->

# Installation
Make sure that you have Python 3.6+ installed. Install with
```
pip3 install -r requirements.txt
```

# Datasets
Our dataset will be publicly avialable after acceptation. 

# Quick Start
Preprocessing the munually labeled datasets from five Universities:
```
python ack_preprocess.py
```
Train the SVM model and use the trained model to predict labels for unlabeled documents:
```
python ack_preprocess_test.py
```
