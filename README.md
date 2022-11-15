# Kannada-MNIST classification

[Kannada-MNIST dataset](https://www.kaggle.com/c/Kannada-MNIST/discussion/122158): a MNIST-like classification dataset (10 classes).

## Approach

- Model: CNN
- Basic data augmentation: rotation, shear, translation
- Optimiser: Adam with plateau schedule
- Final accuracy on Kaggle's hidden test dataset: 96.6%

## Reproduce the results

### Instal packages & run the model

- `pip install -r requirements.txt`
- Run the model: `python train.py`

### MLP vs CNN

A CNN is used for the Kaggle results, but is slow on CPU. So a simple MLP is provided to run the model on CPU. In `train.py`, change the model type: `model_type="mlp"` or `model_type="cnn"`
