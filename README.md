# MNIST CNN with PyTorch

A simple Convolutional Neural Network (CNN) built with **PyTorch** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

This project trains a CNN on the MNIST dataset and evaluates its performance, achieving >98% accuracy after a few epochs.

---

## 📌 Features
- Uses **PyTorch** for building and training the CNN
- Two convolutional layers with max pooling and dropout
- Training and testing loops with accuracy/loss reporting
- Plots test accuracy per epoch
- GPU support (uses CUDA if available)

---

## 🛠 Requirements
Make sure you have the following installed:
- Python 3.8+
- [PyTorch](https://pytorch.org/)  
- torchvision  
- matplotlib  

You can install dependencies using:
```bash
pip install torch torchvision matplotlib


