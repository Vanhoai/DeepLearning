# 🧠 Neural Network from Scratch with NumPy

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Ready%20to%20Use-brightgreen.svg)]()

> 🚀 A complete Neural Network implementation from scratch using pure NumPy - Educational deep learning framework for understanding how neural networks really work!

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Components](#-components)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## 🔍 Overview

This project is a **complete implementation of Neural Networks from scratch** using only NumPy. It's designed for educational purposes to help understand the mathematics and mechanics behind deep learning.

### 🎯 Why This Project?

- ✅ **Learn by Implementation**: Understand backpropagation, gradient descent, and optimization algorithms
- ✅ **No Black Boxes**: Every component is implemented from scratch with clear documentation
- ✅ **Production Features**: Includes regularization, early stopping, model saving, and more
- ✅ **MNIST Ready**: Train on real datasets and achieve 95%+ accuracy
- ✅ **Well Documented**: Extensive guides in both English and Vietnamese

## ✨ Features

### Core Components

| Component             | Description                           | Implementations                              |
| --------------------- | ------------------------------------- | -------------------------------------------- |
| 🔄 **Activations**    | Non-linear activation functions       | ReLU, Sigmoid, Softmax, Tanh, LeakyReLU, ELU |
| 🧱 **Layers**         | Fully connected neural network layers | 7 layer types with different activations     |
| ⚡ **Optimizers**     | Advanced optimization algorithms      | SGD (momentum), Adam, AdaGrad, RMSProp       |
| 📉 **Loss Functions** | Training objective functions          | MSE, Binary Cross Entropy, Cross Entropy     |
| 🛡️ **Regularization** | Prevent overfitting                   | L2 (Ridge) Regularization                    |
| ⏸️ **Early Stopping** | Automatic training termination        | Monitor validation loss/accuracy             |
| 💾 **Model I/O**      | Save and load trained models          | NumPy-based serialization                    |

### Additional Features

- ✅ **Mini-batch Training**: Efficient batch gradient descent
- ✅ **Validation Monitoring**: Track training progress in real-time
- ✅ **History Tracking**: Record loss and accuracy over epochs
- ✅ **MNIST Dataset Loader**: Built-in support with TensorFlow/PyTorch fallback
- ✅ **Visualization Tools**: Plot training curves and predictions
- ✅ **Model Summary**: Display architecture and parameter counts

## 🚀 Quick Start

### 3-Step Getting Started

```bash
# 1. Install dependencies
cd DeepLearning
uv sync

# 2. Test installation
python examples/test_imports.py

# 3. Train on MNIST
python examples/mnist_simple.py
```

### 5-Line Training Example

```python
from nn import Sequential, ReLu, Softmax, CrossEntropy, Adam, load_mnist_dataset

X, Y, Xt, Yt = load_mnist_dataset()
model = Sequential([ReLu(784, 128), Softmax(128, 10)], CrossEntropy(), Adam())
model.fit(X, Y, epochs=50, batch_size=128, validation_data=(Xt, Yt))
print(f"Accuracy: {model.evaluate(Yt, model.predict(Xt)):.2%}")
```

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- NumPy 2.3+
- (Optional) TensorFlow or PyTorch for MNIST dataset

### Installation Steps

```bash
# Clone repository
git clone <your-repo-url>
cd DeepLearning

# Install using uv (recommended)
uv sync

# Or install using pip
pip install -e .
```

### Verify Installation

```bash
python examples/test_imports.py
```

Expected output:

```
==============================================================
All imports successful! ✓
==============================================================
All tests passed! ✓
```

## 💻 Usage Examples

### Example 1: Basic MNIST Training

```python
import numpy as np
from nn import (
    Sequential, ReLu, Softmax,
    CrossEntropy, Adam,
    load_mnist_dataset
)

# Load data
X_train, Y_train, X_test, Y_test = load_mnist_dataset()

# Build model
model = Sequential(
    layers=[
        ReLu(di=784, do=128),
        ReLu(di=128, do=64),
        Softmax(di=64, do=10),
    ],
    loss=CrossEntropy(),
    optimizer=Adam(eta=0.001),
)

# Train
history = model.fit(
    X=X_train,
    Y=Y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, Y_test),
    verbose=True,
)

# Evaluate
predictions = model.predict(X_test)
accuracy = model.evaluate(Y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Example 2: With Regularization & Early Stopping

```python
from nn import (
    Sequential, ReLu, Softmax,
    CrossEntropy, Adam,
    RegularizationType,
    EarlyStopping, MonitorEarlyStopping,
    load_mnist_dataset
)

# Load data
X_train, Y_train, X_test, Y_test = load_mnist_dataset()

# Build model with L2 regularization
model = Sequential(
    layers=[
        ReLu(di=784, do=128),
        ReLu(di=128, do=64),
        Softmax(di=64, do=10),
    ],
    loss=CrossEntropy(),
    optimizer=Adam(eta=0.001),
    regularization=RegularizationType.L2_REGULARIZATION,
    regularization_lambda=0.01,
)

# Setup early stopping
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    is_store=True,
    monitor=MonitorEarlyStopping.VAL_LOSS,
)

# Train
history = model.fit(
    X=X_train,
    Y=Y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, Y_test),
    early_stopping=early_stopping,
)

# Save model
model.save(path='./saved_models/mnist')
```

### Example 3: 2D Visualization

```bash
# Run 2D classification with decision boundary visualization
python examples/2d_classification.py
```

This will generate:

- Training dataset visualization
- Decision boundary plots
- Training history curves

## 📁 Project Structure

```
DeepLearning/
├── packages/nn/src/nn/              # 🧠 Neural Network Library
│   ├── __init__.py                  # Public API exports
│   ├── activation.py                # Activation functions
│   ├── layer.py                     # Layer implementations
│   ├── optimizer.py                 # Optimization algorithms
│   ├── loss.py                      # Loss functions
│   ├── model.py                     # Sequential model
│   ├── regularization.py            # Regularization techniques
│   ├── early_stopping.py            # Early stopping callback
│   ├── datasets.py                  # Dataset loaders
│   └── common.py                    # Base classes
│
├── examples/                         # 📚 Usage Examples
│   ├── test_imports.py              # Installation test
│   ├── mnist_simple.py              # Quick MNIST (3 min)
│   ├── mnist_training.py            # Full MNIST (15 min)
│   ├── 2d_classification.py         # 2D visualization
│   ├── run_all.py                   # Interactive menu
│   └── README.md                    # Examples documentation
│
├── GETTING_STARTED.md               # 📖 Complete guide (EN)
├── HUONG_DAN_SU_DUNG.md            # 📖 Complete guide (VN)
├── SUMMARY_OF_CHANGES.md           # 📋 Implementation details
├── run_examples.sh                  # 🚀 Quick run script
└── README.md                        # This file
```

## 🧩 Components

### Activation Functions

```python
from nn import (
    ReLUActivation,      # f(x) = max(0, x)
    SigmoidActivation,   # f(x) = 1/(1 + e^(-x))
    SoftmaxActivation,   # Multi-class probability distribution
    TanhActivation,      # f(x) = tanh(x)
    LeakyReLUActivation, # f(x) = max(αx, x)
    ELUActivation,       # Exponential Linear Unit
)
```

### Layers

```python
from nn import Linear, ReLu, Sigmoid, Softmax, Tanh, LeakyReLU, ELU

# All layers: Layer(input_dim, output_dim)
layer = ReLu(di=784, do=128)
```

### Optimizers

```python
from nn import SGD, Adam, AdaGrad, RMSProp

# SGD with Nesterov momentum
optimizer = SGD(eta=0.01, momentum=0.9, nesterov=True)

# Adam (recommended)
optimizer = Adam(eta=0.001, beta1=0.9, beta2=0.999)

# AdaGrad
optimizer = AdaGrad(eta=0.01)

# RMSProp
optimizer = RMSProp(eta=0.001, beta=0.9)
```

### Loss Functions

```python
from nn import MSE, BinaryCrossEntropy, CrossEntropy

loss = CrossEntropy()        # Multi-class classification
loss = BinaryCrossEntropy()  # Binary classification
loss = MSE()                 # Regression
```

## 📊 Results

### MNIST Handwritten Digits

| Architecture      | Optimizer | Accuracy   | Training Time |
| ----------------- | --------- | ---------- | ------------- |
| 784→128→64→10     | Adam      | **95-97%** | ~10 minutes   |
| 784→64→10         | Adam      | 85-90%     | ~3 minutes    |
| 784→256→128→64→10 | Adam      | 97-98%     | ~20 minutes   |

### 2D Classification (3 classes, 1000 samples)

| Architecture | Optimizer | Accuracy   | Training Time |
| ------------ | --------- | ---------- | ------------- |
| 2→16→16→3    | Adam      | **95-99%** | ~1 minute     |

### Sample Output

```
Epoch 50/100 - Loss: 0.1234, Accuracy: 0.9567 - Val Loss: 0.1456, Val Accuracy: 0.9512
Test Loss: 0.1401
Test Accuracy: 0.9534 (95.34%)
```

## 📚 Documentation

### Available Guides

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete guide (English)
    - Installation
    - API documentation
    - Best practices
    - Troubleshooting

- **[HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md)** - Complete guide (Vietnamese)
    - Hướng dẫn cài đặt
    - Ví dụ code
    - Giải quyết vấn đề
    - Tips & tricks

- **[examples/README.md](examples/README.md)** - Detailed examples guide
    - Usage examples
    - Parameter tuning
    - Visualization
    - Advanced features

- **[SUMMARY_OF_CHANGES.md](SUMMARY_OF_CHANGES.md)** - Implementation details
    - Architecture overview
    - Component descriptions
    - Bug fixes
    - Future improvements

## 🎓 Running Examples

### Interactive Menu

```bash
python examples/run_all.py
```

Or use the shell script:

```bash
bash run_examples.sh
```

### Individual Examples

```bash
# Test installation (10 seconds)
python examples/test_imports.py

# 2D visualization (1 minute)
python examples/2d_classification.py

# Quick MNIST (3 minutes)
python examples/mnist_simple.py

# Full MNIST with all features (15 minutes)
python examples/mnist_training.py
```

## 🔧 Troubleshooting

### Common Issues

**Loss = NaN**

```python
# Solution: Reduce learning rate
optimizer = Adam(eta=0.0001)  # Instead of 0.001
```

**Overfitting (Train >> Test accuracy)**

```python
# Solution: Add regularization
model = Sequential(
    layers=[...],
    regularization=RegularizationType.L2_REGULARIZATION,
    regularization_lambda=0.01,
)
```

**Training too slow**

```python
# Solution: Increase batch size
model.fit(X, Y, batch_size=128)  # Instead of 32
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for comprehensive troubleshooting.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync

# Run tests
python examples/test_imports.py

# Check all examples work
python examples/run_all.py
```

## 📝 TODO / Future Improvements

- [ ] Dropout layer
- [ ] Batch Normalization
- [ ] Learning rate scheduling
- [ ] Convolutional layers (CNN)
- [ ] Recurrent layers (RNN/LSTM)
- [ ] GPU support (CuPy)
- [ ] More datasets (CIFAR-10, Fashion-MNIST)
- [ ] TensorBoard integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the mathematics in "Deep Learning" by Ian Goodfellow
- Built for educational purposes
- Thanks to the NumPy community for an amazing library

## 📧 Contact

- **Author**: Hinsun
- **Email**: vanhoai.adv@gmail.com
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/DeepLearning/issues)

---

<div align="center">

### 🌟 If this project helped you understand neural networks, please give it a star!

**Made with 💻 and ❤️ for learning Deep Learning from scratch**

</div>
