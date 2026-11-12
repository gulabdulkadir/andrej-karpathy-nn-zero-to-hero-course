# Neural Networks: Zero to Hero — From-Scratch Implementations

This repository contains pure Python and PyTorch implementations of foundational deep learning concepts and neural network architectures, built from scratch while following Andrej Karpathy's **Neural Networks: Zero to Hero** series.

The objective of this repository is to build deep mathematical intuition into automatic differentiation, reverse-mode backpropagation, gradient flow, and character-level language modeling without relying on high-level abstractions.

---

## Repository Structure

```text
├── micrograd_from_scratch.py   # Scalar-valued autograd engine & reverse-mode backpropagation
├── neuron.py                   # Custom Neuron, Layer, and MLP classes built on micrograd
├── mlp_graph                   # Visualized computational graph of the MLP forward/backward pass
├── makemore_p1.py              # Character-level Bigram Language Model (statistical & neural)
├── makemore_p2.py              # Character-level Multi-Layer Perceptron (MLP) language model
└── names.txt                   # Training dataset containing ~32,000 names
```

---

## Implemented Modules

### 1. Micrograd Autograd Engine (`micrograd_from_scratch.py`)
- Built a scalar-valued automatic differentiation engine (`Value` class) from scratch in pure Python.
- Implemented dynamic computational graph tracking, topological sorting, and reverse-mode autodiff (`backward()`) across basic arithmetic operations and non-linearities (`tanh`, `exp`, `pow`).

### 2. Neural Network Library & Graph Visualization (`neuron.py`, `mlp_graph`)
- Constructed modular `Neuron`, `Layer`, and `MLP` abstractions on top of the custom autograd engine.
- Trained a multi-layer perceptron from scratch using manual gradient zeroing and gradient descent updates, exporting the resulting computational graph (`mlp_graph`).

### 3. Makemore Part 1 — Bigram Language Model (`makemore_p1.py`)
- Implemented character-level name generation using both count-based bigram probability distributions and a single-layer neural network in PyTorch.
- Explored negative log-likelihood (NLL) loss, one-hot encoding, regularization, and multinomial sampling on `names.txt`.

### 4. Makemore Part 2 — Multi-Layer Perceptron (`makemore_p2.py`)
- Scaled the character-level language model into a Multi-Layer Perceptron inspired by Bengio et al. (2003).
- Implemented learned character embedding lookup tables, hidden `tanh` layers, mini-batch gradient descent, cross-entropy loss, and train/dev/test dataset splits.

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** PyTorch, NumPy, Matplotlib, Graphviz

---

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<kullanici-adin>/nn-zero-to-hero-from-scratch.git
   cd nn-zero-to-hero-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib graphviz
   ```

3. Run any module script:
   ```bash
   python neuron.py
   python makemore_p1.py
   python makemore_p2.py
   ```
