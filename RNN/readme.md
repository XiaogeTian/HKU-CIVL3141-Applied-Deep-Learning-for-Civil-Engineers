# RNN Basic Practice Tutorial (Simplified Intro Version)
A beginner-friendly RNN introductory project that explains RNN principles through a simple sequence prediction task, including complete code + line-by-line comments + English explanations.

## Project Introduction
This project uses **PyTorch** to implement a **basic RNN model** to complete a **sine sequence prediction task**:
- Input: A segment of continuous sine curve values
- Output: Predict the value at the next moment
- Goal: Understand the core logic of RNN in processing sequence data

## Core Knowledge Points
1. Working principle of RNN (Recurrent Neural Network)
2. Preprocessing methods for sequence data
3. Complete process of building an RNN model with PyTorch
4. Model training + prediction visualization

## Running Environment
Python 3.8+
Dependent Libraries:
- torch
- numpy
- matplotlib

## Quick Start
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Code
```bash
python rnn_example.py
```

### 3. View Results
After running, it will automatically output:
- Loss value during training
- Comparison chart of predicted results
- Model prediction effect

## Code Explanation
### 1. Data Preparation
Generate sine sequence data, split the continuous sequence into "input sequence" and "target value", and build the training dataset.

### 2. RNN Model Definition
Use PyTorch's nn.RNN to build the model, including:
- Input layer
- RNN recurrent layer
- Fully connected output layer

### 3. Model Training
Use MSE loss function + Adam optimizer to train the model iteratively.

### 4. Result Visualization
Plot the real sequence vs predicted sequence to intuitively show the effect.

## Suitable for
- Deep learning beginners
- Learners who want to quickly understand RNN principles
- Developers who need simple RNN practice cases

## Project Highlights
- Minimal code with no redundant logic
- Line-by-line English comments for easy understanding
- Fully runnable without additional data
- Visualized results for intuitive comprehension