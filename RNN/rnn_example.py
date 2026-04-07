# -*- coding: utf-8 -*-
"""
RNN Basic Practice Case - Sine Sequence Prediction
Beginner-friendly with line-by-line comment explanations
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. Hyperparameter Setting =====================
# All adjustable parameters; beginners can run directly
INPUT_SIZE = 1     # Input feature dimension (sequence value is 1-dimensional)
HIDDEN_SIZE = 32   # RNN hidden layer size
OUTPUT_SIZE = 1    # Output dimension (predict 1 value)
TIME_STEP = 10     # Sequence length (use the first 10 data to predict the 11th)
BATCH_SIZE = 32    # Batch size
LR = 0.01          # Learning rate
EPOCHS = 100       # Number of training epochs

# ===================== 2. Generate Data =====================
# Generate sine function data for sequence prediction
def get_data():
    steps = np.linspace(0, 20*np.pi, 500)
    data = np.sin(steps)  # Sine curve
    
    # Construct (number of samples, sequence length, feature dimension)
    xs, ys = [], []
    for i in range(len(data) - TIME_STEP):
        x = data[i:i+TIME_STEP]
        y = data[i+TIME_STEP]
        xs.append(x)
        ys.append(y)
    
    # Convert to PyTorch tensors
    xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return xs, ys

# ===================== 3. Define RNN Model =====================
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        # RNN layer
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,  # Dimension order: (batch, seq_len, feature)
            num_layers=1       # Single-layer RNN
        )
        # Fully connected output layer
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x, h_state=None):
        # x: [batch, TIME_STEP, INPUT_SIZE]
        # out: [batch, TIME_STEP, HIDDEN_SIZE]
        # h_state: [1, batch, HIDDEN_SIZE] Hidden state
        out, h_state = self.rnn(x, h_state)
        
        # Only take the output of the last time step for prediction
        out = self.fc(out[:, -1, :])
        return out

# ===================== 4. Train Model =====================
def train():
    # Initialization
    xs, ys = get_data()
    model = SimpleRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Start training...")
    for epoch in range(EPOCHS):
        # Forward propagation
        pred = model(xs)
        loss = criterion(pred, ys)
        
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # Visualize results
    plot_result(model, xs, ys)

# ===================== 5. Result Visualization =====================
def plot_result(model, xs, ys):
    model.eval()
    with torch.no_grad():
        pred = model(xs)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(ys.numpy(), label='True Value', color='blue')
    plt.plot(pred.numpy(), label='Predicted Value', color='red', linestyle='--')
    plt.title('RNN Sine Sequence Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()