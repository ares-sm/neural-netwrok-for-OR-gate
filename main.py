import numpy as np

# OR dataset
x = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [1]   # last one changed from 0 to 1
])

# Initialize weights and bias randomly
w = np.random.rand(2,1)
b = np.random.rand(1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hyperparameters
lr = 0.1        # learning rate
epochs = 99999   # number of training iterations
n = x.shape[0]  # number of samples

# Training loop
for i in range(epochs):
    # Forward pass
    z = np.dot(x, w) + b
    y_pred = sigmoid(z)
    
    # Compute cost (MSE)
    loss = np.mean((y - y_pred)**2)
    
    # Backpropagation
    dz = 2 * (y_pred - y) * y_pred * (1 - y_pred)
    dw = np.dot(x.T, dz) / n
    db = np.sum(dz) / n
    
    # Update weights and bias
    w -= lr * dw
    b -= lr * db
    
    # Print cost every 100 iterations
    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {loss:.4f}")

# Final predictions
print("\nFinal weights:", w)
print("Final bias:", b)
print("Predictions after training:")
print(y_pred)
