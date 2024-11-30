import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 초기화
def load_data():
    X = np.array([[0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
                  [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    return X, y

# 2. 활성화 함수
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 3. 손실 함수 정의 (Cross-Entropy 사용)
def compute_loss(y, y_pred, weights, l2_lambda):
    cross_entropy = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    l2_penalty = l2_lambda * np.sum(weights ** 2)
    return cross_entropy + l2_penalty

# 4. MLP 학습 함수
def train_mlp(X, y, hidden_size=30, num_epochs=50000, learning_rate=0.01, l2_lambda=0.001):
    np.random.seed(42)
    input_size = X.shape[0]  # Number of features (2)
    output_size = 1
    m = X.shape[1]  # Number of samples (10)

    # 가중치 초기화
    weights_hidden = np.random.randn(hidden_size, input_size) * 0.01  # (10, 2)
    bias_hidden = np.zeros((hidden_size, 1))  # (10, 1)
    weights_output = np.random.randn(output_size, hidden_size) * 0.01  # (1, 10)
    bias_output = np.zeros((output_size, 1))  # (1, 1)

    losses = []

    # Reshape y to match output dimensions
    y = y.reshape(1, m)  # (1, 10)

    # 학습 루프
    for epoch in range(num_epochs):
        # Forward Pass
        Z_hidden = np.dot(weights_hidden, X) + bias_hidden  # (10, 10)
        A_hidden = relu(Z_hidden)  # (10, 10)
        Z_output = np.dot(weights_output, A_hidden) + bias_output  # (1, 10)
        A_output = sigmoid(Z_output)  # (1, 10)

        # Loss 계산 (Cross-Entropy 사용)
        loss = compute_loss(y, A_output, weights_output, l2_lambda)
        losses.append(loss)

        # Backward Pass
        dZ_output = A_output - y  # (1, 10)
        dW_output = (1 / m) * np.dot(dZ_output, A_hidden.T) + l2_lambda * weights_output  # (1, 10)
        db_output = (1 / m) * np.sum(dZ_output, axis=1, keepdims=True)  # (1, 1)

        dA_hidden = np.dot(weights_output.T, dZ_output)  # (10, 10)
        dZ_hidden = dA_hidden * relu_derivative(Z_hidden)  # (10, 10)
        dW_hidden = (1 / m) * np.dot(dZ_hidden, X.T) + l2_lambda * weights_hidden  # (10, 2)
        db_hidden = (1 / m) * np.sum(dZ_hidden, axis=1, keepdims=True)  # (10, 1)

        # Gradient Descent 업데이트
        weights_output -= learning_rate * dW_output
        bias_output -= learning_rate * db_output
        weights_hidden -= learning_rate * dW_hidden
        bias_hidden -= learning_rate * db_hidden

        # 주기적 출력
        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return weights_hidden, bias_hidden, weights_output, bias_output, losses

# 5. 예측 함수
def predict(X, weights_hidden, bias_hidden, weights_output, bias_output):
    Z_hidden = np.dot(weights_hidden, X) + bias_hidden  # (10, 10)
    A_hidden = relu(Z_hidden)  # (10, 10)
    Z_output = np.dot(weights_output, A_hidden) + bias_output  # (1, 10)
    A_output = sigmoid(Z_output)  # (1, 10)
    return (A_output >= 0.5).astype(int).flatten()

# 6. Main 실행 코드
if __name__ == "__main__":
    X, y = load_data()
    weights_hidden, bias_hidden, weights_output, bias_output, losses = train_mlp(
        X, y, hidden_size=30, num_epochs=50000, learning_rate=0.01, l2_lambda=0.001
    )

    # 손실 함수 출력
    plt.plot(range(len(losses)), losses, label="MLP (1 Hidden Layer, ReLU, Sigmoid Output)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Convergence for MLP (ReLU Hidden, Sigmoid Output)")
    plt.legend()
    plt.show()

    # 예측 결과 출력
    predictions = predict(X, weights_hidden, bias_hidden, weights_output, bias_output)
    print("\nTrue Labels:", y.flatten())
    print("Predictions:", predictions)
