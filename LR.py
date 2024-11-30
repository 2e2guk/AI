# LR model
# 학습률 :  0.01, epoch = 50000, 정규화 방식 = L2 norm, optimizer = GD
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 초기화
def load_data():
    X = np.array([[0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
                  [0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595]])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    return X, y

# 2. Logistic Regression with L2 Regularization
def train_logistic_regression(X, y, learning_rate=0.01, num_epochs=50000, l2_lambda=0.001):
    num_features = X.shape[0]
    num_samples = X.shape[1]

    weights = np.zeros((num_features, 1))  # Adjusted to column vector
    bias = 0
    loss_history = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(X, y, weights, bias, l2_lambda):
        z = np.dot(weights.T, X) + bias
        predictions = sigmoid(z)
        cross_entropy = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
        l2_penalty = l2_lambda * np.sum(weights**2)
        return cross_entropy + l2_penalty

    def compute_gradients(X, y, weights, bias, l2_lambda):
        z = np.dot(weights.T, X) + bias
        predictions = sigmoid(z)
        error = predictions - y
        dw = (1 / num_samples) * np.dot(X, error.T) + 2 * l2_lambda * weights
        db = (1 / num_samples) * np.sum(error)
        return dw, db

    for epoch in range(num_epochs):
        loss = compute_loss(X, y, weights, bias, l2_lambda)
        dw, db = compute_gradients(X, y, weights, bias, l2_lambda)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        loss_history.append(loss)

        # Optional: Print loss every 10000 epochs
        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return weights, bias, loss_history

# 3. 예측 함수 추가
def predict(X, weights, bias):
    z = np.dot(weights.T, X) + bias
    predictions = sigmoid(z)
    return (predictions >= 0.5).astype(int).flatten()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. Main Execution
if __name__ == "__main__":
    X, y = load_data()
    weights, bias, loss_history = train_logistic_regression(
        X, y, learning_rate=0.01, num_epochs=50000, l2_lambda=0.001
    )

    # Plotting the loss
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Loss (L2 Regularization)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Convergence for LR (L2 Regularization)')
    plt.legend()
    plt.show()

    print("Final Weights:", weights.flatten())
    print("Final Bias:", bias)

    # 예측 및 결과 출력
    predictions = predict(X, weights, bias)
    print("\nTrue Labels:", y)
    print("Predictions:", predictions)
