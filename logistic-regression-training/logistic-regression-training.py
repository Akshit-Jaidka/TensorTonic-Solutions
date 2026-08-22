import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N,D = X.shape
    w = np.zeros(D)
    b = 0.0
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        w -= lr * (X.T @ (p - y)) / N
        b -= lr * np.mean(p - y)
    return w,b