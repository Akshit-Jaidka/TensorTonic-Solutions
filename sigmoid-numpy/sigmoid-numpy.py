import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x_vector = np.array(x, dtype=float)
    x_vector = 1 / (np.exp(-x_vector) + 1)
    return x_vector