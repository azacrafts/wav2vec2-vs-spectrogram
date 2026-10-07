import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score


# share of correct predictions
def accuracy(probs, y):
    return (probs.argmax(1) == y).mean()


# confidence vs accuracy in 10 confidence bins
def reliability(probs, y, bins=10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (conf > low) & (conf <= high)
        if in_bin.any():
            rows.append((in_bin.mean(), conf[in_bin].mean(), correct[in_bin].mean()))
        else:
            rows.append((0.0, np.nan, np.nan))
    return np.array(rows)


# expected calibration error: gap between confidence and accuracy
def ece(probs, y, bins=10):
    table = reliability(probs, y, bins)
    weight, conf, acc = table[:, 0], table[:, 1], table[:, 2]
    keep = weight > 0
    return (weight[keep] * np.abs(conf[keep] - acc[keep])).sum()


# negative log likelihood of the true class
def nll(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()


# find temperature that minimises nll on validation logits
def fit_temperature(logits, y):
    def loss(log_t):
        return nll(softmax(logits / np.exp(log_t), axis=1), y)

    result = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


# function to compute all metrics
def evaluate(logits, y, temperature):
    probs = softmax(logits, axis=1)
    scaled = softmax(logits / temperature, axis=1)
    return {
        "acc": accuracy(probs, y),
        "f1": f1_score(y, probs.argmax(1), average="macro"),
        "ece": ece(probs, y),
        "nll": nll(probs, y),
        "overconf": probs.max(1).mean() - accuracy(probs, y),
        "ece_ts": ece(scaled, y),
        "temperature": temperature,
    }
