"""
Code to implement Bayes Classifier using Gaussian distribution model.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal as mvn
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import utils.io_utils as io_utils


def clamp_sample(x):
    """Clamp samples to [0, 1] range"""
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x


class BayesClassifier:
    def __init__(self):
        self.gaussians = dict()
        self.num_classes = None
        self.prob_classes = None

    def fit(self, X, y):
        """Fit classifier given training data and labels."""
        # Assuming classes are numbered 0 ... K-1
        self.num_classes = len(set(y))
        self.prob_classes = np.zeros(self.num_classes)

        # Iterate over various classes and fit gaussians
        for class_idx in range(self.num_classes):
            X_class = X[y == class_idx]
            self.prob_classes[class_idx] = len(X_class)

            mean = X_class.mean(axis=0)
            cov = np.cov(X_class.T)

            self.gaussians[class_idx] = {"mean": mean, "cov": cov}

        # Normalize to compute probability
        self.prob_classes /= self.prob_classes.sum()

    def sample_given_y(self, y):
        """Generate a sample given label."""
        gaussian = self.gaussians[y]
        return clamp_sample(x=mvn.rvs(mean=gaussian["mean"], cov=gaussian["cov"]))

    def sample(self):
        """Generate a sample randomly based on prior probabilities of labels."""
        y = np.random.choice(self.num_classes, p=self.prob_classes)
        return clamp_sample(x=self.sample_given_y(y))


def main():
    """Main driver method."""
    # Fit classifier to MNIST data
    X, y = io_utils.load_mnist(data_dir=Path(BASE_DIR, "data", "digit-recognizer"),
                               mode="train")
    clf = BayesClassifier()
    clf.fit(X=X, y=y)

    # Show one sample from each class
    for class_idx in range(clf.num_classes):
        # Sample for each class and resize to MNIST data size
        sample = clf.sample_given_y(y=class_idx).reshape(28, 28)
        mean = clf.gaussians[class_idx]["mean"].reshape(28, 28)

        # Show sampled data as well as fitted gaussian mean side-by-side
        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap="gray")
        plt.title("Sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap="gray")
        plt.title("Mean")
        plt.suptitle(f"Class: {class_idx}")
        plt.show()

    # Show a random sample
    sample = clf.sample().reshape(28, 28)
    plt.imshow(sample, cmap="gray")
    plt.title("Random Sample from Random Class")
    plt.show()


if __name__ == "__main__":
    main()
