import numpy as np


def generate_perfect_ranking(n_actives: int, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate perfect ranking with all actives at the top.
    Returns scores (sorted high to low) and corresponding labels.
    """
    # Create sorted scores: high to low
    scores = np.linspace(1.0, 0.0, n_total)

    # Create corresponding labels
    labels = np.zeros(n_total)
    labels[:n_actives] = 1

    return scores, labels


def generate_worst_ranking(n_actives: int, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate worst ranking with all actives at the bottom.
    Returns scores (sorted high to low) and corresponding labels.
    """
    # Create sorted scores: high to low
    scores = np.linspace(1.0, 0.0, n_total)

    # Create corresponding labels with actives at the bottom
    labels = np.zeros(n_total)
    labels[-n_actives:] = 1

    return scores, labels


def generate_random_ranking(n_actives: int, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random ranking with evenly spaced scores and random label distribution.
    Returns scores (sorted high to low) and corresponding labels.
    """
    # Create evenly spaced scores from 1 to 0
    scores = np.linspace(1.0, 0.0, n_total)

    # Generate random positions for active compounds
    active_positions = np.random.choice(n_total, n_actives, replace=False)
    labels = np.zeros(n_total)
    labels[active_positions] = 1

    return scores, labels


def generate_exponential_ranking(n_actives: int, n_total: int, lambda_param: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ranking following exponential distribution from paper.
    Returns scores (sorted high to low) and corresponding labels.
    """
    # Create sorted scores: high to low
    scores = np.linspace(1.0, 0.0, n_total)
    labels = np.zeros(n_total)

    # Generate positions using exponential distribution
    positions = []
    for _ in range(n_actives):
        while True:
            U = np.random.random()
            X = -1 / lambda_param * np.log(1 - U * (1 - np.exp(-lambda_param)))
            r = int(n_total * X + 0.5)
            if 0 <= r < n_total and r not in positions:
                positions.append(r)
                break

    # Sort positions and assign labels
    positions.sort()
    labels[positions] = 1

    return scores, labels
