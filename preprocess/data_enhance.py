import numpy as np

def drop_words(words, probability):
    """Drops words with the given probability."""
    length = len(words)
    keep = np.random.uniform(size=length) > probability
    if np.count_nonzero(keep) == 0:
        ind = np.random.randint(0, length)
        keep[ind] = True
    words = np.take(words, keep.nonzero())[0]
    return words
    
def rand_perm_with_constraint(words, k):
    """Randomly permutes words ensuring that words are no more than k positions
    away from their original position."""
    length = len(words)
    offset = np.random.uniform(size=length) * (k + 1)
    new_pos = np.arange(length) + offset
    return np.take(words, np.argsort(new_pos))
    
def add_noise(words, dropout=0.1, k=3):
    """Applies the noise model in input words.
    Args:
    words: A numpy vector of word ids.
    dropout: The probability to drop words.
    k: Maximum distance of the permutation.
    Returns:
    A noisy numpy vector of word ids.
    """
    words = drop_words(words, dropout)
    words = rand_perm_with_constraint(words, k)
    return words
    
    