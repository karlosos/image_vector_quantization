import random
import numpy as np
from timebudget import timebudget


@timebudget
def random_codebook(vectors, length=512):
    codebook = random.sample(np.unique(vectors, axis=0).tolist(), length)
    # Following line is 2 times faster but creates codebook with possible duplicates
    # codebook = random.sample(vectors.tolist(), length)
    return np.array(codebook)
