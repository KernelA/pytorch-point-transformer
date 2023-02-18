from typing import List

import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedBatchSampler:
    def __init__(self,
                 batch_size: int,
                 shuffle: bool,
                 class_labels: List[int]):
        self._class_labels = class_labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_batches = len(class_labels) // batch_size
        self._dummpy_indices = np.full((len(class_labels),), 1, dtype=np.uint8)

        self.stratified = StratifiedKFold(
            self.n_batches, shuffle=self.shuffle,
        )

        self.class_labels = class_labels

    def __iter__(self):
        for _, test_idx in self.stratified.split(self._dummpy_indices, self.class_labels):
            yield test_idx

    def __len__(self):
        return self.n_batches
