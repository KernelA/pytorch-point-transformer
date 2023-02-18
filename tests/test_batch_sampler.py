import torch
from torch.utils.data import DataLoader, TensorDataset

from training.samplers import StratifiedBatchSampler


def test_batch_sampler(generator):
    x = torch.randn((36, 2), generator=generator)
    y = torch.arange(3, dtype=torch.long)

    y = y.repeat(x.shape[0] // y.shape[0])

    dataset = TensorDataset(x, y)

    dataloader = DataLoader(
        dataset,
        batch_sampler=StratifiedBatchSampler(
            batch_size=12, shuffle=True, class_labels=y.tolist()),
        pin_memory=True
    )

    first_batch = None

    assert len(dataloader) == 3

    for idx, (x, y) in enumerate(dataloader):
        if idx == 0:
            first_batch = y

    for idx, (x, y) in enumerate(dataloader):
        assert not torch.allclose(y, first_batch)
        break
