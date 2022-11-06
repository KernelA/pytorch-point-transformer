import pytest
import torch

from point_transformer.models import ClsPointTransformer

from .conftest import NUM_COORDS


@torch.no_grad()
def test_cls_networks(sample_batch):
    num_classes = 2
    simple_network = ClsPointTransformer(NUM_COORDS, num_classes, num_transformer_blocks=2)
    predicted_logits = simple_network((sample_batch.x, sample_batch.pos, sample_batch.batch))
    assert predicted_logits.shape == (sample_batch.num_graphs, num_classes)
