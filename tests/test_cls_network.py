import pytest
import torch

from point_transformer.models import ClsPointTransformer


@torch.inference_mode()
def test_cls_networks(sample_batch):
    num_classes = 2
    simple_network = ClsPointTransformer(sample_batch.x.shape[1], num_classes, num_transformer_blocks=2)
    predicted_logits = simple_network((sample_batch.x, sample_batch.pos, sample_batch.batch))
    assert predicted_logits.shape == (sample_batch.num_graphs, num_classes)
