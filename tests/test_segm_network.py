import pytest
import torch

from point_transformer.models import SegmPointTransformer


# TODO: fix test
@torch.inference_mode()
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_segm_model(num_blocks, sample_batch):
    num_classes = 3
    model = SegmPointTransformer(sample_batch.x.shape[1],
                                 num_classes=3,
                                 num_paired_blocks_with_skip_connection=num_blocks,
                                 num_neighs=2)

    predicted_points_logits = model.forward_data(sample_batch)

    assert predicted_points_logits.shape[0] == sample_batch.x.shape[0]
    assert predicted_points_logits.shape[1] == num_classes
