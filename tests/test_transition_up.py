import torch
from point_transformer import TransitionUp

from .conftest import NUM_COORDS, NUM_POINTS


def test_transition_up(sample_batch):
    in_features = NUM_COORDS
    out_features = 16
    transition_up = TransitionUp(
        in_features=in_features, out_features=out_features)

    sample_size = max(NUM_POINTS // 2, 1)
    batch_size = sample_batch.x.shape[0]
    downsample_features = sample_batch.x[:, :sample_size, ...].contiguous()
    downsample_pos = sample_batch.pos[:, :sample_size, ...].contiguous()
    downsample_batch = sample_batch.batch.view(batch_size, NUM_POINTS)[
        :, :sample_size, ...].reshape(-1)

    upsampled_features, upsampled_positions, _ = transition_up((downsample_features, downsample_pos, downsample_batch,
                                                                sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert upsampled_features.shape[-1] == out_features
    assert upsampled_features.shape[:2] == sample_batch.x.shape[:2]
    assert upsampled_positions.shape == sample_batch.pos.shape
