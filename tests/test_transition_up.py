import torch
from point_transformer import TransitionUp

from .conftest import NUM_COORDS


def test_transition_up(sample_batch):
    in_features = NUM_COORDS
    out_features = 16
    transition_up = TransitionUp(
        in_features=in_features, out_features=out_features)

    downsample_features = sample_batch.x[::5, ...]
    downsample_pos = sample_batch.pos[::5, ...]
    downsample_batch = sample_batch.batch[::5, ...]

    upsampled_features, upsampled_positions, new_batch = transition_up((downsample_features, downsample_pos, downsample_batch,
                                                                        sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert upsampled_features.shape[-1] == out_features
    assert upsampled_features.shape[0] == sample_batch.x.shape[0]
    assert upsampled_positions.shape == sample_batch.pos.shape
