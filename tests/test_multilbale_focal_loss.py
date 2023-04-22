import torch

from training.losses import FocalLoss, focal_loss


def test_func(logits_multiclass_tensor, multiclass_labels):
    assert focal_loss(
        logits_multiclass_tensor,
        multiclass_labels,
        alpha=torch.full(
            (logits_multiclass_tensor.shape[1],),
             0.5
             )
        ).shape[0] == logits_multiclass_tensor.shape[0]


def test_loss_module(logits_multiclass_tensor, multiclass_labels):
    loss = FocalLoss(0.5)
    assert loss(logits_multiclass_tensor,
                multiclass_labels).shape[0] == logits_multiclass_tensor.shape[0]


def test_with_high_confidence(logits_multiclass_tensor, multiclass_labels):
    positive_class = torch.zeros_like(logits_multiclass_tensor)
    positive_class[list(range(positive_class.shape[0])), multiclass_labels] = 1000
    assert torch.allclose(focal_loss(positive_class, multiclass_labels,
                          alpha=0.5), torch.zeros(positive_class.shape[0]))


def test_jit(logits_multiclass_tensor, multiclass_labels):
    jitted_loss = torch.jit.script(FocalLoss(0.5))
    assert jitted_loss(logits_multiclass_tensor,
                       multiclass_labels).shape[0] == multiclass_labels.shape[0]
