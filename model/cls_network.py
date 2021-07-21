from torch import nn
from torch_geometric.nn import global_mean_pool

from point_transformer import PointTransformerBlock, TransitionDown


class ClsPointTransformer(nn.Module):
    def __init__(self, in_features: int,
                 num_classes: int,
                 num_neighs: int = 16,
                 num_transformer_blocks: int = 1):
        super().__init__()
        out_features = 32

        self.init_mappin = nn.Linear(in_features, out_features)
        transformer_blocks = [PointTransformerBlock(in_out_features=out_features,
                                                    compress_dim=out_features // 2, num_neighbors=num_neighs)]

        classification_dim = None

        for i in range(1, num_transformer_blocks + 1):
            classification_dim = out_features * 2 * i
            transformer_blocks.extend(
                [TransitionDown(in_features=out_features, out_features=classification_dim,
                                num_neighbors=num_neighs, fps_sample_ratio=0.25),
                 PointTransformerBlock(in_out_features=classification_dim,
                                       compress_dim=out_features * i, num_neighbors=num_neighs)
                 ]
            )

        self.feature_extarctor = nn.Sequential(*transformer_blocks)
        self.classification_head = nn.Linear(classification_dim, num_classes)

    def forward(self, features, positions, batch):
        projected_features = self.init_mappin(features)
        new_features, _, new_batch = self.feature_extarctor((projected_features, positions, batch))
        encoding = global_mean_pool(new_features, new_batch)
        return self.classification_head(encoding)

    def predict_class(self, features, positions, batch):
        predicted_logits = self.forward(features, positions, batch)
        return predicted_logits.argmax(dim=-1)
