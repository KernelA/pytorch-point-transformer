from torch_geometric.data import Data


class BatchedData(Data):
    ...
    # def __cat_dim__(self, key, value):
    #     if key in ("pos", "x", "normal"):
    #         return None
    #     else:
    #         return super().__cat_dim__(key, value)

    # @property
    # def num_nodes(self):
    #     return self.pos.shape[0]
