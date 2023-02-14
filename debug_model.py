from omegaconf import OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
import log_set
import torch
from tqdm.auto import tqdm

if __name__ == "__main__":
    config = OmegaConf.load("./exp/modelnet10/config.yaml")

    datamodule: LightningDataModule = instantiate(config.datasets)
    datamodule.setup("fit")
    # datamodule.setup("validate")

    class_mapping = datamodule.get_class_mapping()

    config.model.num_classes = len(class_mapping)
    model = instantiate(config.model)

    device = torch.device("cuda")

    dataloader = datamodule.train_dataloader()
    model.to(device)

    with torch.autocast(device.type):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = model.forward_data(batch.to(device))
