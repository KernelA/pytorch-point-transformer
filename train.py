import pathlib
import json

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, LightningDataModule, LightningModule, Trainer


@ hydra.main(config_path="configs", version_base="1.2")
def main(config):
    seed_everything(config.params.seed)
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datasets)
    datamodule.setup("fit")
    datamodule.setup("validate")

    class_mapping = datamodule.get_class_mapping()

    config.model.num_classes = len(class_mapping)

    model_trainer: LightningModule = hydra.utils.instantiate(
        config.model_trainer,
        model=hydra.utils.instantiate(config.model),
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        cls_mapping=class_mapping)

    exp_dir = pathlib.Path(config.base_exp_dir) / config.exp_dir
    exp_dir.mkdir(exist_ok=True, parents=True)

    config_dir = exp_dir / config.config_dir
    config_dir.mkdir(exist_ok=True)
    OmegaConf.save(config=config, f=config_dir / "config.yaml", resolve=True)

    with open(config_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f)

    checkpoint_dir = exp_dir / config.checkpoint_dir
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    log_dir = exp_dir / config.log_dir
    log_dir.mkdir(exist_ok=True, parents=True)

    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model_trainer, datamodule=datamodule)


if __name__ == "__main__":
    main()
