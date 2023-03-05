import pathlib
import json
import os

import log_set
import hydra
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything, LightningDataModule, LightningModule, Trainer


@ hydra.main(config_path="configs", version_base="1.2")
def main(config):
    seed_everything(config.params.seed)
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datasets)
    datamodule.setup("fit")
    datamodule.setup("validate")

    assert "class_mapping" not in config, "Key 'class_mapping' already exits in the config"

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

    ckpt_path = None

    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update(OmegaConf.to_object(config))
        cls_config = {"class_mapping": class_mapping}
        trainer.logger.experiment.config.update(cls_config)

        run = trainer.logger.experiment

        if run.resumed:
            artifact = run.use_artifact(f"model-{run.id}:latest")
            datadir = artifact.download()
            ckpt_path = os.path.join(datadir, "model.ckpt")

    trainer.fit(model_trainer, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
