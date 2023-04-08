import hydra
from pytorch_lightning import seed_everything

import log_set


@ hydra.main(config_path="configs", config_name="prepare_dataset", version_base="1.2")
def main(config):
    seed_everything(config.params.seed)

    dataset = hydra.utils.instantiate(config.datasets)

    for stage in ("fit", "test", "validate", "predict"):
        try:
            dataset.setup(stage)
        finally:
            dataset.teardown(stage)


if __name__ == "__main__":
    main()
