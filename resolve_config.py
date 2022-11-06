import os

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="configs", version_base="1.2")
def main(config):
    os.makedirs(config.base_exp_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.base_exp_dir, "config.yaml"), resolve=True)


if __name__ == "__main__":
    main()
