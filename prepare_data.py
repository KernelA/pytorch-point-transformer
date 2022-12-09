import hydra


@ hydra.main(config_path="configs", config_name="prepare_dataset", version_base="1.2")
def main(config):
    dataset = hydra.utils.instantiate(config.datasets)
    for stage in ("fit", "test", "validate", "predict"):
        dataset.setup(stage)


if __name__ == "__main__":
    main()
