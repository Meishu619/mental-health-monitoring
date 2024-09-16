import hydra

from omegaconf import (
    DictConfig, 
    OmegaConf
)

from train import experiment

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    experiment(cfg)

if __name__ == "__main__":
    main()