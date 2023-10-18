import hydra
from omegaconf import DictConfig

from mnistops.infer import infer
from mnistops.train import train_model


@hydra.main(
    config_path="./mnistops/conf", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig):
    print(cfg)
    train_model(cfg)
    infer(cfg)


if __name__ == "__main__":
    main()
