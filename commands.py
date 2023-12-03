import hydra
from omegaconf import DictConfig

from mnistops.export_model import export_2_onnx
from mnistops.infer import infer
from mnistops.run_server import run_server
from mnistops.train import train


@hydra.main(config_path="./configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    train(cfg)
    infer(cfg)
    export_2_onnx(cfg)
    run_server(cfg, "./img/sample.png")


if __name__ == "__main__":
    main()
