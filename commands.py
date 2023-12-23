import fire

from mnistops.export_model import export_2_onnx
from mnistops.infer import infer
from mnistops.run_server import run_server
from mnistops.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export": export_2_onnx,
            "run_server": run_server,
        }
    )
