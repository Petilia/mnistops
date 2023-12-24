import fire

from mnistops.export_model import export_2_onnx
from mnistops.infer import infer
from mnistops.run_server import (
    test_mlflow_server,
    test_triton_server,
    triton_sanity_check,
)
from mnistops.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export": export_2_onnx,
            "run_mlflow_infer": test_mlflow_server,
            "run_triton_infer": test_triton_server,
            "triton_sanity_check": triton_sanity_check,
        }
    )
