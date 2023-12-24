from functools import lru_cache
from pathlib import Path

import mlflow
import numpy as np
import onnx
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from PIL import Image
from scipy.special import softmax
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)

from .model import MNISTModel


def preprocess_image(image_path, cfg):
    width = height = cfg.data.img_size
    with Image.open(image_path) as image:
        image = image.convert("L")  # Convert image to grayscale
        image = image.resize((width, height), Image.LANCZOS)
        image_data = np.asarray(image).astype(np.float32)

        mean = cfg.data.img_mean
        std = cfg.data.img_std

        image_data = (image_data / 255 - mean) / std
        image_data = np.expand_dims(image_data, 0)
        image_data = np.expand_dims(image_data, 0)
    return image_data


@lru_cache(maxsize=1)
def get_triton_client(cfg: OmegaConf):
    return InferenceServerClient(url=cfg.triton.server_url)


def infer_triton(cfg: OmegaConf, image_path: str):
    triton_client = get_triton_client(cfg)
    inputs = []
    outputs = []
    input_data = preprocess_image(image_path, cfg)

    inputs.append(InferInput("IMAGES", input_data.shape, "FP32"))
    inputs[-1].set_data_from_numpy(input_data, binary_data=False)

    outputs.append(InferRequestedOutput("CLASS_PROBS", binary_data=True))

    # Triton Inference
    results = triton_client.infer(
        cfg.triton.model_name, inputs, outputs=outputs
    )

    probs = softmax(results.as_numpy("CLASS_PROBS"))

    # Print the class with the maximum probability
    print(f"Class with max probability: {np.argmax(probs)}")

    # Print probabilities of all classes
    for i, prob in enumerate(probs[0]):
        print(f"Class {i}: {100*prob}%")


def run_mlflow_server(cfg, image_path):
    model_name = f"{cfg.export.export_name}.onnx"
    filepath = Path(cfg.export.export_path) / model_name

    onnx_model = onnx.load_model(filepath)

    mlflow.set_tracking_uri(cfg.loggers.mlflow.tracking_uri)

    input_sample = np.random.randn(*cfg.export.input_sample_shape)

    with mlflow.start_run():
        model_info = mlflow.onnx.log_model(
            onnx_model,
            model_name,
            input_example=input_sample,
        )

    # load the logged model and make a prediction
    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = onnx_pyfunc.predict(preprocess_image(image_path, cfg))

    # Get softmax over predictions
    probs = softmax(predictions["CLASS_PROBS"], axis=1)

    # Print the class with the maximum probability
    print(f"Class with max probability: {np.argmax(probs)}")

    # Print probabilities of all classes
    for i, prob in enumerate(probs[0]):
        print(f"Class {i}: {100*prob}%")


def run_triton_sanity_check(cfg: OmegaConf, image_path: str):
    best_model_name = (
        Path(cfg.artifacts.checkpoint.dirpath)
        / cfg.loggers.experiment_name
        / "best.txt"
    )

    with open(best_model_name, "r") as f:
        best_checkpoint_name = f.readline()

    # torch inference
    model = MNISTModel.load_from_checkpoint(best_checkpoint_name).to("cpu")
    model.eval()

    input_data = preprocess_image(image_path, cfg)

    with torch.no_grad():
        torch_output = model(torch.from_numpy(input_data)).cpu().numpy()

    # triton inference
    triton_client = get_triton_client(cfg)
    inputs = []
    outputs = []

    inputs.append(InferInput("IMAGES", input_data.shape, "FP32"))
    inputs[-1].set_data_from_numpy(input_data, binary_data=False)

    outputs.append(InferRequestedOutput("CLASS_PROBS", binary_data=True))

    # Triton Inference
    results = triton_client.infer(
        cfg.triton.model_name, inputs, outputs=outputs
    )

    triton_output = results.as_numpy("CLASS_PROBS")

    assert np.allclose(torch_output, triton_output, atol=0.01)

    print(f"torch_output: {torch_output}, \ntriton_output: {triton_output}")
    print("Tensors are equal. Sanity check passed!")


def test_mlflow_server(
    image_path: str = "./img/sample.png",
    config_name: str = "config",
    config_path: str = "../configs",
    **kwargs,
):
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="mnistops-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )

    run_mlflow_server(cfg, image_path)


def test_triton_server(
    image_path: str = "./img/sample.png",
    config_name: str = "config",
    config_path: str = "../configs",
    **kwargs,
):
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="mnistops-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )

    infer_triton(cfg, image_path)


def triton_sanity_check(
    image_path: str = "./img/sample.png",
    config_name: str = "config",
    config_path: str = "../configs",
    **kwargs,
):
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="mnistops-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )

    run_triton_sanity_check(cfg, image_path)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py run_server`")
