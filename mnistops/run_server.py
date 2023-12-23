from pathlib import Path

import mlflow
import numpy as np
import onnx
from hydra import compose, initialize
from PIL import Image
from scipy.special import softmax


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


def run_server(
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


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py run_server`")
