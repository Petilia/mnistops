## Project Description

The project was carried out as part of a training course on MLOps.

The project considers the problem of digit classification on the example of
MNIST dataset.

## Usage

The entry point that runs the scripts for training, inferencing, converting and
MLFlow running is [commands.py](./commands.py).

It is assumed that MLFlow is running locally:

```bash
mlflow server --host localhost --port 5000 --artifacts-destination ./outputs/mlflow_artifacts
```

After starting the server, you must execute the command (this assumes that the
python environment built by poetry is being used):

```bash
python3 commands.py
```

## Project Roadmap

[v1-dummy-torch](https://github.com/Petilia/mnistops/tree/v1-dummy-torch) -
Standard learning and inference on torch. Project configuration is hardcoded in
constants.py file.

[v2-hydra-torch](https://github.com/Petilia/mnistops/tree/v2-hydra-torch) -
Standard learning and inference on torch. Project configuration is set as
hydra-config.

[v3-hydra-lightning](https://github.com/Petilia/mnistops/tree/v3-hydra-lightning)
(current version). Learning and inference on the pytorch lightning framework.
Project configuration is set as hydra-config.
