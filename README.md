## Project Description

Проект выполнен в рамках учебного курса по MLOps.

Рассматривается задача классификации цифр на датасете MNIST

## Usage [train + dummy inference + export]

Входной точкой для всего проекта является файл [commands.py](./commands.py).

При работе предполагается, что локально запущен сервер MLFlow:

```bash
mlflow server --host localhost --port 5000 --artifacts-destination ./outputs/mlflow_artifacts
```

После того, как сервер запущен, можно использовать команды из commands.py

Обучение:

```bash
python3 commands.py train
```

Стандартный torch-инференс:

```bash
python3 commands.py infer
```

Экспорт модели в onnx (можно использовать предобученные веса, указав в конфиге
pretrained.use=true или передав это в командной строке при запуске, т.к.
используется fire и он перезапишет этот параметр автоматически):

```bash
python3 commands.py export
# or
python3 commands.py export --pretrained.use true

```

## Usage [inference]

В качестве аргумента для инференса во всех случаях можно передать путь к
интересуюoему изображению через командную строке --image-path <path/to/image>.
Если не передавать, используется дефолтное изображение.

Инференс на MLFlow:

```bash
python3 commands.py run_mlflow_infer
# or
python3 commands.py run_mlflow_infer --image-path <path/to/image>
```

Перед инференсом на Triton необходимо запустить контейнер:

```
cd triton
docker-compose up
```

Инференс на Triton:

```bash
python3 commands.py run_triton_infer
# or
python3 commands.py run_triton_infer --image-path <path/to/image>
```

Проверка корректности triton-инференса (происходит сравнение выходов модели на
torch-е и на triton-е):

```bash
python3 commands.py triton_sanity_check
# or
python3 commands.py triton_sanity_check --image-path <path/to/image>
```

## HW3 Report

В [HW3_report.md](./HW3_report.md) лежит отчет о 3 домашнем задании с метриками
и выводами по оптимизации triton-инференса.
