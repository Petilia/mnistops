# 3d Homework report

<!--
Форма текстового отчета:
- Ваша системная конфигурация
    - OS и версия
    - Модель CPU
    - Количество vCPU и RAM при котором собирались метрики
- Описание решаемой задачи
- Описание структуры вашего model_repository (в формате “$ tree”)
- Секция с метриками по throughput и latency которые вы замерили до всех оптимизаций и после всех оптимизаций
- Объяснение мотивации выбора или не выбора той или иной оптимизации -->

## System configuration

| OS           | CPU                                     | vCPU | RAM    |
| ------------ | --------------------------------------- | ---- | ------ |
| Ubuntu 20.04 | Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz | 8    | 32 GiB |

## Task description

Задача состоит в классификации датасета MNIST на 10 классов.

## Model repository structure

```bash
triton/model_repository
└── mnist_classifier
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## Metrics

Делаем pull sdk-образа:

```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-py3-sdk
```

Запускаем контейнер:

```bash
docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:23.10-py3-sdk
```

В контейнере запускаем perf_analyzer (параметры меняем в зависимости от
эксперимента)

```bash
perf_analyzer -m mnist_classifier --percentile=95 -u localhost:8500 --shape=IMAGES:1,1,28,28 --concurrency-range 1,2,4,8 --measurement-interval 15000
```

В результате экспериментов получем следующую табличку:

| Optimization          | Conq Range | **Throughput, fps** | **Latency, ms** | Details |
| --------------------- | ---------- | ------------------- | --------------- | ------- |
| No                    | $1$        | $417$               | $2.1$           |         |
| No                    | $2$        | $587$               | $4.1$           |         |
| No                    | $4$        | $632$               | $45.6$          |         |
| No                    | $8$        | $754$               | $50.4$          |         |
|                       |            |                     |                 |
| More instances (2)    | $1$        | $607$               | $1.9$           |         |
| More instances (2)    | $2$        | $642$               | $4.1$           |         |
| More instances (2)    | $4$        | $1066$              | $8.6$           |         |
| More instances (2)    | $8$        | $981$               | $42.7$          |         |
|                       |            |                     |                 |
| More instances (4)    | $1$        | $393$               | $2.3$           |         |
| More instances (4)    | $2$        | $379$               | $41.2$          |         |
| More instances (4)    | $4$        | $444$               | $48.4$          |         |
| More instances (4)    | $8$        | $698$               | $50.0$          |         |
|                       |            |                     |                 |
| Dynamic Batching      | $1$        | $577$               | $2.1$           |         |
| Dynamic Batching      | $2$        | $795$               | $3.9$           |         |
| Dynamic Batching      | $4$        | $898$               | $11.8$          |         |
| Dynamic Batching      | $8$        | $708$               | $51.8$          |         |
|                       |            |                     |                 |
| OpenVino              | $1$        | $2001$              | $0.6$           |         |
| OpenVino              | $2$        | $1692$              | $2.9$           |         |
| OpenVino              | $4$        | $2380$              | $3.5$           |         |
| OpenVino              | $8$        | $2229$              | $8.1$           |         |
|                       |            |                     |                 |
| DB + OpenVino + MI(2) | $1$        | $2006$              | $1.9$           |         |
| DB + OpenVino + MI(2) | $2$        | $1864$              | $2.6$           |         |
| DB + OpenVino + MI(2) | $4$        | $2563$              | $3.0$           |         |
| DB + OpenVino + MI(2) | $8$        | $2629$              | $6.4$           |         |

Можно оптимизировать инференс добавлением дополнительных инстансов модели (в
стандартном режиме инстанс 1, а добавление новых может увеличить throughput и
уменьшить latency).

```
instance_group [ { count: 4 }]
```

Также можно оптимизировать инференс с помощью инференса на OpenVino

```
optimization { execution_accelerators {
    cpu_execution_accelerator : [ {
      name : "openvino"
    }]
  }}
```

Также можно оптимизировать dynamic_batching, который позволяет увеличивать
размер батча, если в очереди накопилось много задач.

```
dynamic_batching {}
```

## Conclusion

- Наибольший прирост с точки зрения throughput дает инференс на OpenVino (что
  неудивительно, так как он предназначен для инференса на cpu)

- Увеличение числа инстансов дает прирост относительно базового инференса при
  числе инстансов в 2, при 4 инстансах результаты хуже

- Dynamic Batching также дает прирост относительного базового инференса, но в
  основном в throughput

- В сумме все оптимизации вместе работают лучше, чем по отдельности, но главный
  вклад вносит OpenVino

На основании последовательного подбора выбрана следующая конфигурация инференса:

```
instance_group [ { count: 2 }]
optimization { execution_accelerators {
    cpu_execution_accelerator : [ {
      name : "openvino"
    }]
  }}
dynamic_batching {}
```
