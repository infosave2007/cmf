# Vulkan на арендованном поде

Проверенная последовательность для headless-контейнеров с NVIDIA GPU. Она
многократно отработана на RTX 5090 и RTX PRO 6000. Для wgpu-движка CMF на
арендованных подах следует выбирать RTX-класс: H100/A100 часто выдаются без
необходимой `graphics`-capability, и исправить это изнутри контейнера нельзя.

## 1. Установить GLVND и Vulkan

Пакеты ставятся **безусловно**, без предварительной проверки через `dpkg`:

```sh
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools
```

В контейнерном образе обычно уже доступно ядро драйвера NVIDIA, но отсутствует
GLVND-прослойка, через которую NVIDIA ICD предоставляет Vulkan. Условная
проверка вида `dpkg -l | grep ...` может молча пропустить нужную установку.

## 2. Всегда задать `XDG_RUNTIME_DIR`

```sh
export XDG_RUNTIME_DIR=/tmp
```

Переменная нужна не только при проверке: её следует задавать при **каждом**
запуске `bench`, `run` и `serve`. Без неё загрузчик Vulkan может завершиться с
ошибкой ещё до обнаружения адаптера.

## 3. Проверить адаптер

```sh
XDG_RUNTIME_DIR=/tmp vulkaninfo --summary | grep deviceName
```

Команда должна вывести строку с именем GPU. После этого движку больше ничего не
нужно:

```sh
CMF_GPU=1 XDG_RUNTIME_DIR=/tmp cortiq run model.cmf
CMF_GPU=1 XDG_RUNTIME_DIR=/tmp cortiq bench model.cmf
CMF_GPU=1 XDG_RUNTIME_DIR=/tmp cortiq serve model.cmf
```

## Диагностика и известные ловушки

- `0 adapters` или `Found no drivers` обычно означают отсутствие GLVND, а не
  проблему DRM. Права на `/dev/dri/renderD128` (`660` и подобные) в этом случае
  — ложный след; начинать диагностику с DRM не нужно.
- Минорная версия userspace-библиотек NVIDIA должна совпадать с версией модуля
  драйвера. Версию модуля показывает
  `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. Например,
  userspace `580.178` при модуле `580.126` несовместим; соседняя доступная в
  APT минорная версия не подходит.
- H100/A100-поды — ненадёжный вариант для Vulkan. Даже точные библиотеки из
  официального драйверного `.run`-файла (`NVIDIA-*.run --extract-only`), ручное
  копирование `libGLX_nvidia`, `libnvidia-glcore`, `libnvidia-glsi`,
  `libnvidia-glvkspirv`, `libnvidia-rtcore`, `libnvidia-nvvm`,
  `libnvidia-tls`, `libnvidia-gpucomp`, ICD JSON и последующий `ldconfig` не
  помогают, если контейнерный runtime не предоставил `graphics`-capability.
  Характерный итог — `vkCreateInstance: Found no drivers`. Из контейнера это не
  чинится; нужно сменить инстанс на RTX 5090 или RTX PRO 6000.
- Первый прогон процесса на новой модели компилирует шейдеры, поэтому скорость
  может быть ниже установившейся в 5–10 раз. Скорость измеряется со второго
  прогона либо через `cortiq bench`, где прогрев вынесен за окно измерения.
  Кэш пайплайнов `cortiq-pipelines-*.bin` сохраняется рядом с моделью и
  переживает перезапуски.
