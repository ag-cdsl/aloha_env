# Isaac_Sim_Rearrangement_task

![image_2024-03-14_15-15-08](https://github.com/tttonyalpha/Isaac_Sim_Rearrangement_task/assets/79598074/a90f9af6-204d-41a7-91ef-310a71bbcc4d)


## Запуск

1. Установите docker контейнер с Isaac Sim 2022.2.1
2. Запустите контейнер при помощи команды, при необходимости, примонтируйте нужные дериктории 
```
docker run --name isaac-sim --entrypoint bash -it --gpus device=0 -e "ACCEPT_EULA=Y"
--rm --network=host -e "PRIVACY_CONSENT=Y" -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw
-v ~/docker/isaac-sim/documents:/root/Documents:rw nvcr.io/nvidia/isaac-sim:2022.2.1
```
3. Склонируйте репозиторий по пути: `/isaac-sim/extension_examples/` либо в любую удобную для вас папку
4. Запустите скрипт при помощи команды: `/isaac-sim/python.sh /isaac-sim/standalone_examples/husky/husky_control.py `



## Структура проекта
Этот проект имеет следующую структуру:
- `/assets/husky`: модели для платформы husky и робо-руки UR5 
- `/controllers`: контроллеры для манипуляции UR5
- `/rmpflow`: базовый контроллер для стратегии движения UR5 и необходимые конфиги
- `/scenes`: кастомные сцены/среды для задачи rearrangement
- `/husky_control.py`: скрипт для запуска сценария: подехать к столу, схватить кубик и отвезти его к другому столу 
