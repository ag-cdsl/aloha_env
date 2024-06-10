# Решение задачи навигации с препятствием с помощью RL алогритма SAC 
## На сервере:
### Запуск обучения:
./python.sh standalone_examples/base_aloha_env/Aloha/train_sac.py 
### Тестирование модели:
./python.sh standalone_examples/base_aloha_env/Aloha/eval_sac_obst.py -enable omni.kit.livestream.native --no-window

## Локально:
### Запуск обучения:
./python.sh stable_baseline_aloha/train_sac.py 
## Тестирование модели:
./python.sh stable_baseline_aloha/eval_sac_obst.py

## При работе на локальном компьютере:
нужно загрузить ALOHA.usd из дирректрии ниже (aloha_env/aloha_rl) в дирректорию stable_baseline_aloha/assets/aloha (на сервере AIRI все загруженно)

За удаленный просмотр отвечает блок кода на строках 8-15 в файле /tasks/env_obst.py 

config = {
    "renderer": "RayTracedLighting",
    
    "headless": True,
    
    #headless: False,
    
    "multi_gpu": False,
    
    #"active_gpu": gpu_to_use,
    
    "enable":"omni.kit.livestream.native"
}

 ### Нужно: 
 - закомментировать строчки "headless": True и "enable":"omni.kit.livestream.native"
 - раскомментировать #headless: False,

на строках 8-15 env_obst.py 

## Как подключиться к серверу?
1) Нужно установить (с включенным vpn)  omniverse streaming client

2) подключаемся по ssh: (IP: 141.101.151.104, port: 44444)    
    ssh -p 44444 ccm_team@141.101.151.104
