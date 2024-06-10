Файлы - env_obst.py, train_sac.py, eval_sac_obst.py, wheeled_robot.py
также нужно загрузить ALOHA.usd из дирректрии ниже (aloha_env/aloha_rl) в дирректорию /assets/aloha (на сервере AIRI все загруженно)

обучение модели SAC на сервере AIRI -- ./python.sh standalone_examples/base_aloha_env/Aloha/train_sac.py 
просмотре с помощью omniverse.streaming -- ./python.sh standalone_examples/base_aloha_env/Aloha/train_sac.py -enable omni.kit.livestream.native --no-window

За удаленный просмотр отвечает блок кода 
//для удаленного просмотра
config = {
    "renderer": "RayTracedLighting",
    "headless": True,
    #headless: False,
    "multi_gpu": False, 
    #"active_gpu": gpu_to_use,
    "enable":"omni.kit.livestream.native"
}

на строках 8-15 env_obst.py 
