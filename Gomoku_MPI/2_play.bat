setlocal
cd /d %~dp0

call activate tensorflow-gpu_1-15

python human_play.py
