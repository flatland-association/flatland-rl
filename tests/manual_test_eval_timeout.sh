#!/bin/bash

export FLATLAND_DEFAULT_COMMAND_TIMEOUT=5
export FLATLAND_DEFAULT_COMMAND_TIMEOUT=60
export FLATLAND_INITIAL_PLANNING_TIMEOUT=8

export AICROWD_TESTS_FOLDER=../submission-scoring/Envs/neurips2020_round1_v0

redis-cli KEYS "*" | grep -i flatland | xargs redis-cli DEL


# you need to create the envs in the folder
# best to delete all but 10 small ones
#gnome-terminal --title Service --window -- python -m flatland.evaluators.service --test_folder  ../submission-scoring/Envs/neurips2020_round1_v0/   
xterm -title Service -hold -e "python -m flatland.evaluators.service --test_folder  ../submission-scoring/Envs/neurips2020_round1_v0/   " &

#gnome-terminal --title test_eval_timeout  --window -- python tests/test_eval_timeout.py
xterm -title test_eval_timeout  -hold -e "python tests/test_eval_timeout.py" &



