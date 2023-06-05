from pathlib import Path

from predict_modulation import predict_modulation
from q_param import q_param

q_param(Path("./fpfpp/q_param.txt"), 7, 2, 1, 1.5)

predict_modulation(Path("./fpfpp/"), 0, 0, 0, True)
