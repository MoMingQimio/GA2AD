import os
import sys
import optparse
import traci
from sumolib import checkBinary
import numpy as np

# 检测是否已经添加环境变量
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print(os.environ['SUMO_HOME'])
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量 'SUMO_HOME'")

def main():
    sumoBinary = checkBinary('sumo-gui')  # 或者 'sumo-gui'
    traci.start([sumoBinary, '-c', 'SUMO-RL-ENVIRONMENT/gym_sumo/gym_sumo/envs/xml_files/test.sumocfg'])
    print("yes")
    # 在这里编写您的 Traci 代码
    while True:
        traci.simulationStep()
        if traci.simulation.getTime() > 720:
            break

    traci.close()

if __name__ == "__main__":
    main()