import os
import gym
from build.lib.traci import vehicle
from gym import spaces

import numpy as np 
import sys
import math
import time

from gym_sumo.envs import env_config as c
from gym_sumo.envs.RuleBasedDriverModel import DriverModel as RB
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci
#import sumolib
## add each lane density and mean_speed ##
state_space_list = ['lane_index', 'ego_speed', 'ego_acc', 'ego_heading_angle',
                            'ego_dis_to_leader', 'leader_speed', 'leader_acc',
                            'ego_dis_to_follower', 'follower_speed', 'follower_acc',
                            'dis_to_left_leader', 'left_leader_speed', 'left_leader_acc',
                            'dis_to_right_leader', 'right_leader_speed', 'right_leader_acc',
                            'dis_to_left_follower', 'left_follower_speed', 'left_follower_acc',
                            'dis_to_right_follower', 'right_follower_speed', 'right_follower_acc'
                            ]
for i in range(c.NUM_OF_LANES):
    state_space_list.append("lane_" + str(i) + "_mean_speed")
    state_space_list.append("lane_" + str(i) + "_density")
def creat_observation():

    #print(state_space_list)
    state_space_low = np.array([c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE, -c.HEADING_ANGLE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                -c.RL_SENSING_RADIUS, c.RL_MIN_SPEED_LIMIT, -c.RL_DCE_RANGE,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY
                                ])

    state_space_high = np.array([c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE, c.HEADING_ANGLE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_SENSING_RADIUS, c.RL_MAX_SPEED_LIMIT, c.RL_ACC_RANGE,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY,
                                 c.RL_MIN_SPEED_LIMIT, c.MIN_LANE_DENSITY,
                                 c.RL_MAX_SPEED_LIMIT, c.MAX_LANE_DENSITY
                                 ])

    
    obs = spaces.Box(low=state_space_low,high=state_space_high,dtype=np.float64)
    return obs

class SumoEnv(gym.Env):
    """docstring for SumoEnv"""
    metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None):
        super(SumoEnv, self).__init__()
        self.action_list = np.arange(-c.RL_DCE_RANGE, c.RL_ACC_RANGE + c.ACC_INTERVAL, c.ACC_INTERVAL)
        self.action_space = spaces.MultiDiscrete([6,len(self.action_list)+3])
        self.observation_space = creat_observation()

        ## class variable
        self.ego = c.EGO_ID
        self.num_of_lanes = c.NUM_OF_LANES
        self.max_speed_limit = c.RL_MAX_SPEED_LIMIT
        self.min_speed_limit = c.RL_MIN_SPEED_LIMIT
        self.is_collided = False
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.rbm = RB()
        self.running_distance = 0
        self.running_time = 0
        self.k = c.activation_steepness

        self.excepted_risk = [0,0,0,0,0,0]
        self.surrounding_vehicles = ["","","","","",""]
        self.total_risk = 0
        self.ABV = 7

        # print(self.render_mode)






    def _getInfo(self):
        return {"current_episode":0}

    def _startSumo(self):
        sumoBinary = "sumo"
        if self.render_mode=="human":
            sumoBinary = "sumo-gui"
        sumoCmd = [sumoBinary,
                   "-c", "sumo_networks/test.sumocfg",
                   "--lateral-resolution","3.8",
                   "--start", "true",
                   "--quit-on-end", "true",
                   "--no-warnings","True",
                   "--no-step-log", "True",
                   "--collision.mingap-factor", "0.0",
                   "--collision.action", "warn",
                   #"--log","/runs/sumo_log.txt",
                   ]

        traci.start(sumoCmd)


    @staticmethod
    def seed(seed = None):
        np.random.seed(seed)





    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.is_collided = False
        self._startSumo()
        self._warmup()
        #
        index = np.array(np.random.randint(0, self.num_of_lanes))
        obs = np.array(self.observation_space.sample())

        
        obs =np.insert(obs, 0, index)

        #info = self._getInfo()
        # info = ["","","","","",""]
        #traci.vehicle.setSpeed(self.ego, 20)
        # return obs, info, np.zeros((1,6))
        return obs

    def _getCloseLeader(self, leaders):
        if len(leaders) <= 0:
            return "", -1
        min_dis = float('inf')
        current_leader = None,
        for leader in leaders:
            leader_id, dis = leader
            if dis < min_dis:
                current_leader = leader_id
                min_dis = dis
        return (current_leader, min_dis)

    def _getLaneDensity(self):
        road_id = traci.vehicle.getRoadID(self.ego)
        density = []
        mean_speed = []
        for i in range(self.num_of_lanes):
            density.append(len(traci.lane.getLastStepVehicleIDs(road_id+"_"+str(i))))
            mean_speed.append(traci.lane.getLastStepMeanSpeed(road_id+"_"+str(i)))
        return density, mean_speed

    def _get_rand_obs(self):
        index = np.array(np.random.randint(0, self.num_of_lanes))
        obs = np.array(self.observation_space.sample())
        obs = np.insert(obs, 0, index)
        #obs = np.array(self.observation_space.sample())
        return obs

    def get_observation(self,veh_id):
        if not self.isVehRunninng(veh_id):
            return self._get_rand_obs() ,["","","","","",""]
        #Basic Information
        lane_index = traci.vehicle.getLaneIndex(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        acceleration = traci.vehicle.getAccel(veh_id)
        heading_angle = (traci.vehicle.getAngle(veh_id) -90.0)/c.HEADING_ANGLE


        leader = traci.vehicle.getLeader(veh_id)
        if leader is not None:
            leader_id, l_distance = leader
        else:
            leader_id, l_distance = "", -1
        leader_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else 0.01
        leader_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
        left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(veh_id, blockingOnly=True))
        left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else 0.01
        left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6
        if left_leader == "":
            left_l_dis = -1

        right_leader, right_l_dis = traci.vehicle.getRightLeaders(veh_id, blockingOnly=True)
        right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else 0.01
        right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6
        if right_leader == "":
            right_l_dis = -1

        follower = traci.vehicle.getFollower(veh_id)
        if follower is not None:
            follower_id, f_distance = follower
        else:
            follower_id, f_distance = "", -1
        follower_speed = traci.vehicle.getSpeed(follower_id) if follower_id != "" else 0.01
        follower_acc = traci.vehicle.getAccel(follower_id) if follower_id != "" else -2.6

        left_follower, left_f_dis =traci.vehicle.getLeftFollowers(veh_id, blockingOnly=True)
        left_follower_speed = traci.vehicle.getSpeed(left_follower) if left_follower != "" else 0.01
        left_follower_acc = traci.vehicle.getAccel(left_follower) if left_follower != "" else -2.6
        if left_follower == "":
            left_f_dis = -1

        right_follower, right_f_dis = traci.vehicle.getRightFollowers(veh_id, blockingOnly=True)
        right_follower_speed = traci.vehicle.getSpeed(right_follower) if right_follower != "" else 0.01
        right_follower_acc = traci.vehicle.getAccel(right_follower) if right_follower != "" else -2.6
        if right_follower == "":
            right_f_dis = -1


        states = [lane_index, speed, acceleration,heading_angle,
                  l_distance, leader_speed, leader_acc,
                  left_l_dis, left_l_speed, left_l_acc,
                  right_l_dis, right_l_speed, right_l_acc,
                  f_distance, follower_speed, follower_acc,
                  left_f_dis, left_follower_speed, left_follower_acc,
                  right_f_dis, right_follower_speed, right_follower_acc]

        density, mean_speed = self._getLaneDensity()
        for i in range(self.num_of_lanes):
            states.append(density[i])
            states.append(mean_speed[i])
        observations = np.array(states)
        # surrounding vehicle id list
        surrounding_vehicles =[leader_id, follower_id, left_leader,left_follower, right_leader, right_follower]

        return observations , surrounding_vehicles



    def _get_observation(self):
        if not self._isEgoRunning():
            return self._get_rand_obs()

        ego_speed = traci.vehicle.getSpeed(self.ego)
        ego_acceleration = traci.vehicle.getAccel(self.ego)
        ego_leader = traci.vehicle.getLeader(self.ego)
        if ego_leader is not None:
            leader_id, distance = ego_leader
        else:
            leader_id, distance = "", -1
        l_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else -0.01
        l_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
        left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=True))
        left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else -0.01
        left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6

        right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=True))
        right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else -0.01
        right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6

        states = [ego_speed, ego_acceleration, distance, l_speed, l_acc, left_l_dis, left_l_speed, left_l_acc,
            right_l_dis, right_l_speed, right_l_acc]
        density, mean_speed = self._getLaneDensity()
        for i in range(self.num_of_lanes):
            states.append(density[i])
            states.append(mean_speed[i])

        observations = np.array(states)
        return observations

    # def _applyAction(self, action):
    #     if self._isEgoRunning() == False:
    #         return
    #     current_lane_index = traci.vehicle.getLaneIndex(self.ego)
    #
    #     if action == 0:
    #         # do nothing: stay in the current lane
    #         pass
    #     elif action == 1:
    #
    #         target_lane_index = min(current_lane_index+1, self.num_of_lanes-1)
    #         traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)
    #
    #     elif action == 2:
    #
    #         target_lane_index = max(current_lane_index-1, 0)
    #         traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)
    #
    #     elif action == 3:
    #         traci.vehicle.setAcceleration(self.ego,0.2, 0.1)
    #     elif action == 4:
    #         traci.vehicle.setAcceleration(self.ego, -4.5, 0.1)


    def _collision_reward(self):
        collide_vehicles = traci.simulation.getCollidingVehiclesIDList()
        if self.ego in collide_vehicles:
            self.is_collided = True
            return -1
        return 0.0
    def _efficiency(self):
        speed = traci.vehicle.getSpeed(self.ego)
        if speed < self.min_speed_limit:
            return (speed-self.min_speed_limit)/(self.max_speed_limit-self.min_speed_limit)
        if speed > self.max_speed_limit:
            return (self.max_speed_limit-speed)/(self.max_speed_limit-self.min_speed_limit)
        return speed/self.max_speed_limit
    def _lane_change_reward(self,action):
        if action ==(len(self.action_list) + 1) or action == (len(self.action_list) + 2):
            return -1.0
        return 0

    def _reward(self, action):
        c_reward = self._collision_reward()
        if self.is_collided or self._isEgoRunning()==False:
            return c_reward
        return (10 * c_reward) + self._efficiency() + self._lane_change_reward(action)

    def _BVreward(self, rl_action, rb_action,index):
        c_reward = self._collision_reward()
        a_reward = 0
        if rl_action == rb_action:
            a_reward = -1

        return (-20 * c_reward) + 1 * a_reward + 0.5 * self.excepted_risk[index]
    def Activation_Function(self,risk_bv,risk_total):
        activation = 1/(1+math.exp(self.k*(risk_bv-risk_total)))
        return activation
    
    def step(self, final_actions):
        self.ABV = 7
        # AV_action, Veh_id, RL_action, epsilon = final_actions
        # AV_action, Veh_id,BV_candidate_index, RL_action= final_actions
        AV_action,BV_candidate_index, RL_action= final_actions
        Veh_id = self.surrounding_vehicles[BV_candidate_index]
        # observation, surrounding_vehicles = self.get_observation(self.ego)
        # excepted_risk, total_risk = self.risk_assessment(observation, surrounding_vehicles)
        # activation = self.Activation_Function(self.excepted_risk[BV_candidate_index],self.total_risk)
        # print(self.excepted_risk)
        # print(self.total_risk)
        #print(activation)
        
        self._applyAction(self.ego,AV_action)
        
        
        
        
        RB_action =  self.rbm.get_action(self.get_observation(Veh_id))
        Adversarial_flag = False
        # #print(activation)
        # if activation > 0 and activation < 0.8:
        #     Adversarial_flag = True
        #     if Veh_id != "":
        #         traci.vehicle.setSpeedMode(Veh_id, 32)
        #         traci.vehicle.setLaneChangeMode(Veh_id, 1109)
        #         # The lane change mode when controlling BV is 0b010001010101 = 1109
        #         # which means that the laneChangeModel may execute all changes unless in conflict
        #         # with TraCI.Requests from TraCI are handled urgently without consideration for safety constraints.
        #
        #         #traci.vehicle.setLaneChangeMode(Veh_id, 1621)
        #     BV_final_action = RL_action
        # else:
        #     BV_final_action = RB_action
        # self._applyAction(Veh_id,BV_final_action)

        if self.total_risk > 0.5 and self.total_risk < 1.5:
            Adversarial_flag = True
            self.ABV = BV_candidate_index
            # if Veh_id != "":
            #     # traci.vehicle.setSpeedMode(Veh_id, 32)
            #     # traci.vehicle.setLaneChangeMode(Veh_id, 1109)
            #     vehicle_list = traci.vehicle.getIDList()
            #
            #     
            #     if Veh_id in vehicle_list:
            #
            #         traci.vehicle.setSpeedMode(Veh_id, 0)
            #         traci.vehicle.setLaneChangeMode(Veh_id, 0)
                # The lane change mode when controlling BV is 0b010001010101 = 1109
                # which means that the laneChangeModel may execute all changes unless in conflict
                # with TraCI.Requests from TraCI are handled urgently without consideration for safety constraints.

                #traci.vehicle.setLaneChangeMode(Veh_id, 1621)
            BV_final_action = RL_action
            self._applyAction(Veh_id, BV_final_action)
        # else:
        #     BV_final_action = RB_action

        self.running_distance = traci.vehicle.getDistance(self.ego)
        # self.running_time = traci.simulation.getTime()
        # traci.simulationStep(self.running_time + 1)
        traci.simulationStep()
        # traci.gui.screenshot("View #0", "snap_shot_images/" + str(k) + "-" + str(i) + ".png")
        # time.sleep(0.1)

        # if Veh_id != "":
        #     traci.vehicle.setSpeedMode(Veh_id, 31)
        #     traci.vehicle.setLaneChangeMode(Veh_id, 1621)
        reward = self._reward(AV_action)
        observation, surrounding_vehicles = self.get_observation(self.ego)
        self.surrounding_vehicles = surrounding_vehicles
        BV_reward = self._BVreward(RL_action, RB_action,BV_candidate_index)
        excepted_risk, total_risk = self.risk_assessment(observation,surrounding_vehicles)
        self.excepted_risk = excepted_risk
        self.total_risk = total_risk

        done = self.is_collided or (self._isEgoRunning()==False)

        collision_flag = self.is_collided

        if done == False and traci.simulation.getTime() > 360:
            done = True
        
        other_information = {"BV_reward" : BV_reward,
                             # "surrounding_vehicles" : surrounding_vehicles,
                             # "excepted_risk" :  excepted_risk,
                             "total_risk" : total_risk,
                             "collision_flag" : collision_flag,
                             "Adversarial_flag" :Adversarial_flag}

        #other_information = (BV_reward,surrounding_vehicles, excepted_risk, total_risk, collision_flag, Adversarial_flag)
        return observation, reward, done, other_information

    def isVehRunninng(self, veh_id):
        """
            Check if a vehicle is running on any of the specified edges.

            Parameters:
            self (object): The instance of the class.
            veh_id (str): The ID of the vehicle to check.

            Returns:
            bool: True if the vehicle is running on any of the edges E0, E1, or E2, False otherwise.
        """
        v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
        v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
        v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
        if veh_id in v_ids_e0 or veh_id in v_ids_e1 or veh_id in v_ids_e2:
            return True
        return False



    def _isEgoRunning(self):
        v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
        v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
        v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
        if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
            return True
        return False

    def _warmup(self):
        while True:
            v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
            v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
            v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
            if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
                traci.vehicle.setLaneChangeMode(self.ego,0)
                #traci.vehicle.setSpeedMode(self.ego, 32)
                traci.vehicle.setLaneChangeMode(self.ego, 1109)
                #traci.vehicle.setLaneChangeMode(self.ego, 0)
                return True
            traci.simulationStep()


    def closeEnvConnection(self):
        traci.close()

    def move_gui(self):

        if self.render_mode == "human":
            x, y = traci.vehicle.getPosition('av_0')
            traci.gui.setOffset("View #0",x-23.0,108.49)




    def risk_assessment(self,observation,surrounding_vehicles):


        state = dict(zip(state_space_list, observation))
        state['lane_index'] = int(state['lane_index'])
        SSM = [self.calc_safety_surrogate_metric(state['ego_speed'], state['leader_speed'], state['ego_dis_to_leader']),
               self.calc_safety_surrogate_metric(state['follower_speed'], state['ego_speed'],state['ego_dis_to_follower']),
               self.calc_safety_surrogate_metric(state['ego_speed'], state['left_leader_speed'],state['dis_to_left_leader']),
               self.calc_safety_surrogate_metric(state['left_follower_speed'], state['ego_speed'],state['dis_to_left_follower']),
               self.calc_safety_surrogate_metric(state['ego_speed'], state['right_leader_speed'],state['dis_to_right_leader']),
               self.calc_safety_surrogate_metric(state['right_follower_speed'], state['ego_speed'],state['dis_to_right_follower'])]

        prob_norm = self.calc_risk_prob(SSM)
        
        
   



        p_s = 1
        p_d = 1
        for i in range(6):
            p_d = p_d * (1-prob_norm[2,i])
            p_s = p_s * prob_norm[0,i]
        PD = 1 - p_d
        PS = p_s
        PA = 1 - PD -PS
        total_prob = np.array([PS, PA, PD])
        #print(lane_risk_prob)
        #risk_level = np.argmax(prob_norm, axis=0)
        excepted_risk = np.dot(np.array([0, 1, 2]),prob_norm) 

        total_risk = np.dot(np.array([0, 1, 2]),total_prob)

        return excepted_risk, total_risk



    def calc_risk_prob(self, SSM):

        dangerous_prob = self.calc_prob(SSM, 'dangerous')
        attentive_prob = self.calc_prob(SSM, 'attentive')
        safety_prob = self.calc_prob(SSM, 'safety')
        risk_prob = np.array([safety_prob, attentive_prob, dangerous_prob])
        c_sums = risk_prob.sum(axis=0)
        c_sums[c_sums == 0] = 1e-10 

        prob_norm = risk_prob / c_sums[np.newaxis,: ]
       
        #prob_norm = risk_prob / np.linalg.norm(risk_prob, axis=0, keepdims=True)
        return prob_norm

    def calc_prob(self, SSM, prob_type):
        prob = []
        for i in SSM:
            if prob_type == 'dangerous':
                if i < c.SSM_D_THRESHOLD:
                    prob_val = 1
                else:
                    prob_val = np.exp(-np.square((i - c.SSM_D_THRESHOLD) / c.SSM_UNCERTAINTY) / 2)
            elif prob_type == 'attentive':
                if i < c.SSM_A_THRESHOLD:
                    prob_val = np.exp(-np.square((i - c.SSM_D_THRESHOLD) / c.SSM_UNCERTAINTY) / 2)
                elif i > c.SSM_A_THRESHOLD:
                    prob_val = np.exp(-np.square((i - c.SSM_A_THRESHOLD) / c.SSM_UNCERTAINTY) / 2)
                else:
                    prob_val = 1
            elif prob_type == 'safety':
                if i < c.SSM_A_THRESHOLD:
                    prob_val = np.exp(-np.square((i - c.SSM_A_THRESHOLD) / c.SSM_UNCERTAINTY) / 2)
                else:
                    prob_val = 1
            prob.append(prob_val)
        return np.array(prob)

    def calc_safety_surrogate_metric(self, ego_speed, leader_speed,dis_to_leader):
            

           
            if ego_speed <= leader_speed or dis_to_leader <= 0:
                return float('inf')  

            ttc = dis_to_leader / (ego_speed - leader_speed)
            #print(ttc)
            return ttc


    def _applyAction(self,veh_id,action_index):
        if not self.isVehRunninng(veh_id):
            return
        current_lane_index = traci.vehicle.getLaneIndex(veh_id)

        if action_index == len(self.action_list):
            target_lane_index = max(current_lane_index - 1, 0)
            traci.vehicle.changeLane(veh_id, target_lane_index, 1)

        elif action_index == len(self.action_list)+1:
            target_lane_index = min(current_lane_index + 1, self.num_of_lanes - 1)
            traci.vehicle.changeLane(veh_id, target_lane_index, 1)
        elif action_index == len(self.action_list)+2:
            pass
        else:
            #traci.vehicle.setAcceleration(veh_id, self.action_list[action_index],0.1)
            self.change_vehicle_speed(veh_id, float(self.action_list[action_index]), 1)

    def change_vehicle_speed(self, vehID, acceleration, duration=1.0):
        """Fix the acceleration of a vehicle to be a specified value in the specified duration.

        Args:
            vehID (str): Vehicle ID
            acceleration (float): Specified acceleration of vehicle.
            duration (float, optional): Specified time interval to fix the acceleration in s. Defaults to 1.0.
        """
        # traci.vehicle.slowDown take duration + deltaT to reach the desired speed
        init_speed = traci.vehicle.getSpeed(vehID)
        final_speed = init_speed + acceleration * (duration)
        if final_speed < 0:
            final_speed = 0
        traci.vehicle.slowDown(vehID, final_speed, duration)



    def getRunningTime(self):
        #start = traci.vehicle.getDeparture(self.ego)
        # end =
        return self.running_time

    def getDistance(self):
        return self.running_distance


    def record(self):
       

        if self._isEgoRunning() == False:
            return



        info = {'ego': {'position': traci.vehicle.getLaneID(self.ego), 'lane_position': traci.vehicle.getLanePosition(self.ego),
                        "x_pos" :traci.vehicle.getPosition(self.ego)[0], "y_pos": traci.vehicle.getPosition(self.ego)[1],
                            'speed': traci.vehicle.getSpeed(self.ego), 'acceleration': traci.vehicle.getAccel(self.ego)}}
        for i in range(6):
            if self.surrounding_vehicles[i] != "":
                adversarial = 0
                if i == self.ABV:
                    adversarial = 1
                info['background_vehicle_'+str(i)] = {'position': traci.vehicle.getLaneID(self.surrounding_vehicles[i]),
                                                      'lane_position': traci.vehicle.getLanePosition(self.surrounding_vehicles[i]),
                                                      "x_pos": traci.vehicle.getPosition(self.surrounding_vehicles[i])[0],
                                                      "y_pos": traci.vehicle.getPosition(self.surrounding_vehicles[i])[1],
                                                      'speed': traci.vehicle.getSpeed(self.surrounding_vehicles[i]),
                                                      'acceleration': traci.vehicle.getAccel(self.surrounding_vehicles[i]),
                                                      'adversarial': adversarial}
        return info

        
    def snapshot(self,k,i):
        traci.gui.screenshot("View #0", "snap_shot_images/" + str(k)+"-"+str(i) + ".png")
















