import numpy as np
from gym_sumo.envs import env_config as c

import math
from typing import Dict, Tuple, Union, Any
#import traci

class DriverModel:

    """Creating the DriverModel specific to the created vehicle based on
    the Intelligent Driver Model and the MOBIL lane change model
    """

    def __init__(self) -> None:

        """Intializing IDM based on level of driving cautiousness
        """

        self.v_0 = c.IDM_v0 # Max desired speed of vehicle
        self.s_0 = c.IDM_s0 # Min desired distance between vehicles
        self.a = c.IDM_a # Max acceleration
        self.b = c.IDM_b # Comfortable deceleration
        self.delta = c.IDM_delta # Acceleration component
        self.T = c.IDM_delta #  Time safe headway
        #
        # self.tau = c.IDM_TAU  # Reaction time
        # self.action_step_length = c.IDM_ACTIONSTEPLENGTH  # Action step length
        # self.last_decision_time = 0


        self.left_bias = c.MOBIL_left_Bias # Keep left bias
        self.politeness = c.MOBIL_politeness # Change lane politeness
        self.change_threshold = c.MOBIL_change_threshold # Change lane threshold


    def calc_acceleration(self, v: float, surrounding_v: float, s: float) -> float:

        """Calculates the vehicle acceleration based on the IDM

        Args:
            v (float): current vehicle velocity
            surrounding_v (float): velocity of other vehicle
            s (float): current actual distance

        Returns:
            float: vehicle acceleration
        """
        # current_time = traci.simulation.getTime()
        # if current_time - self.last_decision_time < self.action_step_length:
        #     return 0  # 在非决策时间步中，保持不变的加速度
        #
        #     # 更新最后决策时间
        # self.last_decision_time = current_time
        #
        # # 计算反应时间后的速度和距离
        # effective_v = v  # 在此版本中，假设当前速度在反应时间内不变
        # effective_surrounding_v = surrounding_v  # 假设前车速度也保持不变
        # effective_s = s - effective_surrounding_v * self.tau  # 修正与前车的距离，考虑反应时间
        #
        # delta_v = effective_v - effective_surrounding_v
        # # 计算期望安全距离 s_star
        # s_star = self.s_0 + max(0, self.T * effective_v + (effective_v * delta_v) / (2 * math.sqrt(self.a * self.b)))
        #
        # # 计算并返回加速度
        # return self.a * (1 - math.pow(effective_v / self.v_0, self.delta) - math.pow(s_star / effective_s, 2))

        delta_v = v - surrounding_v
        s_star = self.s_0 + max(0, self.T * v + (v * delta_v) / (2 * math.sqrt(self.a * self.b)))

        return self.a * (1 - math.pow(v/self.v_0, self.delta) - math.pow(s_star/s, 2))


    def calc_disadvantage(self, v: float, new_surrounding_v: float, new_surrounding_dist: float, old_surrounding_v: float, old_surrounding_dist: float) -> Tuple[float, float]:

        """Calculates the disadvantage of changing lanes based on the MOBIL Model

        Args:
            v (float): current vehicle velocity
            new_surrounding_v (float): velocity of targeted front vehicle
            new_surrounding_dist (float): distance between targeted front vehicle and current vehicle if change lane
            old_surrounding_v (float): velocity of current front vehicle
            old_surrounding_dist (float): distance between current front vehicle and current vehicle

        Returns:
            Tuple[float, float]: disadvantage of changing lanes and new acceleration if lane is changed
        """

        # Acceleration of the trailing vehicle behind the current vehicle
        new_back_acceleration = self.calc_acceleration(v, new_surrounding_v, new_surrounding_dist)

        # Acceleration of the currently investigated vehicle
        current_acceleration = self.calc_acceleration(v, old_surrounding_v, old_surrounding_dist)

        # Disadvantage on the trailing vehicle if lane changed
        disadvantage = current_acceleration - new_back_acceleration

        return disadvantage, new_back_acceleration


    def calc_incentive(self, change_direction: str, v: float, new_front_v: float, new_front_dist: float,
                    old_front_v: float, old_front_dist: float, disadvantage: float, new_back_accel: float, onramp_flag: bool) -> bool:

        """Determine the incentive of changing lanes based on the IDM and MOBIL Model

        Args:
            change_direction (str): the direction of the lane change
            v (float): current vehicle velocity
            new_front_v (float): velocity of targeted front vehicle
            new_front_dist (float): distance between targeted front vehicle and current vehicle if change lane
            old_front_v (float): velocity of current front vehicle
            old_front_dist (float): distance between current front vehicle and current vehicle
            disadvantage (float): difference in acceleration if lane changed
            new_back_accel (float): new acceleration if lane is changed
            onramp_flag (bool): onramp vehicle check

        Returns:
            bool: if a lane change should happen
        """

        # IDM
        # Acceleration of the new front vehicle in the targeted lane
        new_front_acceleration = self.calc_acceleration(v=v, surrounding_v=new_front_v, s=new_front_dist)

        # Acceleration of the currently investigated vehicle
        current_acceleration = self.calc_acceleration(v=v, surrounding_v=old_front_v, s=old_front_dist)

        # MOBIL Model
        if change_direction == 'right':
            a_bias = self.left_bias
        elif change_direction == 'left': # Reduce the RHS of the MOBIL incentive equation
            a_bias = -self.left_bias
        elif onramp_flag: # If the vehicle is onramp, decrease threshold for right lane change
            a_bias = -self.left_bias
        else:
            a_bias = 0 # No lane change

        # MOBIL incentive equation
        change_incentive = new_front_acceleration - current_acceleration - (self.politeness * disadvantage) > self.change_threshold + a_bias

        # MOBIL kinematic-based safety criterion
        safety_criterion = new_back_accel >= -self.b

        return change_incentive and safety_criterion


    def get_action(self, obs):
        state_space_list = ['lane_index','ego_speed', 'ego_acc', 'ego_heading_angle',
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

        state = {key: value for key, value in zip(state_space_list, obs[0])}

        #state = dict(zip(state_space_list, obs))




        #calculate the acceleration of the ego vehicle
        action_space = np.arange(-c.RL_DCE_RANGE, c.RL_ACC_RANGE+c.ACC_INTERVAL, c.ACC_INTERVAL)
        ego_acc = self.calc_acceleration(state["ego_speed"], state["leader_speed"], state["ego_dis_to_leader"])
        #写一个映射，将ego_acc映射到动作空间,这里有actions include 31 discrete longitudinal accelerations ([−4, 2] with 0.2 m s−2 discrete resolution)
        if ego_acc < -c.RL_DCE_RANGE:
            index = 0
        elif ego_acc > c.RL_ACC_RANGE:
            index = len(action_space) - 1
        else:
            index = np.searchsorted(action_space, ego_acc, side='left')



        #left lane change
        left_disadvantage, new_back_accel_left = self.calc_disadvantage(state["ego_speed"], state["left_leader_speed"], state["dis_to_left_leader"], state["leader_speed"], state["ego_dis_to_leader"])
        left_incentive = self.calc_incentive('left', state["ego_speed"], state["left_leader_speed"], state["dis_to_left_leader"], state["leader_speed"], state["ego_dis_to_leader"], left_disadvantage, new_back_accel_left, False)
        if state["lane_index"] == 0:
            left_incentive = False
        #right lane change
        right_disadvantage, new_back_accel_right = self.calc_disadvantage(state["ego_speed"], state["right_leader_speed"], state["dis_to_right_leader"], state["leader_speed"], state["ego_dis_to_leader"])
        right_incentive = self.calc_incentive('right', state["ego_speed"], state["right_leader_speed"], state["dis_to_right_leader"], state["leader_speed"], state["ego_dis_to_leader"], right_disadvantage, new_back_accel_right, False)
        if state["lane_index"] == c.NUM_OF_LANES - 1:
            right_incentive = False

        if left_incentive:
            action_index = len(action_space)
        elif right_incentive:
            action_index = len(action_space) + 1
        else:
            action_index = index

        return action_index
















