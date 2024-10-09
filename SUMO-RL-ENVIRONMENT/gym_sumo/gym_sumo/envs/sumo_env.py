import os
import gym
from gym import spaces
import pygame
import numpy as np 
import sys
import math
from gym_sumo.envs import env_config as c

#要使用该库，<SUMO_HOME>/tools 目录必须位于python加载路径上。 通常如下：
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

## add each lane density and mean_speed ##
state_space_high_for_norm = np.array([c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.HEADING_ANGLE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE
		,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY])
def creat_observation():
	state_space_list = ['ego_speed','ego_acc','ego_dis_to_leader','leader_speed','leader_acc','dis_to_left_leader','left_leader_speed','left_leader_acc',
	'dis_to_right_leader','right_leader_speed','right_leader_acc']
	for i in range(c.NUM_OF_LANES):
		state_space_list.append("lane_"+str(i)+"_mean_speed")
		state_space_list.append("lane_"+str(i)+"_density")
	#print(state_space_list)

	state_space_low = np.array([c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.HEADING_ANGLE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE
		,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY])
	state_space_high = np.array([c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.HEADING_ANGLE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE
		,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY])
    #与state_space_high_for_norm完全一致

	obs = spaces.Box(low=state_space_low,high=state_space_high,dtype=np.float64)
	#
	# A (possibly unbounded) box in R^n. Specifically, a Box represents the
    # Cartesian product of n closed intervals. Each interval has the form of one
    # of [a, b], (-oo, b], [a, oo), or (-oo, oo).
	#
    # There are two common use cases:
	#
    # * Identical bound for each dimension::
    #     >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
    #     Box(3, 4)
	#
    # * Independent bound for each dimension::
    #     >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
    #     Box(2,)
	#

	return obs

class SumoEnv(gym.Env):
	"""docstring for SumoEnv"""
	metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
	def __init__(self,seed="123456",render_mode=None,p1=0.5,p2=0.5,p3=0.5,p4=0.5,q1=0.5):
		"""
		初始化环境对象。

		参数：
			render_mode (str): 可视化模式，可以是 'human' 或 'rgb_array'。默认为 'human'。

		返回：
			None
		"""
		super(SumoEnv, self).__init__()
		# 调用父类的初始化方法
		self.action_space = spaces.Discrete(5)  #一共有五个 action
		r"""Constructor of :class:`Discrete` space.

		This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

		Args:
			n (int): The number of elements of this space.
			seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
			start (int): The smallest element of this space.
		Example::
			>>> Discrete(2)            # {0, 1}
			>>> Discrete(3, start=-1)  # {-1, 0, 1}
		"""
		self.observation_space = creat_observation()
		self.seed = seed

		## class variable
		self.ego = c.EGO_ID
		self.num_of_lanes = c.NUM_OF_LANES
		self.max_speed_limit = c.RL_MAX_SPEED_LIMIT
		self.is_collided = False # indicator of collision
		assert render_mode is None or render_mode in self.metadata['render_modes']
		# raise an error if render_mode is not None and not in metadate
		self.render_mode = render_mode
		self.min_speed_limit = c.RL_MIN_SPEED_LIMIT
		self.w1 = c.W1 # efficiency coefficient
		self.w2 = c.W2 # collision coefficient
		self.w3 = c.W3 # lane change coefficient

		self.p1 = p1
		self.p2 = p2
		self.p3 = p3
		self.p4 = p4
		self.q1 = q1
		## 背景车辆相关参数
		self.ego_x = 0
		self.ego_y = 0
		self.ego_speed = 0
		self.ego_acc = 0

	def _getInfo(self):
		return {"current_episode":0}

	def _startSumo(self, isGui=False):
		r"""
		启动SUMO仿真器。

		参数：
			isGui (bool): 是否使用GUI界面。默认为 False。

		返回：
			None
		"""

		#通常，从Python接入SUMO非常容易（以下示例是 tutorial / traci_tls 的修改）
		# from sumolib import checkBinary  # noqa
		# this script has been called from the command line. It will start sumo as a
		# server, then connect and run
		# if options.nogui:
		# 	sumoBinary = checkBinary('sumo')
		# else:
		# 	print("checkBinary('sumo-gui')")
		# 	sumoBinary = checkBinary('sumo-gui')
		#首先，编写命令行以启动 SUMO 或 SUMO-GUI：
		#省略在0.28.0版本之前需要的远程端口选项 --remote-port）

		sumoBinary = "sumo"
		if self.render_mode == "human":
			sumoBinary = "sumo-gui" if isGui else "sumo"

		sumoCmd = [sumoBinary, "-c", "SUMO-RL-ENVIRONMENT/gym_sumo/gym_sumo/envs/xml_files/test.sumocfg","--lateral-resolution","3.2",
		 "--start", "true", "--quit-on-end", "true","--no-warnings","True", "--no-step-log", "True", "--step-length",str(c.STEP_LENGTH),
		 "--random","false"]
		# -c < FILE >
		# --configuration - file < FILE >
		# Loads the named config on startup

		# --lateral-resolution <FLOAT>
		# Defines the resolution in m when handling lateral positioning within a lane
		# (with -1 all vehicles drive at the center of their lane; default: -1

		#-S <BOOL>
		#--start <BOOL>	Start the simulation after loading; default: false

		#-Q <BOOL>
		#--quit-on-end <BOOL> Quits the GUI when the simulation stops; default: false

		#-W <BOOL>
		#--no-warnings <BOOL> 	Disables output of warnings; default: false

  		#--no-step-log <BOOL>	Disable console output of current simulation step; default: false

		#--step-length <TIME>	Defines the step duration in seconds; default: 1

		#-random <BOOL>	Initialises the random number generator with the current system time; default: false

		#https://sumo.dlr.de/docs/sumo.html

		traci.start(sumoCmd)

	def mean_normalization(self, obs):
		#print(obs)
		return obs
		#print(f'After Normalization: {obs/state_space_high_for_norm}')
		# mu=np.mean(obs)
		# std = np.std(obs)
		# X = (obs-mu)/(max(obs)-min(obs))
		# return X

	def reset(self, isGui=False,seed=None, options=None):
		"""
		重置环境的状态并返回初始观测值。

		参数：
			isGui (bool): 是否使用GUI界面。默认为 False。
			seed (int): 随机种子值。默认为 None。
			options (dict): 其他选项参数。默认为 None。

		返回：
			np.array: 初始观测值数组。
			dict: 辅助信息字典。
		"""

		# reset函数用于得到done信号后启用新的脚本（episode），
		# 并需用在step函数前。可以传递seed关键字重置环境的任何随机数生成器（self.np_random），
		# 以保证初始化为同一确定性状态。如果在同一范围内使用，不必每次调用同一随机数生成器，
		# 但是需要调用super().reset(seed=seed)
		# 保证gym.Env类中的seed的范围正确。reset方法要么返回初始状态的观察值，要么返回初始观察值的元组和一些辅助信息，这取决于return_info是否为True，可以使用方法_get_obs和_get_info实现。
		# 原文链接：https: // blog.csdn.net / Rita_Aloha / article / details / 124696221
		super().reset(seed=seed)
		self.is_collided = False
		self._startSumo(isGui=isGui)
		self._warmup()
		obs = np.array(self.observation_space.sample()/state_space_high_for_norm)
		info = self._getInfo()
		return self.mean_normalization(obs), info

	def _getCloseLeader(self, leaders):
		"""
		获取距离最近的领导车辆信息。

		参数：
			leaders (list): 领导车辆列表，包含车辆ID和距离。

		返回：
			tuple: 最近的领导车辆ID和距离。
		"""
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
		"""
	    获取每个车道的车辆密度和平均速度。

	    参数：
		   无

	    返回：
		   tuple: 包含车辆密度列表和平均速度列表。
	    """

		road_id = traci.vehicle.getRoadID(self.ego)
		density = []
		mean_speed = []
		for i in range(self.num_of_lanes):
			density.append(len(traci.lane.getLastStepVehicleIDs(road_id+"_"+str(i))))
			mean_speed.append(traci.lane.getLastStepMeanSpeed(road_id+"_"+str(i)))
		return density, mean_speed
		# 返回车辆密度列表和平均速度列表

	def _get_rand_obs(self):
		"""
		获取随机观测值。

		参数：
			无

		返回：
			np.array: 随机观测值数组。
		"""
		return np.array(self.observation_space.sample())
	def _get_observation(self):
		"""
		获取当前观测值。

		参数：
			无

		返回：
			np.array: 当前观测值数组。
		"""
		if self._isEgoRunning()==False:
			return self._get_rand_obs()
		self.x = traci.vehicle.getLaneIndex(self.ego)
		self.y = traci.vehicle.getLanePosition(self.ego)/ c.RL_SENSING_RADIUS
		# get the speed of the ego vehicle
		ego_speed = traci.vehicle.getSpeed(self.ego)/c.RL_MAX_SPEED_LIMIT
		self.ego_speed = ego_speed
		# get the acceleration of the ego vehicle
		ego_accleration = traci.vehicle.getAccel(self.ego)/c.RL_ACC_RANGE
		self.ego_acc = ego_accleration
		ego_leader = traci.vehicle.getLeader(self.ego)
		#Return the leading vehicle id together with the distance.
		ego_heading_angle = (traci.vehicle.getAngle(self.ego)-90.0)/c.HEADING_ANGLE
		if ego_leader is not None:
			leader_id, distance = ego_leader
		else:
			leader_id, distance = "", -1
		l_speed = traci.vehicle.getSpeed(leader_id)/c.NON_RL_VEH_MAX_SPEED if leader_id != "" else 0.01/c.NON_RL_VEH_MAX_SPEED
		l_acc = traci.vehicle.getAccel(leader_id)/c.RL_ACC_RANGE if leader_id != "" else -2.6/c.RL_ACC_RANGE
		left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=False))
		#   """ getLeftLeaders(string, bool) -> list(pair(string, double))
        # 		Convenience method, see getNeighbors()
				#blockingOnly = False --> mode = 2
        # """
		#The parameter mode is a bitset (UBYTE), specifying the following:
        # bit 1: query lateral direction (left:0, right:1)
        # bit 2: query longitudinal direction (followers:0, leaders:1)
        # bit 3: blocking (return all:0, return only blockers:1)
		left_l_speed = traci.vehicle.getSpeed(left_leader)/c.NON_RL_VEH_MAX_SPEED if left_leader != "" else 0.01/c.NON_RL_VEH_MAX_SPEED
		left_l_acc = traci.vehicle.getAccel(left_leader)/c.RL_ACC_RANGE if left_leader != "" else -2.6/c.RL_ACC_RANGE

		right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=False))
		right_l_speed = traci.vehicle.getSpeed(right_leader)/c.NON_RL_VEH_MAX_SPEED if right_leader != "" else 0.01/c.NON_RL_VEH_MAX_SPEED
		right_l_acc = traci.vehicle.getAccel(right_leader)/c.RL_ACC_RANGE if right_leader != "" else -2.6/c.RL_ACC_RANGE

		states = [ego_speed, ego_accleration, ego_heading_angle, distance/c.RL_SENSING_RADIUS, l_speed, l_acc, left_l_dis/c.RL_SENSING_RADIUS, left_l_speed, left_l_acc,
			right_l_dis/c.RL_SENSING_RADIUS, right_l_speed, right_l_acc]
		density, mean_speed = self._getLaneDensity()
		for i in range(self.num_of_lanes):
			states.append(density[i]/c.MAX_LANE_DENSITY)
			states.append(mean_speed[i]/c.LANE_MEAN_SPEED)

		observations = np.array(states)
		return observations

	def _applyAction(self, action):
		"""
		根据动作执行相应的操作。

		参数：
			action (int): 动作编号。

		返回：
			None
		"""
		if self._isEgoRunning() == False:
			return
		current_lane_index = traci.vehicle.getLaneIndex(self.ego)
		accel = traci.vehicle.getAcceleration(self.ego)
		#print(f'Acceleration: {accel}')
		if action == 0:
			# do nothing: stay in the current lane
			pass
		elif action == 1:
			#right lane change
			target_lane_index = min(current_lane_index+1, self.num_of_lanes-1)
			traci.vehicle.changeLane(self.ego, target_lane_index, c.STEP_LENGTH)
		elif action == 2:
			#left lane change
			target_lane_index = max(current_lane_index-1, 0)
			traci.vehicle.changeLane(self.ego, target_lane_index, c.STEP_LENGTH)
		elif action == 3:
			#accelerate
			traci.vehicle.setAcceleration(self.ego,0.1, c.STEP_LENGTH)
		elif action == 4:
			#decelerate
			traci.vehicle.setAcceleration(self.ego, -4.5, c.STEP_LENGTH)

	def _applyBVAction(self, BV_id,BV_action):
		"""
		根据动作执行相应的操作。

		参数：
			BV_id(string):背景车辆编号。
			action (int): 动作编号。

		返回：
			None
		"""
		current_lane_index = traci.vehicle.getLaneIndex(BV_id)
		accel = traci.vehicle.getAcceleration(BV_id)
		# print(f'Acceleration: {accel}')
		if BV_action == 0:
			# do nothing: stay in the current lane
			pass
		elif BV_action == 1:
			# right lane change
			target_lane_index = min(current_lane_index + 1, self.num_of_lanes - 1)
			traci.vehicle.changeLane(BV_id, target_lane_index, c.STEP_LENGTH)
		elif BV_action == 2:
			# left lane change
			target_lane_index = max(current_lane_index - 1, 0)
			traci.vehicle.changeLane(BV_id, target_lane_index, c.STEP_LENGTH)
		elif BV_action == 3:
			# accelerate
			traci.vehicle.setAcceleration(BV_id, 0.1, c.STEP_LENGTH)
		elif BV_action == 4:
			# decelerate
			traci.vehicle.setAcceleration(BV_id, -4.5, c.STEP_LENGTH)

	def _collision_reward(self):
		"""
		计算碰撞奖励。

		参数：
			无

		返回：
			float: 碰撞奖励值。
		"""
		# if the ego vehicle gets collided, then the reward is -10, if not, 0.
		collide_vehicles = traci.simulation.getCollidingVehiclesIDList()
		if self.ego in collide_vehicles:
			self.is_collided = True
			return -10
		return 0.0
	def _efficiency(self):
		"""
	    计算效率奖励。

	    参数：
		    无

	    返回：
		    float: 效率奖励值。
	    """
		speed = traci.vehicle.getSpeed(self.ego)
		if speed <= self.min_speed_limit:
			return (speed-self.min_speed_limit)/(self.max_speed_limit-self.min_speed_limit)
		if speed > self.max_speed_limit:
			return (self.max_speed_limit-speed)/(self.max_speed_limit-self.min_speed_limit)
		return speed/self.max_speed_limit
	def _lane_change_reward(self,action):
		"""
		计算车道变更奖励。

		参数：
			action (int): 动作编号。

		返回：
			float: 车道变更奖励值。
		"""
		if action == 1 or action == 2:
			return -1.0
		return 0

	def time_loss_reward(self):
		"""
	    计算时间损失奖励。

	    参数：
		    无

	    返回：
		    float: 时间损失奖励值。
	    """
		if self._isEgoRunning():
			return traci.vehicle.getTimeLoss(self.ego)
		return 0.0

	def _reward(self, action):
		"""
		计算总奖励。

		参数：
			action (int): 动作编号。

		返回：
			float: 总奖励值。
		"""
		c_reward = self._collision_reward()
		if self.is_collided or self._isEgoRunning()==False:
			return c_reward*self.w2
		return c_reward*self.w2 + self._efficiency()*self.w1 + self._lane_change_reward(action)*self.w3



	def step(self, action):

		#getneighbors
		#nei
		#ann(nei)
		#(nei_actions)
		"""
		执行一步动作。

		参数：
			action (int): 动作编号。

		返回：
			tuple: 包含下一个观测值、奖励值、是否结束、额外信息的元组。
		"""
		self._applyAction(action)
		ego_vehicle_id = self.ego
		surrounding_vehicles_info = self.get_surrounding_vehicles_info(ego_vehicle_id)

		# 判断对自主车辆最危险的车辆
		most_dangerous_vehicle_id,most_dangerous_action = self.get_most_dangerous_vehicle(surrounding_vehicles_info,action)

		# 让最危险的车辆执行行动，以使其与自主车辆产生碰撞
		if most_dangerous_vehicle_id:
			# 在这里执行最危险车辆的行动
			self._applyBVAction(most_dangerous_vehicle_id,most_dangerous_action)




		traci.simulationStep()
		reward = self._reward(action)
		observation = self._get_observation()
		time_loss = self.time_loss_reward()
		done = self.is_collided or (self._isEgoRunning()==False)
		if traci.simulation.getTime() > 720:
			done = True
		return (self.mean_normalization(observation), reward, done, {})


	def _isEgoRunning(self):
		"""
		判断自主车辆是否在道路上运行。

		参数：
			无

		返回：
			bool: 自主车辆是否在道路上运行的布尔值。
		"""
		# indicator of if EgoVehicle os on the three edges.
		v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
		v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
		v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
		if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
			return True
		return False

	def _warmup(self):
		"""
	    执行仿真预热操作。

	    参数：
		    无

	    返回：
		    None
	    """
		#set the laneChangeMode of ego_vehicle as none.
		while True:
			v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
			#getLastStepVehicleIDs(self, edgeID)
			#	getLastStepVehicleIDs(string) -> list(string)
			#Returns the ids of	the vehicles for the last time step on the given edge.
			v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
			v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
			if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
				traci.vehicle.setLaneChangeMode(self.ego,0)
				#setLaneChangeMode(string, integer) -> None
				# #Sets the vehicle's lane change mode as a bitset.
				# laneChangeMode equals zero means that the corresponding bitset is equal to 0b000000000000, which means "do no any change"
				# I guess this code is written for avoiding conflict between sumo lane change mode and traci command.
				#traci.vehicle.setSpeedMode(self.ego,0)
				return True
			traci.simulationStep()
			# simulationStep(float) -> list
			# Make a simulation step and simulate up to the given second in sim time.
			# If the given value is 0 or absent, exactly one step is performed.
			# Values smaller than or equal to the current sim time result in no action.
			# It returns the subscription results for the current step in a list.
			#

	def closeEnvConnection(self):
		"""
		关闭环境连接。

		参数：
			无

		返回：
			None
		"""
		traci.close()

	def move_gui(self):
		"""
	    更新GUI视图的位置。

	    参数：
		    无

	    返回：
		    None
	    """
		if self.render_mode == "human":
			x, y = traci.vehicle.getPosition('av_0')
			traci.gui.setOffset("View #0",x-50.0,108.49)
			# Set the position of the view named viewId to x and y.
			# setOffset(string, float, float) -> None
			# This value is the offset from the middle of the view.
			# setOffset only works with named views, which have to be defined in the gui.add command.
			# The naming of a view is done by the id attribute.
			# All views in the same level are centered in the same point,
			# no matter how many views there are, they are displayed evenly around that point.
			# But if there is another level of views, the views are organized in a kind of table,
			# with all views in a row being centered in the point specified by the previous level.
			# One can use a background image for each view, but that image must be square.
			#

	def get_vehicles_info(self,ego_vehicle_id,direction):
		"""
		获取车辆信息。

		参数：
			ego_vehicle_id (str): 自主车辆的ID。
			direction(str):周围车辆的方向,共有front，left_front, right_front,rear,left_rear,right_rear六种

		返回：
			surrounding_vehicles_info(dict): 周围车辆信息。
		"""
		# ["front_vehicle", "left_front_vehicle", "left_rear_vehicle", "rear_vehicle", "right_rear_vehicle",
		#  "right_front_vehicle"]
		surrounding_vehicles_info  = None
		_vehicle_info = [None,None]
		ID = None
		dis = np.inf
		if direction == "front_vehicle":
			if not traci.vehicle.getLeader(ego_vehicle_id) is None:
				ID, dis = traci.vehicle.getLeader(ego_vehicle_id)
		if direction == "left_front_vehicle":
			if not traci.vehicle.getLeftLeaders(ego_vehicle_id) is None:
				ID, dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=False))
		if direction == "left_rear_vehicle":
			if not traci.vehicle.getLeftFollowers(ego_vehicle_id) is None:
				ID, dis = self._getCloseLeader(traci.vehicle.getLeftFollowers(self.ego, blockingOnly=False))
		if direction == "rear_vehicle":
			if not traci.vehicle.getFollower(ego_vehicle_id) is None:
				ID, dis = traci.vehicle.getFollower(ego_vehicle_id)
		if direction == "right_rear_vehicle":
			if not traci.vehicle.getRightFollowers(ego_vehicle_id) is None:
				ID, dis =self._getCloseLeader(traci.vehicle.getRightFollowers(self.ego, blockingOnly=False))
		if direction == "right_front_vehicle":
			if not traci.vehicle.getRightLeaders(ego_vehicle_id) is None:
				ID, dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=False))
		surrounding_vehicles_info = self._get_vehicle_info(ID, dis)
		return surrounding_vehicles_info

	def _get_vehicle_info(self, vehicle_id, distance):
		"""
        获取单个车辆的详细信息。

        参数：
            vehicle_id (str): 车辆的ID。

        返回：
            dict: 包含车辆信息的字典。
        """
		# print("vehicle_id = ",vehicle_id)
		vehicle_info = None
		if vehicle_id is not None and vehicle_id != "":
			vehicle_info = {
				"id": vehicle_id,
				"distance": distance / c.RL_SENSING_RADIUS,
				"speed": traci.vehicle.getSpeed(vehicle_id) / c.NON_RL_VEH_MAX_SPEED,
				"acceleration": traci.vehicle.getAccel(vehicle_id) / c.RL_ACC_RANGE,
				"lane_index": traci.vehicle.getLaneIndex(vehicle_id),  # 获取车辆所在车道Index
				"lane_position": traci.vehicle.getLanePosition(vehicle_id) / c.RL_SENSING_RADIUS # 获取车辆在车道上的位置
			}
		return vehicle_info



	def get_surrounding_vehicles_info(self, ego_vehicle_id):
		"""
        获取自主车辆周围的全部车辆信息。

        参数：
            ego_vehicle_id (str): 自主车辆的ID。

        返回：
            dict: 包含周围车辆信息的字典。
        """
		surrounding_vehicles_info = {
			"front_vehicle": None,
			"left_front_vehicle": None,
			"left_rear_vehicle": None,
			"rear_vehicle": None,
			"right_rear_vehicle": None,
			"right_front_vehicle": None
		}
		if self._isEgoRunning() == False:
			return surrounding_vehicles_info


		# ego_speed = traci.vehicle.getSpeed(self.ego) / c.RL_MAX_SPEED_LIMIT
		# ego_accleration = traci.vehicle.getAccel(self.ego) / c.RL_ACC_RANGE
		# ego_leader = traci.vehicle.getLeader(self.ego)
		# # Return the leading vehicle id together with the distance.
		# ego_heading_angle = (traci.vehicle.getAngle(self.ego) - 90.0) / c.HEADING_ANGLE
		# if ego_leader is not None:
		# 	leader_id, distance = ego_leader
		# else:
		# 	leader_id, distance = "", -1
		# l_speed = traci.vehicle.getSpeed(
		# 	leader_id) / c.NON_RL_VEH_MAX_SPEED if leader_id != "" else 0.01 / c.NON_RL_VEH_MAX_SPEED
		# l_acc = traci.vehicle.getAccel(leader_id) / c.RL_ACC_RANGE if leader_id != "" else -2.6 / c.RL_ACC_RANGE
		# left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=False))
		# left_l_speed = traci.vehicle.getSpeed(
		# 	left_leader) / c.NON_RL_VEH_MAX_SPEED if left_leader != "" else 0.01 / c.NON_RL_VEH_MAX_SPEED
		# left_l_acc = traci.vehicle.getAccel(
		# 	left_leader) / c.RL_ACC_RANGE if left_leader != "" else -2.6 / c.RL_ACC_RANGE
		#
		# right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=False))
		# right_l_speed = traci.vehicle.getSpeed(
		# 	right_leader) / c.NON_RL_VEH_MAX_SPEED if right_leader != "" else 0.01 / c.NON_RL_VEH_MAX_SPEED
		# right_l_acc = traci.vehicle.getAccel(
		# 	right_leader) / c.RL_ACC_RANGE if right_leader != "" else -2.6 / c.RL_ACC_RANGE
		#
		# states = [ego_speed, ego_accleration, ego_heading_angle, distance / c.RL_SENSING_RADIUS, l_speed, l_acc,
		# 		  left_l_dis / c.RL_SENSING_RADIUS, left_l_speed, left_l_acc,
		# 		  right_l_dis / c.RL_SENSING_RADIUS, right_l_speed, right_l_acc]



		surrounding_vehicles_list = ["front_vehicle","left_front_vehicle","left_rear_vehicle","rear_vehicle","right_rear_vehicle","right_front_vehicle"]
		for i in surrounding_vehicles_list:
			surrounding_vehicles_info[i] = self.get_vehicles_info(ego_vehicle_id,i)

		# # 获取前方车辆信息
		# if not traci.vehicle.getLeader(ego_vehicle_id) is None:
		# 	front_vehicle_info = traci.vehicle.getLeader(ego_vehicle_id)
		# 	front_vehicle_id = front_vehicle_info[0]
		# 	front_distance = front_vehicle_info[1]
		# 	if front_vehicle_id:
		# 		front_vehicle_speed = traci.vehicle.getSpeed(front_vehicle_id)
		# 		front_vehicle_acceleration = traci.vehicle.getAccel(front_vehicle_id)
		# 		surrounding_vehicles_info["front_vehicle"] = {
		# 			"id": front_vehicle_id,
		# 			"distance": front_distance,
		# 			"speed": front_vehicle_speed,
		# 			"acceleration": front_vehicle_acceleration
		# 		}
		#
		# # 获取左前方车辆信息
		# left_front_vehicles = traci.vehicle.getLeftLeaders(ego_vehicle_id)
		# if left_front_vehicles:
		# 	left_front_vehicle_id, left_front_distance = left_front_vehicles[0]
		# 	left_front_vehicle_speed = traci.vehicle.getSpeed(left_front_vehicle_id)
		# 	left_front_vehicle_acceleration = traci.vehicle.getAccel(left_front_vehicle_id)
		# 	surrounding_vehicles_info["left_front_vehicle"] = {
		# 		"id": left_front_vehicle_id,
		# 		"distance": left_front_distance,
		# 		"speed": left_front_vehicle_speed,
		# 		"acceleration": left_front_vehicle_acceleration
		# 	}
		#
		# # 获取左后方车辆信息
		# left_rear_vehicles = traci.vehicle.getLeftFollowers(ego_vehicle_id)
		# if left_rear_vehicles:
		# 	left_rear_vehicle_id, left_rear_distance = left_rear_vehicles[0]
		# 	left_rear_vehicle_speed = traci.vehicle.getSpeed(left_rear_vehicle_id)
		# 	left_rear_vehicle_acceleration = traci.vehicle.getAccel(left_rear_vehicle_id)
		# 	surrounding_vehicles_info["left_rear_vehicle"] = {
		# 		"id": left_rear_vehicle_id,
		# 		"distance": left_rear_distance,
		# 		"speed": left_rear_vehicle_speed,
		# 		"acceleration": left_rear_vehicle_acceleration
		# 	}
		#
		# # 获取后方车辆信息
		# rear_vehicle_id, rear_distance = traci.vehicle.getFollower(ego_vehicle_id)
		# if rear_vehicle_id:
		# 	rear_vehicle_speed = traci.vehicle.getSpeed(rear_vehicle_id)
		# 	rear_vehicle_acceleration = traci.vehicle.getAccel(rear_vehicle_id)
		# 	surrounding_vehicles_info["rear_vehicle"] = {
		# 		"id": rear_vehicle_id,
		# 		"distance": rear_distance,
		# 		"speed": rear_vehicle_speed,
		# 		"acceleration": rear_vehicle_acceleration
		# 	}
		#
		# # 获取右后方车辆信息
		# right_rear_vehicles = traci.vehicle.getRightFollowers(ego_vehicle_id)
		# if right_rear_vehicles:
		# 	right_rear_vehicle_id, right_rear_distance = right_rear_vehicles[0]
		# 	right_rear_vehicle_speed = traci.vehicle.getSpeed(right_rear_vehicle_id)
		# 	right_rear_vehicle_acceleration = traci.vehicle.getAccel(right_rear_vehicle_id)
		# 	surrounding_vehicles_info["right_rear_vehicle"] = {
		# 		"id": right_rear_vehicle_id,
		# 		"distance": right_rear_distance,
		# 		"speed": right_rear_vehicle_speed,
		# 		"acceleration": right_rear_vehicle_acceleration
		# 	}
		#
		# # 获取右前方车辆信息
		# right_front_vehicles = traci.vehicle.getRightLeaders(ego_vehicle_id)
		# if right_front_vehicles:
		# 	right_front_vehicle_id, right_front_distance = right_front_vehicles[0]
		# 	right_front_vehicle_speed = traci.vehicle.getSpeed(right_front_vehicle_id)
		# 	right_front_vehicle_acceleration = traci.vehicle.getAccel(right_front_vehicle_id)
		# 	surrounding_vehicles_info["right_front_vehicle"] = {
		# 		"id": right_front_vehicle_id,
		# 		"distance": right_front_distance,
		# 		"speed": right_front_vehicle_speed,
		# 		"acceleration": right_front_vehicle_acceleration
		# 	}

		return surrounding_vehicles_info

	def get_most_dangerous_vehicle(self, surrounding_vehicles_info,Av_action):
		"""
        判断对自主车辆最危险的车辆。

        参数：
            surrounding_vehicles_info (dict): 周围车辆信息字典。

        返回：
            str: 最危险的车辆ID。
        """
		most_dangerous_vehicle_id = None
		most_dangerous_action = 0
		max_critical = -np.inf
		for direction, vehicle_info in surrounding_vehicles_info.items():
			if vehicle_info is not None:
				action, critical = self.make_most_dangerous_action(vehicle_info,Av_action)
				if critical > max_critical:
					max_critical = critical
					most_dangerous_vehicle_id = vehicle_info["id"]
					most_dangerous_action = action
			#
			# if vehicle_info and vehicle_info["distance"] < min_distance:
			# 	min_distance = vehicle_info["distance"]
			# 	most_dangerous_vehicle_id = vehicle_info["id"]

		return most_dangerous_vehicle_id, most_dangerous_action

	def make_most_dangerous_action(self, vehicle_info,AV_action):
		"""
        使最危险的车辆执行行动，以使其与自主车辆产生碰撞。

        参数：
            most_dangerous_vehicle_id (str): 最危险的车辆ID。
            ego_vehicle_id (str): 自主车辆ID。
        """
		# 默认行动为减速
		max_critical = -np.inf
		action = 4  # 减速
		for BV_action in range(self.action_space.n):
			critical = self.dangerous_critical_measurement(vehicle_info, AV_action, BV_action)
			if critical > max_critical:
				max_critical = critical
				action = BV_action
		return action, max_critical
		# 可以根据具体情况调整行动
		# 比如，如果最危险车辆在前方且距离较近，则加速或变换车道
		# 如果最危险车辆在后方，则保持减速状态

		# # 获取最危险车辆的位置和速度信息
		# most_dangerous_vehicle_pos = traci.vehicle.getPosition(most_dangerous_vehicle_id)
		# most_dangerous_vehicle_speed = traci.vehicle.getSpeed(most_dangerous_vehicle_id)
		#
		# # 获取自主车辆的位置信息
		# ego_vehicle_pos = traci.vehicle.getPosition(self.ego)
		#
		# # 如果最危险车辆在前方且距离较近，则加速或变换车道
		# if most_dangerous_vehicle_pos[0] > ego_vehicle_pos[0] and most_dangerous_vehicle_speed > 0:
		# 	action = 3  # 加速
		#
		# # 返回最危险的行动
		# return action

	def dangerous_critical_measurement(self,vehicle_info,AV_action,BV_action):
		simulation_step_length = traci.simulation.getDeltaT() / 1000

		current_lane_index = self.ego_x
		bv_current_lane_index = vehicle_info["lane_index"]
		target_lane_index = current_lane_index
		bv_target_lane_index = bv_current_lane_index

		lane_position= self.ego_y
		bv_lane_position = vehicle_info["lane_position"]

		av_ACC = self.ego_acc
		av_speed = self.ego_speed
		bv_ACC = vehicle_info["acceleration"]
		bv_speed = vehicle_info["speed"]

		if AV_action == 0:
			# do nothing: stay in the current lane
			pass
		elif AV_action == 1:
			#right lane change
			target_lane_index = min(current_lane_index+1, self.num_of_lanes-1)
		elif AV_action == 2:
			#left lane change
			target_lane_index = max(current_lane_index-1, 0)
		elif AV_action == 3:
			#accelerate
			av_ACC= 0.1
		elif AV_action == 4:
			#decelerate
			av_ACC =-4.5

		if BV_action == 0:
			# do nothing: stay in the current lane
			pass
		elif BV_action == 1:
			# right lane change
			bv_target_lane_index = min(bv_current_lane_index + 1, self.num_of_lanes - 1)
		elif BV_action == 2:
			# left lane change
			bv_target_lane_index = max(bv_current_lane_index - 1, 0)
		elif BV_action == 3:
			# accelerate
			bv_ACC = 0.1
		elif BV_action == 4:
			# decelerate
			bv_ACC = -4.5
		critical = -np.abs(self.p1*(target_lane_index - bv_target_lane_index)) - np.abs(self.p2*(lane_position - bv_lane_position) + self.p3*(av_speed - bv_speed)*simulation_step_length+  0.5 *self.p4*(av_ACC - bv_ACC)*simulation_step_length**2) +self.q1
		return critical