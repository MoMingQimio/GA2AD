
import os
from datetime import datetime
from itertools import count
import torch
import numpy as np
import math
import gym
from multiprocessing import Process
import json
from PPO import PPO
from dqn_sumo_gym import Agent
import warnings
warnings.filterwarnings("ignore")

def train(seed = str(0), print_flag = False):

    # set device to cpu or cuda
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        if print_flag:
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        if print_flag:
            print("Device set to : cpu")

    ####### initialize environment hyperparameters ######
    env_name = "sumo-v0"
    render_mode = "human"
    # render_mode = ""
    has_continuous_action_space = False  # continuous action space; else discrete


    max_training_timesteps = int(3e4)  # break training loop if timeteps > max_training_timesteps
    save_model_freq = int(1e2)   # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)这个参数在离散动作空间中没用

    update_timestep = 1000 # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = int(seed)  # set random seed if required (0 = no random seed)

    env = gym.make(env_name, render_mode=render_mode)
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        
        action_dim = [env.action_space.nvec[0], env.action_space.nvec[1]]
        # action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + '/' + 'BV' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #### get number of log files in log directory

    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num)+"_"+seed + ".csv"

    #### create new log file for each run
    av_dir = "log"
    if not os.path.exists(av_dir):
        os.makedirs(av_dir)
    av_dir = av_dir + '/' + 'TV' + '/'
    if not os.path.exists(av_dir):
        os.makedirs(av_dir)

    current_num_files = next(os.walk(av_dir))[2]
    av_run_num = len(current_num_files)
    av_log_f_name = av_dir + "DQN_" + env_name + "_log_" + str(av_run_num) +"_"+seed+ ".csv"

    #### log pretrained model
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + 'BV' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    directory = "preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + 'TV' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    av_checkpoint_path = directory + "DQN_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    if print_flag:
        print("save checkpoint path : " + checkpoint_path)

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    BV_agent = PPO(state_dim + 1, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std)
    TV_agent = Agent("Agent", env.action_space.nvec[1], state_dim + 1)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    if print_flag:
        print("Started training at (GMT) : ", start_time)

    # print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,collision_counts,adversarial_counts,reward\n')
    av_log_f = open(av_log_f_name, "w+")
    av_log_f.write(
        "episode,timestep,reward,if_collision,collision_rate,adversarial_counts,epsilon,av_speed_avg,av_total_risk_avg,av_distance,av_running_time,av_acc_avg,av_dece_avg,av_left_avg,av_right_avg\n")
  

    time_step = 0
    i_episode = 0
    collision_counts = 0
    adversarial_counts = 0
    epsilon = 0
    collision_rate = 0.2
    collision_intervals = 0
    total_collision_counts = 0
    while time_step <= max_training_timesteps:

        # observation, surrounding_vehicles, excepted_risk = env.reset()
        observation = env.reset()
        # print(observation)
        TV_ep_reward = 0
        BV_ep_reward = 0
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = observation
        av_speed = []
        av_acceleration = []
        av_deceleration = []
        av_left_change = []
        av_right_change = []
        av_total_risk = []
        av_distance = 0
        av_running_time = 0
        collision_flag = False
        data_records = []

        for t in count():

            # select action with policy

            TV_action = TV_agent.select_action(observation)
            BV_action, BV_action_logprob, BV_state_val = BV_agent.select_action(observation)
            
            BV_candidate_index = BV_action[0]
            BV_maneuver_index = BV_action[1]
            # Veh_id = surrounding_vehicles[BV_candidate_index]

            # final_actions = (TV_action.item(), Veh_id, BV_maneuver_index, epsilon)
            # final_actions = (TV_action.item(), Veh_id,BV_candidate_index, BV_maneuver_index)
            final_actions = (TV_action.item(), BV_candidate_index, BV_maneuver_index)
            observation, TV_reward, done, other_information = env.step(final_actions)
            av_speed.append(observation[1])
            if observation[2] > 0:
                av_acceleration.append(observation[2])

            else:
                av_deceleration.append(observation[2])
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            BV_reward = other_information["BV_reward"]
            # surrounding_vehicles = other_information["surrounding_vehicles"]
            # excepted_risk = other_information["excepted_risk"]
            total_risk = other_information["total_risk"]
            collision_flag = other_information["collision_flag"]
            Adversarial_flag = other_information["Adversarial_flag"]


            step_data = env.record()
            # step_data 加入t
            step_data["time"] = t
            data_records.append(step_data)


               
            if TV_action.item() == env.action_space.nvec[1] - 3:
                av_left_change.append(1)
                
            if TV_action.item() == env.action_space.nvec[1] - 2:
                av_right_change.append(1)
                
            av_total_risk.append(total_risk)

            time_step += 1
            # if epsilon > 0.5:
            if collision_rate < 0.02:
            # if collision_rate > 1.2:
                if Adversarial_flag:
                    adversarial_counts += 1
                    # BV_reward = BV_reward + excepted_risk[BV_candidate_index]

                    BV_agent.buffer.states.append(observation)
                    BV_agent.buffer.actions.append(BV_action)
                    BV_agent.buffer.logprobs.append(BV_action_logprob)
                    BV_agent.buffer.state_values.append(BV_state_val)
                    BV_agent.buffer.rewards.append(BV_reward)
                    BV_agent.buffer.is_terminals.append(done)

                    BV_ep_reward += BV_reward
                    if (adversarial_counts+1) % update_timestep == 0:
                        BV_agent.update()

                # if collision_flag:
                #     #print("Collision detected")
                #     collision_counts += 1
                #     BV_ep_reward = round(BV_ep_reward, 4)
                #     log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, collision_counts, adversarial_counts,
                #                                           BV_ep_reward))
                #     # print("BV--> Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                #     #                                                                               log_avg_reward))
                #     log_f.flush()
                #     if (collision_counts + 1) % save_model_freq == 0:
                #         BV_agent.save(checkpoint_path)
                #     BV_ep_reward = 0



            # done = terminated
            else:
                TV_ep_reward += TV_reward
                TV_reward = torch.tensor([TV_reward], device=device)
                if done:
                    next_state = None
                else:
                    next_state = observation
                TV_agent.memory.push(state, TV_action, next_state, TV_reward)
                state = next_state

                TV_agent.learn_model()
                TV_agent.updateTargetNetwork()

            if done:

                av_running_time = env.getRunningTime()
                av_distance = env.getDistance()


                env.closeEnvConnection()

                break

            env.move_gui()

        #     #print("Collision detected")

        BV_ep_reward = round(BV_ep_reward, 4)
        log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, total_collision_counts, adversarial_counts,
                                              BV_ep_reward))
        # print("BV--> Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
        #                                                                               log_avg_reward))
        log_f.flush()
        # if (adversarial_counts + 1) % save_model_freq == 0:
        #     BV_agent.save(checkpoint_path)


        # if (i_episode + 1) % 2 == 0:
        #     torch.save(TV_agent.policy_net.state_dict(), av_checkpoint_path)

        TV_ep_reward = round(TV_ep_reward, 4)
        if_collision = 0
        if collision_flag:
            if_collision = 1
            collision_counts +=1
            total_collision_counts +=1
        if collision_counts > 5 or collision_intervals > 30:
            collision_rate = collision_counts / (collision_intervals)
            epsilon = 1 - math.exp(0.05 * (1 - 1 / (collision_rate + 1e-3)))
            collision_intervals = 0
            collision_counts = 0

        file_name = f"vehicle_data_{i_episode}.json"
        file_path = os.path.join(log_dir, file_name)

        
        try:
            with open(file_path, "w") as f1:
                json.dump(data_records, f1, indent=4)
            print(f"Data saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

        av_speed_avg = np.mean(av_speed)
        av_acc_avg = np.sum(av_acceleration) / (t+1)
        av_dece_avg = np.sum(av_deceleration) /(t+1)
        av_left_avg = np.sum(av_left_change) / (t + 1)
        av_right_avg = np.sum(av_right_change) / (t + 1)
        av_total_risk_avg = np.mean(av_total_risk)

        av_log_f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, TV_ep_reward, if_collision,
                                                                                collision_rate, adversarial_counts,
                                                                                epsilon, av_speed_avg,
                                                                                av_total_risk_avg, av_distance,
                                                                                av_running_time, av_acc_avg,
                                                                                av_dece_avg, av_left_avg, av_right_avg))
        av_log_f.flush()
        # print(
        #     'Ep={}, Step={},R={},C_F={},C_R = {},Adv_c={},epsilon={},SPD={},Risk={},Dis={},T={},Acc={},Dec={},Left={},Right={}'.format(
        #         i_episode, time_step, TV_ep_reward, if_collision, round(collision_rate, 4), adversarial_counts,
        #         round(epsilon, 4), round(av_speed_avg, 4),
        #         round(av_total_risk_avg, 4), round(av_distance, 4), round(av_running_time, 4), round(av_acc_avg, 4),
        #         round(av_dece_avg, 4), round(av_left_avg, 4), round(av_right_avg, 4)))

        i_episode += 1
        collision_intervals +=1


    log_f.close()
    av_log_f.close()
    torch.save(TV_agent.policy_net.state_dict(), av_checkpoint_path)
    BV_agent.save(checkpoint_path)
    #print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



if __name__ == '__main__':

    train()
#
# def main():
# 	ra = [i for i in range(5)]
# 	proces = []
# 	for i in ra:
# 	    proc = Process(target=train, args=(str(i+1),))
# 	    proces.append(proc)
# 	    proc.start()
#
# 	# complete
# 	for proc in proces:
# 	    proc.join()
#
# if __name__ == '__main__':
# 	main()
