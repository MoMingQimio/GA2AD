import gym
import gym_sumo
from dqn_sumo_gym import Agent
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np


def train_agent(p1, p2, p3, p4, q1):
    p1 = round(p1, 1)
    p2 = round(p2, 1)
    p3 = round(p3, 1)
    p4 = round(p4, 1)
    q1 = round(q1, 1)
    # 创建强化学习环境
    env = gym.make("sumo-v0", render_mode="human", p1=p1, p2=p2, p3=p3, p4=p4, q1=q1)

    # 实例化代理
    agent = Agent("Agent", hp=f"_{p1}_{p2}_{p3}_{p4}_{q1}")

    # 训练代理
    agent.train_RL(env)

    # 返回训练完成后最后一百个episode的平均奖励
    return agent.mean_reward


# 参数空间
pbounds = {'p1': (0, 1),
           'p2': (0, 1),
           'p3': (0, 1),
           'p4': (0, 1),
           'q1': (0, 1)}

# 实例化贝叶斯优化对象
optimizer = BayesianOptimization(
    f=train_agent,
    pbounds=pbounds,
    random_state=1,  # 设置随机种子
)

# 采集函数
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# 开始优化
optimizer.maximize(
    init_points=5,  # 初始化点的数量
    n_iter=10,  # 迭代次数
    acquisition_function=utility  # 使用创建的UtilityFunction实例
)

#
# # 保存每次迭代后的采集函数图像
# for i, res in enumerate(optimizer.res):
#     # 绘制采集函数图像
#     x = np.linspace(0, 1, 1000).reshape(-1, 1)
#     utility_values = utility.utility(x, optimizer._gp, 0)
#     fig, ax = plt.subplots()
#     ax.plot(x, utility_values, label='Utility Function')
#     ax.scatter(res['params']['p1'], res['target'], color='red', label='Next Sampling Point')
#     ax.set_title(f"Iteration {i+1}")
#     ax.legend()
#     plt.xlabel('p1')
#     plt.ylabel('Utility Value')
#     plt.savefig(f"bayesian_optimization_iteration_{i}.png")
#     plt.close(fig)

# res: 是一个列表，存储了每次迭代后的参数组合、目标函数值以及其他相关信息。每次迭代后，贝叶斯优化算法会将新的参数组合及其对应的目标函数值添加到 res 列表中。
# max: 是一个字典，存储了搜索过程中目标函数取得的最大值及其对应的参数组合。通过调用 max 方法，你可以获取搜索过程中目标函数取得的最大值以及对应的参数组合
# 输出优化结果
# print("优化结果为:", optimizer.max)
# print("优化过程列表:", optimizer.res)


# def visualize_results(results):
#     # 提取参数和对应的平均奖励值
#     params = [(round(p1, 1), round(p2, 1), round(p3, 1), round(p4, 1), round(q1, 1)) for
#               (p1, p2, p3, p4, q1, mean_reward) in results]
#     params = list(map(str, params))  # 将参数转换为字符串类型
#     mean_rewards = [mean_reward for (p1, p2, p3, p4, q1, mean_reward) in results]
#
#     # 绘制折线图
#     plt.figure(figsize=(10, 6))
#     plt.plot(mean_rewards, marker='o')
#     plt.xticks(range(len(params)), params, rotation=45)
#     plt.xlabel('Parameters')
#     plt.ylabel('Mean Reward')
#     plt.title('Mean Reward vs Parameters')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


results = optimizer.res
# visualize_results(results)
import json

# 将优化结果转换为JSON字符串
results_json = json.dumps(optimizer.res, indent=4)

# 将JSON字符串写入文件
with open("bayesian_optimization_results.json", "w") as f:
    f.write(results_json)

print("优化结果已保存到 'bayesian_optimization_results.json'")

