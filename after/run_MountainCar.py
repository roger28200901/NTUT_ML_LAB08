"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DeepQNetwork
import math
import warnings
import os
import numpy as np
from reward_functions import calculate_mountain_car_reward

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env = gym.make('MountainCar-v0')
env = env.unwrapped # 返回環境的原始版本，移除所有裝飾器 (wrappers)

print("action_space",env.action_space)
print("observation_space",env.observation_space)
print("observation_space.high",env.observation_space.high)
print("observation_space.low",env.observation_space.low)

RL = DeepQNetwork(
    n_actions=3,
    n_features=2,
    learning_rate=0.0015,        # 提升學習率 0.001 -> 0.0015
    e_greedy=0.9,
    replace_target_iter=200,     # 減少目標網絡更新頻率 300 -> 200
    memory_size=20000,          # 增加記憶體大小 500 -> 20000
    batch_size=64,              # 批次大小以增加更新頻率 32 -> 64
    e_greedy_increment=0.0002, 
    output_graph=True
)
# n_actions=3, n_features=2, learning_rate=0.0001, e_greedy=0.9,
#                   replace_target_iter=300, memory_size=10000,
#                   e_greedy_increment=0.00005

total_steps = 0


# 主要訓練循環
for i_episode in range(100):  # 增加訓練回合數
    observation = env.reset()    # 重置環境
    ep_r = 0                     # 初始化每集的總獎勵
    steps_in_episode = 0        # 新增：記錄每回合的步數
    max_steps = 1000  # 增加最大步數限制
    
    while True:  # 使用步數限制
        # env.render()  # 註釋掉訓練時的渲染
        action = RL.choose_action(observation)  # 根據當前狀態選擇動作

        # 執行動作並獲取下一狀態、獎勵和結束標記
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        
        # 使用導入的獎勵函數
        reward = calculate_mountain_car_reward(position, velocity, env.unwrapped.goal_position)

        if position >= env.unwrapped.goal_position:
            done = True

        # 儲存當前的轉換 (狀態, 動作, 獎勵, 新狀態)
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:  # 更頻繁地學習
          RL.learn()

        ep_r += reward
        steps_in_episode += 1    # 新增：步數計數

        # FIXME: here has some problem 
        if done:
            print(f'回合: {i_episode}, 步數: {steps_in_episode}, '
                  f'總獎勵: {ep_r:.1f}, 最終位置: {position:.3f}, '
                  f'最終速度: {velocity:.3f}, Epsilon: {RL.epsilon:.3f}')
            break

        # 更新狀態
        observation = observation_
        total_steps += 1

        # 添加早停機制
        if i_episode > 200 and ep_r < -500:  # 如果表現太差就重新開始訓練
            print("重置模型...")
            RL.reset_model()
            continue
          
    # 在訓練結束後保存模型
    RL.save_model()

RL.plot_cost()  # 繪製學習過程中的損失變化
