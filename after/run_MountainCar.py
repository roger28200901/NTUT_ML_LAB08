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
    learning_rate=0.001,        # 降低學習率，使學習更穩定
    e_greedy=0.9,                # 略微降低初始探索率
    replace_target_iter=300,      # 增加目標網絡更新間隔
    memory_size=3000,           # 增加記憶體大小
    batch_size=64,               # 批次大小
    e_greedy_increment=0.0002     # 逐漸減少探索，增加利用
)
# n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
#                   replace_target_iter=300, memory_size=3000,
#                   e_greedy_increment=0.0002

total_steps = 0


# 主要訓練循環
for i_episode in range(12):

    observation = env.reset()    # 重置環境
    ep_r = 0                     # 初始化每集的總獎勵
    steps_in_episode = 0        # 新增：記錄每回合的步數
    max_steps = 200  # 添加最大步數限制
    
    while True:
        # env.render()  # 註釋掉訓練時的渲染
        action = RL.choose_action(observation)  # 根據當前狀態選擇動作

        # 執行動作並獲取下一狀態、獎勵和結束標記
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        
        # 優化獎勵函數
        reward = 0
        # 基於位置的獎勵，越接近目標獎勵越大
        position_reward = (position - (-1.2)) / (0.6 - (-1.2))
        reward += 2.0 * position_reward  # 增加位置獎勵的權重

        # 基於速度的獎勵，鼓勵更大的速度
        velocity_reward = abs(velocity) / 0.07
        reward += velocity_reward

        # 如果正在往右移動且位置較高，給予更大的獎勵
        if velocity > 0 and position > -0.5:
            reward += 1.0

        # 到達目標時給予更大的獎勵
        if position >= env.unwrapped.goal_position:
            reward = 5.0
            done = True

        # 如果超過最大步數，給予懲罰並結束回合
        if steps_in_episode >= max_steps:
            reward = -1.0
            done = True

        reward = np.clip(reward, -1, 5)  # 調整獎勵範圍

        # 儲存當前的轉換 (狀態, 動作, 獎勵, 新狀態)
        RL.store_transition(observation, action, reward, observation_)

        # 如果已經有足夠的記憶數據，開始學習
        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        steps_in_episode += 1    # 新增：步數計數

        if done:
            get = '| Get' if observation_[
                0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            
            # 新增：打印詳細的統計資訊
            print(f'回合: {i_episode}, 總步數: {steps_in_episode}, '
                  f'總獎勵: {ep_r:.1f}, 最終位置: {observation_[0]:.3f}')
            break

        # 更新狀態
        observation = observation_
        total_steps += 1

        # 每100回合渲染一次
        # if i_episode % 100 == 0:
        #     env.render()

    # 在訓練結束後保存模型
    RL.save_model()

RL.plot_cost()  # 繪製學習過程中的損失變化
