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
    learning_rate=0.0005,        # 降低學習率，使學習更穩定
    reward_decay=0.95,           # 稍微降低折扣因子
    e_greedy=0.9,                # 略微降低初始探索率
    replace_target_iter=200,      # 增加目標網絡更新間隔
    memory_size=20000,           # 增加記憶體大小
    batch_size=32,               # 減小批次大小
    e_greedy_increment=0.001     # 逐漸減少探索，增加利用
)

total_steps = 0


# 主要訓練循環
for i_episode in range(100):

    observation = env.reset()    # 重置環境
    ep_r = 0                     # 初始化每集的總獎勵
    while True:
        # env.render()  # 註釋掉訓練時的渲染
        action = RL.choose_action(observation)  # 根據當前狀態選擇動作

        # 執行動作並獲取下一狀態、獎勵和結束標記
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        
        # 優化獎勵函數
        reward = 0
        # 基礎懲罰
        reward -= 1
        
        # 位置獎勵
        reward += math.pow(position + 0.5, 2)  # 鼓勵向右移動
        
        # 速度獎勵
        reward += math.pow(abs(velocity), 2)   # 鼓勵積累動能
        
        # 到達目標的額外獎勵
        if position >= env.unwrapped.goal_position:
            reward += 100

        # 儲存當前的轉換 (狀態, 動作, 獎勵, 新狀態)
        RL.store_transition(observation, action, reward, observation_)

        # 如果已經有足夠的記憶數據，開始學習
        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[
                0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        # 更新狀態
        observation = observation_
        total_steps += 1

        # 每100回合渲染一次
        if i_episode % 100 == 0:
            env.render()

    # 在訓練結束後保存模型
    RL.save_model()

RL.plot_cost()  # 繪製學習過程中的損失變化
