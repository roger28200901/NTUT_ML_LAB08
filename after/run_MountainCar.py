"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(
    n_actions=3,                  # 動作空間
    n_features=2,                 # 狀態特徵
    learning_rate=0.0005,         # 降低學習率
    reward_decay=0.95,            # 提高獎勵衰減率
    e_greedy=0.9,
    replace_target_iter=200,      # 更頻繁的更新目標網絡
    memory_size=10000,            # 增加記憶體大小
    batch_size=64,                # 增大批次大小
    e_greedy_increment=0.001      # 調整探索率增量
)

total_steps = 0


# 主要訓練循環
for i_episode in range(100):

    observation = env.reset()    # 重置環境
    ep_r = 0                     # 初始化每集的總獎勵
    while True:
        env.render()             # 渲染畫面

        action = RL.choose_action(observation)  # 根據當前狀態選擇動作

        # 執行動作並獲取下一狀態、獎勵和結束標記
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # 設定獎勵：鼓勵車子向右運動接近目標位置
        reward = abs(position - (-0.5))  # r在[0, 1]之間

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

RL.plot_cost()  # 繪製學習過程中的損失變化
