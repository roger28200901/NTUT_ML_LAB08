import gym
from RL_brain import DeepQNetwork
from reward_functions import calculate_mountain_car_reward
import warnings
import os
import numpy as np
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_model(n_episodes=10, render=True):
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # 初始化 DQN，使用與訓練時相同的參數
    RL = DeepQNetwork(
        n_actions=3,
        n_features=2,
        learning_rate=0.001,
        e_greedy=1,  # 測試時設為1，完全使用最優策略
        replace_target_iter=200,
        memory_size=20000,
        batch_size=64,
        e_greedy_increment=None,
        output_graph=False
    )

    # 載入已訓練的模型
    try:
        RL.load_model()
        print("成功載入模型")
    except:
        print("載入模型失敗！請確認模型文件存在")
        return

    success_count = 0  # 記錄成功次數
    total_steps_list = []  # 記錄每回合步數
    
    for episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 1000  # 設置最大步數限制
        
        while steps < max_steps:
            if render:
                env.render()
            
            # 選擇動作
            action = RL.choose_action(observation)
            
            # 執行動作
            observation_, reward, done, _ = env.step(action)
            position, velocity = observation_
            
            # 使用導入的獎勵函數計算獎勵
            reward = calculate_mountain_car_reward(position, velocity, env.unwrapped.goal_position)
            
            if position >= env.unwrapped.goal_position:
                done = True
            
            steps += 1
            total_reward += reward
            
            if done:
                if position >= env.unwrapped.goal_position:
                    success_count += 1
                total_steps_list.append(steps)
                print(f'回合 {episode + 1}: 步數 = {steps}, '
                      f'最終位置 = {position:.3f}, '
                      f'最終速度 = {velocity:.3f}, '
                      f'總獎勵 = {total_reward:.1f}, '
                      f'是否成功 = {"是" if position >= env.unwrapped.goal_position else "否"}')
                break
            
            observation = observation_

        if steps >= max_steps:
            total_steps_list.append(max_steps)
            print(f'回合 {episode + 1}: 達到最大步數限制 (位置 = {position:.3f})')

    # 輸出統計信息
    print("\n測試結果統計：")
    print(f"成功率: {success_count/n_episodes*100:.1f}%")
    print(f"平均步數: {np.mean(total_steps_list):.1f}")
    print(f"最少步數: {np.min(total_steps_list)}")
    print(f"最多步數: {np.max(total_steps_list)}")

    env.close()

if __name__ == '__main__':
    test_model(n_episodes=100, render=False)