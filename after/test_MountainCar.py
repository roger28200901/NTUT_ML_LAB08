import gym
from RL_brain import DeepQNetwork
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_mountaincar():
    # 創建環境
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # 創建 DQN 實例（使用與訓練時相同的參數）
    RL = DeepQNetwork(
        n_actions=3,
        n_features=2,
        learning_rate=0.0005,
        e_greedy=1.0,            # 測試時設為 1.0，完全使用學習到的策略
        replace_target_iter=200,
        memory_size=10000,
        batch_size=64,
        e_greedy_increment=None  # 測試時不需要探索
    )

    # 載入訓練好的模型
    RL.load_model()

    # 測試回合數
    n_episodes = 10
    
    for episode in range(n_episodes):
        # 重置環境
        observation = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 渲染環境
            env.render()
            
            # 選擇動作
            action = RL.choose_action(observation)
            
            # 執行動作
            observation_, reward, done, info = env.step(action)
            
            # 累計獎勵
            total_reward += reward
            steps += 1
            
            # 更新狀態
            observation = observation_
            
            # 如果回合結束
            if done:
                print(f'Episode {episode + 1}: Total Steps = {steps}, Final Position = {observation_[0]:.3f}')
                break

    env.close()

if __name__ == '__main__':
    test_mountaincar() 