import gym
from RL_brain import DeepQNetwork
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 創建環境
env = gym.make('MountainCar-v0')
env = env.unwrapped

# 創建 DQN 實例
RL = DeepQNetwork(
    n_actions=3,
    n_features=2,
    learning_rate=0.001,
    reward_decay=0.99,
    e_greedy=1.0,  # 測試時設為 1.0，總是選擇最優動作
    replace_target_iter=200,
    memory_size=10000,
    batch_size=32,
    e_greedy_increment=None
)

# 加載已保存的模型
RL.load_model()

# 測試模型
for episode in range(5):  # 測試 5 個回合
    observation = env.reset()
    total_reward = 0
    
    while True:
        env.render()
        
        # 選擇動作
        action = RL.choose_action(observation)
        
        # 執行動作
        observation_, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        if done:
            print(f'回合: {episode}, 總獎勵: {total_reward}')
            break
            
        observation = observation_

env.close()
