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

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0002,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    steps_in_episode = 0 # 新增：紀錄每回的步數
    while True:
        # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]

        RL.store_transition(observation, action, reward, observation_)

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

        observation = observation_
        total_steps += 1

RL.plot_cost()
