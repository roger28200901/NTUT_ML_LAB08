import numpy as np

def calculate_mountain_car_reward(position, velocity, goal_position):
    """
    優化後的獎勵函數，提供更強的位置和速度獎勵
    """
    reward = 0
    
    # 增加基於位置的獎勵
    position_reward = (position - (-1.2)) / (0.6 - (-1.2))
    reward += 5.0 * position_reward  # 增加位置獎勵的權重
    
    # 優化速度獎勵
    velocity_reward = abs(velocity) / 0.07
    if velocity > 0 and position > -0.4:
        velocity_reward *= 2.0  # 在上坡時給予更多速度獎勵
    reward += 3.0 * velocity_reward
    
    # 特殊位置的額外獎勵
    if position >= -0.2:
        reward *= 1.5  # 接近頂部時給予額外獎勵
    
    # 達到目標的獎勵
    if position >= goal_position:
        reward = 20.0  # 增加目標獎勵
    
    # 懲罰在低位置停留
    if abs(velocity) < 0.001 and position < -0.8:
        reward -= 1.0
    
    # 限制獎勵範圍
    reward = np.clip(reward, -3, 20)
    
    return reward 