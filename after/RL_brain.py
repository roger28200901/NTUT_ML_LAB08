"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,      # 動作空間大小
            n_features,     # 狀態特徵數量
            learning_rate=0.001,     # 略微提高學習率
            reward_decay=0.95,        # 提高獎勵衰減率 (gamma)
            e_greedy=0.9,             # epsilon-貪婪策略中的 epsilon 最大值
            replace_target_iter=100,  # 減少目標網絡更新間隔
            memory_size=5000,        # 適當減少記憶體大小以加快採樣
            batch_size=32,            # 減小批次大小以加快訓練
            e_greedy_increment=0.002,  # 加快探索率增長
            output_graph=False,       # 是否輸出計算圖
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e)
                                  for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # 輸入
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')  # 用於計算損失
        
        with tf.variable_scope('eval_net'):
            # c_names 是用於存儲變量的集合
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, 32, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # 第一層使用 ELU 激活函數
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.elu(tf.matmul(self.s, w1) + b1)
                
            # 新增一層
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.elu(tf.matmul(l1, w2) + b2)

            # 輸出層
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            # 添加梯度裁剪
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            gradients = optimizer.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self._train_op = optimizer.apply_gradients(capped_gradients)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一層
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.elu(tf.matmul(self.s_, w1) + b1)
            
            # 新增中間層，與 eval_net 保持一致
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.elu(tf.matmul(l1, w2) + b2)

            # 輸出層
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        # 儲存轉換經驗到記憶體
        # s: 當前狀態
        # a: 執行的動作
        # r: 獲得的獎勵
        # s_: 下一個狀態

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # 根據當前狀態選擇動作
        # 使用 epsilon-greedy 策略：
        # epsilon 的機率選擇最優動作
        # 1-epsilon 的機率隨機選擇動作

        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 從經驗回放記憶體中學習
        # 1. 定期更新目標網絡
        # 2. 從記憶體中採樣批次數據
        # 3. 計算目標Q值
        # 4. 訓練評估網絡

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # 知道 DQN 正在執行其穩定性機制，保持學習過程的穩定性。
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """繪製學習曲線（僅顯示平滑曲線）"""
        import matplotlib.pyplot as plt
        
        # 創建新的圖形，使用深色背景主題
        plt.style.use('seaborn-darkgrid')
        fig = plt.figure(figsize=(12, 6))
        
        # 計算移動平均來平滑曲線
        window_size = 50
        smoothed_costs = pd.Series(self.cost_his).rolling(window=window_size, min_periods=1).mean()
        
        # 只繪製平滑後的曲線
        plt.plot(np.arange(len(smoothed_costs)), smoothed_costs, 'b-', linewidth=2, label='Training Cost')
        
        # 設置圖表樣式
        plt.title('DQN Training Cost over Time', fontsize=14, pad=15)
        plt.ylabel('Cost', fontsize=12)
        plt.xlabel('Training Steps', fontsize=12)
        plt.legend(loc='upper right')
        
        # 添加網格和優化視覺效果
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 顯示圖表
        plt.show()

    def save_model(self, save_path='./saved_model/model.ckpt'):
        """保存模型"""
        import os
        # 確保保存目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 創建 Saver 對象
        saver = tf.train.Saver()
        
        # 保存模型
        save_path = saver.save(self.sess, save_path)
        print(f"模型已保存至: {save_path}")

    def load_model(self, load_path='./saved_model/model.ckpt'):
        """加載模型"""
        # 創建 Saver 對象
        saver = tf.train.Saver()
        
        # 加載模型
        saver.restore(self.sess, load_path)
        print(f"模型已從 {load_path} 加載")
