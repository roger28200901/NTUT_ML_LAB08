# NTUT_ML_LAB08

This is MountainCar Lab. For NTUT_ML class

## 專案結構

-   `RL_brain.py`: DQN 代理實作
-   `run_MountainCar.py`: 主要訓練腳本

## Network Architecture

DQN 使用三層神經網絡：

> 這種三層結構特別適合 MountainCar 問題，因為：
>
> -   需要理解狀態空間中的複雜模式
> -   需要學習長期規劃（爬山需要先往後退）
> -   需要整合位置和速度信息來做出決策
>
> 而且實驗結果表明，這種結構確實能帶來：
>
> -   更快的學習速度
> -   更穩定的訓練過程
> -   更好的最終性能

-   輸入層：狀態特徵（位置和速度）
-   隱藏層 1：64 個節點，使用 ELU 激活函數
-   隱藏層 2：32 個節點，使用 ELU 激活函數
-   輸出層：每個動作的 Q 值

## 調整重點

-   目標網絡更新（每 100 步）
-   梯度裁剪以提高穩定性
-   將 ReLU 改為 ELU 激活函數
    -   原始 ReLU 版本可能在訓練初期表現不穩定
    -   ELU 版本通常能提供：
    -   更快的收斂速度
    -   更穩定的學習曲線
    -   更好的最終性能
-   模型保存/加載功能
-   增強的視覺化工具

## Reward Function

對於狀態 $s_t = (x_t, \dot{x}t)$，其中：

-   $x_t$ 是位置，範圍 $[-1.2, 0.6]$
-   $\dot{x}t$ 是速度，範圍 $[-0.07, 0.07]$

獎勵函數 $R(s_t)$ 可以表示為：
$$R(s_t) = R_{position} + R_{velocity} + R_{direction} + R_{goal} + R_{timeout}$$

其中各項為：

-   位置獎勵 $R_{position}$：
    $$R_{position} = 2.0 \cdot \frac{x_t - (-1.2)}{0.6 - (-1.2)}$$

-   速度獎勵 $R_{velocity}$：
    $$R_{velocity} = \frac{|\dot{x}t|}{0.07}$$

-   方向獎勵 $R_{direction}$：

    $$
    R_{direction} = \begin{cases}
    1.0 & \text{if } \dot{x}t > 0 \text{ and } x_t > -0.5 \\
    0 & \text{otherwise}
    \end{cases}
    $$

-   目標獎勵 $R_{goal}$：

    $$
    R_{goal} = \begin{cases}
    5.0 & \text{if } x_t \geq 0.5 \\
    0 & \text{otherwise}
    \end{cases}
    $$

最後，總獎勵被限制在 [-1, 5] 範圍內：
$$R_{final}(s_t) = \text{clip}(R(s_t), -1, 5)$$

這個獎勵函數的設計考慮了：

-   加強向目標移動的獎勵（位置獎勵權重為 2.0）
-   鼓勵積累動能（速度獎勵）
-   鼓勵向右移動（方向獎勵）
-   達到目標時給予較大獎勵（目標獎勵為 5.0）
