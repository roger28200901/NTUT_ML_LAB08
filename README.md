# NTUT_ML_LAB08

This is MountainCar Lab. For NTUT_ML class

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
-   避免過長的回合（超時懲罰）
