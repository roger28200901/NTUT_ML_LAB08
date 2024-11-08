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

獎勵函數 $R(s_t)$ 由以下幾個部分組成：

-   位置獎勵 $R_{position}$：
    $$R_{position} = 5.0 \cdot \frac{x_t - (-1.2)}{0.6 - (-1.2)}$$

-   速度獎勵 $R_{velocity}$：

    $$
    R_{velocity} = 3.0 \cdot \frac{|\dot{x}t|}{0.07} \cdot \begin{cases}
    2.0 & \text{if } \dot{x}t > 0 \text{ and } x_t > -0.4 \\
    1.0 & \text{otherwise}
    \end{cases}
    $$

-   接近頂部獎勵 $R_{top}$：

    $$
    R_{top} = \begin{cases}
    1.5 \cdot (R_{position} + R_{velocity}) & \text{if } x_t \geq -0.2 \\
    R_{position} + R_{velocity} & \text{otherwise}
    \end{cases}
    $$

-   目標獎勵 $R_{goal}$：

    $$
    R_{goal} = \begin{cases}
    20.0 & \text{if } x_t \geq 0.5 \\
    R_{top} & \text{otherwise}
    \end{cases}
    $$

-   低位置懲罰 $R_{penalty}$：
    $$
    R_{penalty} = \begin{cases}
    -1.0 & \text{if } |\dot{x}t| < 0.001 \text{ and } x_t < -0.8 \\
    0 & \text{otherwise}
    \end{cases}
    $$

最後，總獎勵被限制在 [-3, 20] 範圍內：
$$R_{final}(s_t) = \text{clip}(R_{goal} + R_{penalty}, -3, 20)$$

這個優化後的獎勵函數特點：

-   提供更強的位置獎勵（權重為 5.0）
-   在上坡時給予雙倍速度獎勵
-   接近頂部時（x ≥ -0.2）提供 1.5 倍獎勵
-   達到目標時給予較大獎勵（20.0）
-   懲罰在低位置停滯不前的情況
-   更大的獎勵範圍（-3 到 20）以提供更強的學習信號
