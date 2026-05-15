"""
Adaptive Conformal Inference (ACI) 模块。

在线更新 conformal coverage level alpha，使得实际覆盖率逼近目标覆盖率。
同时计算 COC (Change of Conformity) 信号，用于检测分布偏移。

核心公式:
  alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
  其中 err_t = 1 if ||a_expert - a_mean||_Mahal > q_t else 0

COC 信号:
  COC_t = |alpha_t - alpha_{t-w:t}.mean()| / alpha_{t-w:t}.std()
  高 COC → 分布正在快速变化 → CRM 应更激进地校正
"""
import torch
import numpy as np
from collections import deque


class AdaptiveConformalInference:
    """在线自适应 conformal prediction，维护动态 alpha 和 COC 信号。"""
    
    def __init__(self, alpha_target: float = 0.1, gamma: float = 0.005, window_size: int = 50):
        """
        Args:
            alpha_target: 目标 miscoverage rate (1-alpha_target = coverage)
            gamma: alpha 更新步长
            window_size: COC 信号的滑动窗口大小
        """
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.window_size = window_size
        
        self.alpha = alpha_target  # 当前 alpha
        self.alpha_history = deque(maxlen=window_size)
        self.alpha_history.append(self.alpha)
    
    def update(self, expert_action: torch.Tensor, mean_action: torch.Tensor, 
               cov_matrix: torch.Tensor, conformal_radius: float) -> dict:
        """
        用一个新观测更新 alpha 并计算 COC。
        
        更新规则 (Gibbs & Candès 2021):
          alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
          - err=1 (miss): alpha 减小 -> 降低 miscoverage 容忍 -> 放大 conformal set
          - err=0 (cover): alpha 增大 -> 提高 miscoverage 容忍 -> 缩小 conformal set
        
        Args:
            expert_action: (D,) 专家动作（训练时可用）
            mean_action: (D,) 预测均值动作
            cov_matrix: (D, D) 协方差矩阵
            conformal_radius: 当前 conformal radius q_t
            
        Returns:
            dict with keys: alpha, coc, err, mahal_dist
        """
        # Mahalanobis distance
        diff = expert_action - mean_action
        cov_inv = torch.linalg.inv(cov_matrix)
        mahal_dist = torch.sqrt(diff @ cov_inv @ diff).item()
        
        # Coverage error
        err = 1.0 if mahal_dist > conformal_radius else 0.0
        
        # Update alpha
        self.alpha = self.alpha + self.gamma * (self.alpha_target - err)
        self.alpha = np.clip(self.alpha, 0.01, 0.5)  # 安全范围
        self.alpha_history.append(self.alpha)
        
        # COC signal
        coc = self._compute_coc()
        
        return {
            "alpha": self.alpha,
            "coc": coc,
            "err": err,
            "mahal_dist": mahal_dist,
        }
    
    def _compute_coc(self) -> float:
        """
        计算 Change of Conformity 信号。
        
        Returns:
            coc: float, 标准化的 alpha 变化率。高值 → 分布快速变化。
        """
        if len(self.alpha_history) < 3:
            return 0.0
        
        history = np.array(self.alpha_history)
        window_mean = history.mean()
        window_std = history.std()
        
        if window_std < 1e-8:
            return 0.0
        
        coc = abs(self.alpha - window_mean) / window_std
        return float(coc)
    
    def reset(self):
        """重置状态（新 episode 开始时）。"""
        self.alpha = self.alpha_target
        self.alpha_history.clear()
        self.alpha_history.append(self.alpha)
