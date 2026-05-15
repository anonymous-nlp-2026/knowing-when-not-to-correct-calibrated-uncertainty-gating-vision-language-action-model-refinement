"""ACI 符号修复验证测试。"""
import sys
sys.path.insert(0, '/root/acr-vla-conformal-refinement')
import torch
from src.aci import AdaptiveConformalInference

def test_aci_sign():
    alpha_target = 0.1
    gamma = 0.1
    
    initial_alpha = alpha_target
    print(f"Initial alpha: {initial_alpha}")
    
    dummy_mean = torch.zeros(6)
    dummy_cov = torch.eye(6)
    
    # 场景 1: 全覆盖 (err=0) -> alpha 应增大（收紧 conformal set）
    aci_cover = AdaptiveConformalInference(alpha_target=alpha_target, gamma=gamma)
    expert_close = torch.zeros(6)  # mahal dist = 0, radius=100 -> 覆盖
    result = aci_cover.update(expert_close, dummy_mean, dummy_cov, conformal_radius=100.0)
    print(f"After cover (err=0): alpha={result['alpha']:.4f}, err={result['err']}")
    assert result['err'] == 0.0, f"Expected err=0, got {result['err']}"
    assert result['alpha'] > initial_alpha, f"After cover, alpha should increase (tighten). Got {result['alpha']:.4f} <= {initial_alpha}"
    print("  Cover test passed: alpha increased (tightened)")
    
    # 场景 2: 全 miss (err=1) -> alpha 应减小（放松 conformal set）
    aci_miss = AdaptiveConformalInference(alpha_target=alpha_target, gamma=gamma)
    expert_far = torch.ones(6) * 10.0  # mahal dist = sqrt(6)*10 >> 0.001
    result = aci_miss.update(expert_far, dummy_mean, dummy_cov, conformal_radius=0.001)
    print(f"After miss (err=1): alpha={result['alpha']:.4f}, err={result['err']}")
    assert result['err'] == 1.0, f"Expected err=1, got {result['err']}"
    assert result['alpha'] < initial_alpha, f"After miss, alpha should decrease (relax). Got {result['alpha']:.4f} >= {initial_alpha}"
    print("  Miss test passed: alpha decreased (relaxed)")
    
    # 场景 3: 连续 miss 不应导致 alpha 爆炸到极值
    aci_seq = AdaptiveConformalInference(alpha_target=alpha_target, gamma=0.005)
    for i in range(100):
        result = aci_seq.update(expert_far, dummy_mean, dummy_cov, conformal_radius=0.001)
    print(f"After 100 misses: alpha={result['alpha']:.4f}")
    assert 0.01 <= result['alpha'] <= 0.5, f"Alpha out of safe range: {result['alpha']}"
    print("  Stability test passed")
    
    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_aci_sign()
