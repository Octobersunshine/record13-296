import warnings
import numpy as np


def _normalize_std(std: float) -> float:
    if std < 0:
        warnings.warn(f"标准差为负数（{std}），已自动取绝对值", UserWarning, stacklevel=3)
        return abs(std)
    return std


class NormalDistributionService:
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int | None = None):
        self.mean = mean
        self.std = _normalize_std(std)
        self._rng = np.random.default_rng(seed)

    def generate(self, size: int = 1) -> np.ndarray:
        if size <= 0:
            raise ValueError("样本数量必须为正整数")
        return self._rng.normal(loc=self.mean, scale=self.std, size=size)

    def generate_single(self) -> float:
        return float(self._rng.normal(loc=self.mean, scale=self.std))

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def update_params(self, mean: float | None = None, std: float | None = None) -> None:
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = _normalize_std(std)

    def stats(self, samples: np.ndarray) -> dict:
        return {
            "count": len(samples),
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        }


def generate_normal(
    mean: float = 0.0,
    std: float = 1.0,
    size: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    std = _normalize_std(std)
    if size <= 0:
        raise ValueError("样本数量必须为正整数")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=size)


if __name__ == "__main__":
    service = NormalDistributionService(mean=100, std=15, seed=42)
    samples = service.generate(size=1000)
    stats = service.stats(samples)
    print("生成 1000 个正态分布随机数（均值=100, 标准差=15）：")
    print(f"  样本均值: {stats['mean']:.4f}")
    print(f"  样本标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
