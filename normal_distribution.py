import warnings
import numpy as np


def _normalize_std(std: float) -> float:
    if std < 0:
        warnings.warn(f"标准差为负数（{std}），已自动取绝对值", UserWarning, stacklevel=3)
        return abs(std)
    return std


def _normalize_uniform_range(low: float, high: float) -> tuple[float, float]:
    if low > high:
        warnings.warn(
            f"均匀分布上下限颠倒（low={low}, high={high}），已自动交换",
            UserWarning,
            stacklevel=3,
        )
        return high, low
    return low, high


def _normalize_scale(scale: float) -> float:
    if scale < 0:
        warnings.warn(f"尺度参数为负数（{scale}），已自动取绝对值", UserWarning, stacklevel=3)
        return abs(scale)
    return scale


class RandomDistributionService:
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        size: int = 1,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("样本数量必须为正整数")
        std = _normalize_std(std)
        return self._rng.normal(loc=mean, scale=std, size=size)

    def generate_normal_single(self, mean: float = 0.0, std: float = 1.0) -> float:
        std = _normalize_std(std)
        return float(self._rng.normal(loc=mean, scale=std))

    def generate_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int = 1,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("样本数量必须为正整数")
        low, high = _normalize_uniform_range(low, high)
        return self._rng.uniform(low=low, high=high, size=size)

    def generate_uniform_single(self, low: float = 0.0, high: float = 1.0) -> float:
        low, high = _normalize_uniform_range(low, high)
        return float(self._rng.uniform(low=low, high=high))

    def generate_exponential(
        self,
        scale: float = 1.0,
        size: int = 1,
    ) -> np.ndarray:
        if size <= 0:
            raise ValueError("样本数量必须为正整数")
        scale = _normalize_scale(scale)
        return self._rng.exponential(scale=scale, size=size)

    def generate_exponential_single(self, scale: float = 1.0) -> float:
        scale = _normalize_scale(scale)
        return float(self._rng.exponential(scale=scale))

    def stats(self, samples: np.ndarray) -> dict:
        return {
            "count": len(samples),
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        }


class NormalDistributionService:
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int | None = None):
        self.mean = mean
        self.std = _normalize_std(std)
        self._service = RandomDistributionService(seed=seed)
        self._rng = self._service._rng

    def generate(self, size: int = 1) -> np.ndarray:
        return self._service.generate_normal(mean=self.mean, std=self.std, size=size)

    def generate_single(self) -> float:
        return self._service.generate_normal_single(mean=self.mean, std=self.std)

    def set_seed(self, seed: int) -> None:
        self._service.set_seed(seed)
        self._rng = self._service._rng

    def update_params(self, mean: float | None = None, std: float | None = None) -> None:
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = _normalize_std(std)

    def stats(self, samples: np.ndarray) -> dict:
        return self._service.stats(samples)


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


def generate_uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    low, high = _normalize_uniform_range(low, high)
    if size <= 0:
        raise ValueError("样本数量必须为正整数")
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=size)


def generate_exponential(
    scale: float = 1.0,
    size: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    scale = _normalize_scale(scale)
    if size <= 0:
        raise ValueError("样本数量必须为正整数")
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=scale, size=size)


if __name__ == "__main__":
    service = RandomDistributionService(seed=42)

    print("=" * 60)
    print("【正态分布】 均值=100, 标准差=15, 样本数=1000")
    samples = service.generate_normal(mean=100, std=15, size=1000)
    s = service.stats(samples)
    print(f"  理论均值=100.0000, 样本均值={s['mean']:.4f}")
    print(f"  理论标准差=15.0000, 样本标准差={s['std']:.4f}")
    print(f"  范围: [{s['min']:.2f}, {s['max']:.2f}]")

    print()
    print("=" * 60)
    print("【均匀分布】 下限=10, 上限=20, 样本数=1000")
    samples = service.generate_uniform(low=10, high=20, size=1000)
    s = service.stats(samples)
    print(f"  理论均值=15.0000, 样本均值={s['mean']:.4f}")
    print(f"  理论标准差≈2.8868, 样本标准差={s['std']:.4f}")
    print(f"  范围: [{s['min']:.4f}, {s['max']:.4f}]")

    print()
    print("=" * 60)
    print("【指数分布】 尺度=5.0 (即均值=5.0), 样本数=1000")
    samples = service.generate_exponential(scale=5.0, size=1000)
    s = service.stats(samples)
    print(f"  理论均值=5.0000, 样本均值={s['mean']:.4f}")
    print(f"  理论标准差=5.0000, 样本标准差={s['std']:.4f}")
    print(f"  范围: [{s['min']:.4f}, {s['max']:.2f}]")

    print()
    print("=" * 60)
    print("【兼容性测试】 NormalDistributionService (旧接口)")
    legacy = NormalDistributionService(mean=0, std=1, seed=42)
    samples = legacy.generate(size=5)
    print(f"  前5个样本: {samples}")
    print("  ✅ 旧接口仍可用")

    print()
    print("=" * 60)
    print("【便捷函数测试】")
    print(f"  generate_normal(0,1,3) = {generate_normal(0, 1, 3, seed=0)}")
    print(f"  generate_uniform(0,10,3) = {generate_uniform(0, 10, 3, seed=0)}")
    print(f"  generate_exponential(2,3) = {generate_exponential(2, 3, seed=0)}")
