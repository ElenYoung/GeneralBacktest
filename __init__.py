"""
GeneralBacktest - 通用量化策略回测框架

支持特性：
- 灵活的调仓时间（不固定频率）
- 向量化计算（高性能）
- 丰富的性能指标（15+）
- 多样化的可视化（8+图表）
"""

from .backtest import GeneralBacktest
from .utils import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    calculate_all_metrics
)

__version__ = '1.0.0'
__all__ = ['GeneralBacktest']
