"""
GeneralBacktest - 通用量化策略回测框架

支持特性：
- 灵活的调仓时间（不固定频率）
- 向量化计算（高性能）
- 丰富的性能指标（15+）
- 多样化的可视化（8+图表）
- T+0 日内回转回测支持

使用示例：
    >>> from GeneralBacktest import GeneralBacktest
    >>> bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")
    >>> results = bt.run_backtest(weights_data, price_data, ...)

T+0 回测示例：
    >>> from GeneralBacktest import TBacktest
    >>> tb = TBacktest(start_date="2024-01-01", end_date="2024-12-31")
    >>> results = tb.run_t0_backtest(weights_data, price_data, ...)
"""

# 配置matplotlib支持中文显示（Windows兼容）
try:
    import matplotlib.pyplot as plt
    # 设置中文字体（按优先级）
    plt.rcParams['font.sans-serif'] = [
        'SimHei',           # Windows黑体
        'Microsoft YaHei',  # Windows雅黑
        'Arial Unicode MS', # macOS
        'DejaVu Sans'       # Linux
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
except ImportError:
    pass  # 如果没有安装matplotlib，跳过配置

from .backtest import GeneralBacktest
from .tbacktest import TBacktest

__version__ = '1.2.0'
__author__ = 'Elen Young'
__all__ = ['GeneralBacktest', 'TBacktest']
