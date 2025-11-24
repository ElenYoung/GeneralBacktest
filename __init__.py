"""
GeneralBacktest - 通用量化策略回测框架

支持特性：
- 灵活的调仓时间（不固定频率）
- 向量化计算（高性能）
- 丰富的性能指标（15+）
- 多样化的可视化（8+图表）
"""

# 尝试相对导入（作为包时）和绝对导入（直接导入时）
try:
    from .backtest import GeneralBacktest

except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from backtest import GeneralBacktest

    except ImportError as e:
        import sys
        import os
        # 添加当前目录到路径
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backtest import GeneralBacktest


__version__ = '1.0.0'
__all__ = ['GeneralBacktest']
