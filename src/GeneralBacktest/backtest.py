"""
通用量化策略回测框架
支持灵活的调仓时间、向量化计算、丰富的性能指标和可视化功能
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings

# 动态导入 pandas（解决 Windows 兼容性问题）
try:
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Failed to import pandas: {e}")
    raise

# 导入 matplotlib 组件
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"WARNING: Failed to import matplotlib: {e}")
    print("Visualization features will not work.")

# 灵活导入utils模块
try:
    from .utils import (
        validate_data, align_dates, calculate_all_metrics,
        calculate_returns, calculate_max_drawdown, calculate_turnover,
        calculate_transaction_costs, calculate_monthly_returns, calculate_adjusted_weights
    )
except ImportError:
    from utils import (
        validate_data, align_dates, calculate_all_metrics,
        calculate_returns, calculate_max_drawdown, calculate_turnover,
        calculate_transaction_costs, calculate_monthly_returns, calculate_adjusted_weights
    )

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def _display_plot(save_path: str = None, show: bool = False) -> None:
    """Helper function to handle plot display/saving"""
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()  # Close figure to free memory
    elif show:
        plt.show()
    else:
        plt.close()  # Just close if not saving or showing


class GeneralBacktest:
    """
    通用量化策略回测类
    
    支持：
    - 灵活的调仓时间（不固定频率）
    - 向量化计算（避免循环）
    - 丰富的性能指标
    - 多样化的可视化
    """
    
    def __init__(self, start_date: str, end_date: str):
        """
        初始化回测框架
        
        Parameters:
        -----------
        start_date : str
            回测开始日期，格式 'YYYY-MM-DD'
        end_date : str
            回测结束日期，格式 'YYYY-MM-DD'
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.backtest_results = None
        self.metrics = None
        self.daily_nav = None
        self.daily_positions = None
        self.trade_records = None
        self.turnover_records = None
        self.benchmark_name = "Benchmark"
        
    def run_backtest(
        self,
        weights_data: pd.DataFrame,
        price_data: pd.DataFrame,
        buy_price: str,
        sell_price: str,
        adj_factor_col: str,
        close_price_col: str,
        date_col: str = 'date',
        asset_col: str = 'code',
        weight_col: str = 'weight',
        position_ratio_col: Optional[str] = None,
        rebalance_threshold: float = 0.005,
        transaction_cost: List[float] = [0.001, 0.001],
        initial_capital: float = 1.0,
        slippage: float = 0.0,
        benchmark_weights: Optional[pd.DataFrame] = None,
        benchmark_name: str = "Benchmark"
    ) -> Dict:
        """
        运行通用化回测框架

        Parameters:
        -----------
        weights_data : pd.DataFrame
            包含不同时间戳上资产权重的数据，必须包含 `date_col`、`asset_col` 和 `weight_col`
        price_data : pd.DataFrame
            包含多种价格的日频数据，包含 `date_col`、`asset_col`、`adj_factor_col`和各种价格字段
        buy_price : str
            买入价格字段名，如 'open'、'close'
        sell_price : str
            卖出价格字段名，如 'open'、'close'
        adj_factor_col : str
            累计复权因子字段名
        date_col : str
            日期列名
        asset_col : str
            资产列名
        weight_col : str
            权重列名
        position_ratio_col : str, optional
            仓位比例列名。如果指定，weights_data 中需要包含此列，表示每个调仓日的目标总仓位比例。
            如果为 None（默认），则使用满仓（权重归一化到1）。
            例如某天 position_ratio=0.8 表示80%仓位，20%现金。
        rebalance_threshold : float
            调仓阈值，如果某只标的的仓位变化绝对值不超过rebalance_threshold，则不进行调仓
        transaction_cost : list of float
            交易成本，格式为 [买入成本, 卖出成本]
        initial_capital : float
            初始资金
        slippage : float
            滑点率
        benchmark_weights : pd.DataFrame, optional
            基准权重数据，格式与weights_data相同
        benchmark_name : str
            基准名称

        Returns:
        --------
        dict
            回测结果字典
        """
        print("=" * 60)
        print("Start Backtesting...")
        print("=" * 60)

        self.benchmark_name = benchmark_name

        # 1. 数据预处理和验证
        ## 验证字段是否齐全
        weights_data, price_data, benchmark_weights = self._preprocess_data(
            weights_data, price_data, date_col, asset_col, weight_col,
            buy_price, sell_price, adj_factor_col, close_price_col, benchmark_weights,
            position_ratio_col=position_ratio_col
        )
        
        # 2. 计算调仓日和持仓
        rebalance_dates = sorted(weights_data[date_col].unique())
        print(f"  - The number of rebalance days: {len(rebalance_dates)}")
        print(f"  - The first day of rebalance: {rebalance_dates[0]}")
        print(f"  - The last day of rebalance: {rebalance_dates[-1]}")
        
        # 3. 生成每日净值和持仓
        daily_results = self._calculate_daily_nav(
            weights_data, price_data, rebalance_dates,
            date_col, asset_col, weight_col, 
            rebalance_threshold, transaction_cost, initial_capital, slippage=slippage
        )
        
        self.daily_nav = daily_results['nav_series']
        self.daily_positions = daily_results['positions_df']
        self.trade_records = daily_results['trade_records']
        self.turnover_records = daily_results['turnover_records']
        
        print(f"  - The number of trading days: {len(self.daily_nav)}")
        print(f"  - The number of Rebalance: {len(self.trade_records)}")
        
        # 4. 计算基准（如果提供）
        benchmark_nav = None
        if benchmark_weights is not None:
            

            benchmark_results = self._calculate_daily_nav(
                benchmark_weights, price_data, 
                sorted(benchmark_weights[date_col].unique()),
                date_col, asset_col, weight_col,
                rebalance_threshold, [0, 0],  # 基准不考虑交易成本
                initial_capital,
                slippage=0.0
            )
            benchmark_nav = benchmark_results['nav_series']
        else:
            print("There isn't benchmark weight data...")
        
        # 5. 计算评价指标
        self.metrics = calculate_all_metrics(
            nav_series=self.daily_nav,
            benchmark_nav=benchmark_nav,
            trade_dates=rebalance_dates,
            turnover_series=self.turnover_records['turnover'] if len(self.turnover_records) > 0 else None
        )
        
        # 6. 整理回测结果
        self.backtest_results = {
            'nav_series': self.daily_nav,
            'positions_df': self.daily_positions,
            'trade_records': self.trade_records,
            'turnover_records': self.turnover_records,
            'metrics': self.metrics,
            'benchmark_nav': benchmark_nav
        }
        
        print("\n" + "=" * 60)
        print("Backtest ")
        print("=" * 60)
        
        return self.backtest_results
    
    def _preprocess_data(
        self,
        weights_data: pd.DataFrame,
        price_data: pd.DataFrame,
        date_col: str,
        asset_col: str,
        weight_col: str,
        buy_price: str,
        sell_price: str,
        adj_factor_col: str,
        close_price_col: str,
        benchmark_weights: Optional[pd.DataFrame] = None,
        position_ratio_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        数据预处理和验证

        Parameters:
        -----------
        position_ratio_col : str, optional
            仓位比例列名。如果指定，weights_data 中需要包含此列。
            如果为 None（默认），则权重归一化到1（满仓）。
        """
        # 验证数据
        required_weight_cols = [date_col, asset_col, weight_col]
        if position_ratio_col is not None:
            required_weight_cols.append(position_ratio_col)
        validate_data(weights_data, required_weight_cols, "weights_data")
        validate_data(
            price_data, 
            [date_col, asset_col, buy_price, sell_price, adj_factor_col, close_price_col], 
            "price_data"
        )
        if benchmark_weights is not None:
            validate_data(benchmark_weights, [date_col, asset_col, weight_col], "benchmark_weights")
        
        # 转换日期格式
        weights_data = weights_data.copy()
        price_data = price_data.copy()
        weights_data[date_col] = pd.to_datetime(weights_data[date_col])
        price_data[date_col] = pd.to_datetime(price_data[date_col])
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights[date_col] = pd.to_datetime(benchmark_weights[date_col])
        
        # 筛选回测时间范围
        weights_data = weights_data[
            (weights_data[date_col] >= self.start_date) &
            (weights_data[date_col] <= self.end_date)
        ]
        price_data = price_data[
            (price_data[date_col] >= self.start_date) &
            (price_data[date_col] <= self.end_date)
        ]
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights[
                (benchmark_weights[date_col] >= self.start_date) &
                (benchmark_weights[date_col] <= self.end_date)
            ]
        
        # 计算复权价格（用于收益率计算）
        price_data['adj_buy_price'] = price_data[buy_price] * price_data[adj_factor_col]
        price_data['adj_sell_price'] = price_data[sell_price] * price_data[adj_factor_col]
        price_data['adj_close_price'] = price_data[close_price_col] * price_data[adj_factor_col]

        # 权重归一化处理
        weights_sum = weights_data.groupby(date_col)[weight_col].transform('sum')

        if position_ratio_col is not None:
            # 使用 position_ratio_col 指定的仓位比例
            # 先归一化到1，再乘以每日的 position_ratio
            position_ratio = weights_data[position_ratio_col]
            weights_data[weight_col] = np.where(
                weights_sum == 0,
                0.0,
                weights_data[weight_col] / weights_sum * position_ratio
            )
        else:
            # 默认满仓：归一化到1
            weights_data[weight_col] = np.where(
                weights_sum == 0,
                0.0,
                weights_data[weight_col] / weights_sum
            )

        # 基准权重始终使用满仓（归一化到1）
        if benchmark_weights is not None:
            bench_weights_sum = benchmark_weights.groupby(date_col)[weight_col].transform('sum')
            benchmark_weights[weight_col] = np.where(
                bench_weights_sum == 0,
                0.0,
                benchmark_weights[weight_col] / bench_weights_sum
            )
        return weights_data, price_data, benchmark_weights
    

    def _calculate_daily_nav(
            self,
            weights_data: pd.DataFrame,
            price_data: pd.DataFrame,
            rebalance_dates: List,
            date_col: str,
            asset_col: str,
            weight_col: str,
            rebalance_threshold: float,
            transaction_cost: List[float], # [buy_rate, sell_rate]
            initial_capital: float,
            slippage: float = 0.0
        ) -> Dict:
        
        # --- 1. 数据准备 (使用Pivot极速查找) ---
        all_dates = sorted(price_data[date_col].unique())
        p_close = price_data.pivot(index=date_col, columns=asset_col, values='adj_close_price')
        p_buy = price_data.pivot(index=date_col, columns=asset_col, values='adj_buy_price')
        p_sell = price_data.pivot(index=date_col, columns=asset_col, values='adj_sell_price')
        
        current_nav = initial_capital
        current_positions = pd.Series(dtype=float) # 记录当前的【权重】
        
        nav_dict = {}
        trade_records = []
        turnover_records = []
        positions_records = []

        for i, date in enumerate(all_dates):
            # 每日价格切片
            daily_close = p_close.loc[date]
            if i > 0:
                prev_date = all_dates[i-1]
                prev_close = p_close.loc[prev_date]
            else:
                prev_close = pd.Series(dtype=float)

            is_rebalance = date in rebalance_dates
            
            # ============================================================
            # 场景 A: 调仓日 (Rebalance Day)
            # ============================================================
            if is_rebalance:
                # 1. 确定目标权重 & 缓冲带逻辑
                raw_target = weights_data[weights_data[date_col] == date].set_index(asset_col)[weight_col]
                target_weights = calculate_adjusted_weights(
                    weight_before=current_positions,
                    weight_after=raw_target,
                    rebalance_threshold=rebalance_threshold
                )
                
                # 2. 对齐资产索引，准备计算
                all_assets = current_positions.index.union(target_weights.index)
                w_old = current_positions.reindex(all_assets, fill_value=0)
                w_new = target_weights.reindex(all_assets, fill_value=0)
                
                # 3. 持仓分解 (Decomposition)
                # 任何时刻，w_new = w_kept + w_bought
                # 任何时刻，w_old = w_kept + w_sold
                
                # Kept: 新旧权重的交集（最小值），这部分是从昨天一直拿到今天的
                w_kept = np.minimum(w_old, w_new)
                
                # Bought: 目标比保留多的部分 (w_new - w_kept)
                w_bought = w_new - w_kept
                
                # Sold: 旧仓比保留多的部分 (w_old - w_kept)
                w_sold = w_old - w_kept
                
                # 过滤掉 0 值以提高效率
                w_kept = w_kept[w_kept > 0]
                w_bought = w_bought[w_bought > 0]
                w_sold = w_sold[w_sold > 0]

                # 4. 分段计算收益贡献 (Contribution)
                
                # --- 4.1 Sold 部分: 收益区间 [Prev_Close -> Sell_Price] ---
                # 考虑滑点: 卖得更便宜
                contrib_sold = 0.0
                if not w_sold.empty:
                    assets = w_sold.index
                    p_s = p_sell.loc[date].reindex(assets)
                    p_prev = prev_close.reindex(assets)
                    
                    # 执行价
                    p_exec_sell = p_s * (1 - slippage)
                    
                    # 收益率
                    r_sold = (p_exec_sell - p_prev) / p_prev
                    contrib_sold = (w_sold * r_sold).sum()

                    if np.any(p_prev == 0):
                        print(f"Warning: Zero previous close price on {date.date()} for assets: {p_prev[p_prev == 0].index.tolist()}")
                    if any(r_sold > 0.2):
                        print(f"Warning: High sell return on {date.date()} for assets: {r_sold[r_sold > 0.2].to_dict()}")
                        print(f"  - p_exec_sell: {p_exec_sell[r_sold > 0.2].to_dict()}")
                        print(f"  - p_prev: {p_prev[r_sold > 0.2].to_dict()}")

                # --- 4.2 Kept 部分: 收益区间 [Prev_Close -> Curr_Close] ---
                # 这部分没有交易滑点，也没有买卖价差，只有全天持有收益
                contrib_kept = 0.0
                if not w_kept.empty:
                    assets = w_kept.index
                    p_c = daily_close.reindex(assets)
                    p_prev = prev_close.reindex(assets)
                    
                    r_kept = (p_c - p_prev) / p_prev
                    contrib_kept = (w_kept * r_kept).sum()

                    if np.any(p_prev == 0):
                        print(f"Warning: Zero previous close price on {date.date()} for assets: {p_prev[p_prev == 0].index.tolist()}")
                    if any(r_kept > 0.2):
                        print(f"Warning: High kept return on {date.date()} for assets: {r_kept[r_kept > 0.2].to_dict()}")
                        print(f"  - p_c: {p_c[r_kept > 0.2].to_dict()}")
                        print(f"  - p_prev: {p_prev[r_kept > 0.2].to_dict()}")
                # --- 4.3 Bought 部分: 收益区间 [Buy_Price -> Curr_Close] ---
                # 考虑滑点: 买得更贵
                contrib_bought = 0.0
                if not w_bought.empty:
                    assets = w_bought.index
                    p_b = p_buy.loc[date].reindex(assets)
                    p_c = daily_close.reindex(assets)
                    
                    # 执行价
                    p_exec_buy = p_b * (1 + slippage)
                    
                    # 收益率 (日内收益)
                    r_bought = (p_c - p_exec_buy) / p_exec_buy
                    contrib_bought = (w_bought * r_bought).sum()

                    if np.any(p_exec_buy == 0):
                        print(f"Warning: Zero executed buy price on {date.date()} for assets: {p_exec_buy[p_exec_buy == 0].index.tolist()}")
                    if any(r_bought > 0.2):
                        print(f"Warning: High buy return on {date.date()} for assets: {r_bought[r_bought > 0.2].to_dict()}")
                        print(f"  - p_c: {p_c[r_bought > 0.2].to_dict()}")
                        print(f"  - p_exec_buy: {p_exec_buy[r_bought > 0.2].to_dict()}")
                # 5. 计算交易成本
                # 假设费率是对成交金额收取的
                cost_buy = w_bought.sum() * transaction_cost[0]
                cost_sell = w_sold.sum() * transaction_cost[1]
                total_fee = cost_buy + cost_sell
                
                # 6. 更新当日净值
                # 总收益 = 卖出部分收益 + 保留部分收益 + 买入部分日内收益 - 手续费
                total_return = contrib_sold + contrib_kept + contrib_bought - total_fee
                current_nav *= (1 + total_return)

                if total_return > 0.2:
                    print(f"Warning: High daily return {total_return:.2%} on {date.date()}")
                    print(f"  - contrib_sold: {contrib_sold:.2%}, contrib_kept: {contrib_kept:.2%}, contrib_bought: {contrib_bought:.2%}, total_fee: {total_fee:.2%}")
                    print(f"  - w_sold sum: {w_sold.sum():.2%}, w_kept sum: {w_kept.sum():.2%}, w_bought sum: {w_bought.sum():.2%}")
                   
                # 7. 计算当日收盘后的真实权重 (Weight Drift)
                # 我们不能简单令 current_positions = w_new，因为收盘时各资产涨幅不同。
                # 需要计算各部分在收盘时的"市值因子"。

                market_value_factors = pd.Series(0.0, index=all_assets)

                # (A) Kept 部分的期末市值因子: w * (1 + r_day)
                if not w_kept.empty:
                    assets = w_kept.index
                    r_day = (daily_close.reindex(assets) - prev_close.reindex(assets)) / prev_close.reindex(assets)
                    market_value_factors[assets] += w_kept * (1 + r_day)

                # (B) Bought 部分的期末市值因子: w * (1 + r_intraday)
                if not w_bought.empty:
                    assets = w_bought.index
                    p_exec_buy = p_buy.loc[date].reindex(assets) * (1 + slippage)
                    r_intraday = (daily_close.reindex(assets) - p_exec_buy) / p_exec_buy
                    market_value_factors[assets] += w_bought * (1 + r_intraday)

                # 归一化得到新的权重（相对于总资产，包含现金部分）
                # 注意：sold部分已经变现，不包含在期末持仓中
                # 期末权重 = 各资产市值因子 / (1 + total_return)
                # 这样可以保持现金仓位的存在
                if market_value_factors.sum() > 0:
                    current_positions = market_value_factors / (1 + total_return)
                    # 过滤掉极小值
                    current_positions = current_positions[current_positions > 1e-8]
                else:
                    current_positions = pd.Series(dtype=float)

                # 记录交易数据
                turnover = (w_bought.sum() + w_sold.sum()) / 2 # 单边
                turnover_records.append({'date': date, 'turnover': turnover})
                trade_records.append({
                    'date': date, 
                    'commission': total_fee, 
                    'return_sold': contrib_sold,
                    'return_bought': contrib_bought,
                    'return_kept': contrib_kept
                })

            # ============================================================
            # 场景 B: 非调仓日 (Non-Rebalance Day)
            # ============================================================
            else:
                if not current_positions.empty and i > 0:
                    assets = current_positions.index
                    p_c = daily_close.reindex(assets)
                    p_prev = prev_close.reindex(assets)
                    
                    # 计算常规日收益
                    asset_ret = (p_c - p_prev) / p_prev
                    port_ret = (current_positions * asset_ret).sum()
                    
                    current_nav *= (1 + port_ret)

                    if port_ret > 0.2:
                        print(f"Warning: High daily return {port_ret:.2%} on {date.date()} (Non-Rebalance Day)")
                        print(f"  - current_positions: {current_positions[current_positions != 0].to_dict()}")
                        print(f"  - p_c: {p_c[p_c != 0].to_dict()}")
                        print(f"  - p_prev: {p_prev[p_prev != 0].to_dict()}")
                        print(f"  - asset_ret: {asset_ret[asset_ret > 0.2].to_dict()}")
                    
                    # 自然飘移
                    current_positions = current_positions * (1 + asset_ret) / (1 + port_ret)
            
            # 记录
            nav_dict[date] = current_nav
            if not current_positions.empty:
                valid_pos = current_positions[current_positions > 1e-6]
                for asset, w in valid_pos.items():
                    positions_records.append({'date': date, 'asset': asset, 'weight': w})

        # 结果输出
        return {
            'nav_series': pd.Series(nav_dict, name='nav').sort_index(),
            'positions_df': pd.DataFrame(positions_records),
            'trade_records': pd.DataFrame(trade_records),
            'turnover_records': pd.DataFrame(turnover_records)
        }

    def get_metrics(self) -> pd.DataFrame:
        """
        获取性能指标
        
        Returns:
        --------
        pd.DataFrame
            性能指标表
        """
        if self.metrics is None:
            raise ValueError("请先运行 run_backtest()")
        
        # 转换为 DataFrame 便于查看
        metrics_df = pd.DataFrame([self.metrics]).T
        metrics_df.columns = ['值']
        
        return metrics_df
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """
        获取交易分析
        
        Returns:
        --------
        pd.DataFrame
            交易记录
        """
        if self.trade_records is None:
            raise ValueError("请先运行 run_backtest()")
        
        return self.trade_records
    
    def get_daily_positions(self) -> pd.DataFrame:
        """
        获取每日标的权重记录
        
        Returns:
        --------
        pd.DataFrame
            包含日期、资产、权重的每日持仓记录
            列：['date', 'asset', 'weight']
        """
        if self.daily_positions is None:
            raise ValueError("请先运行 run_backtest()")
        
        return self.daily_positions.copy()
    
    def get_position_matrix(self) -> pd.DataFrame:
        """
        获取每日标的权重矩阵（透视表格式）
        
        Returns:
        --------
        pd.DataFrame
            行索引为日期，列为资产代码，值为权重
            方便查看每日各标的的权重分布
        """
        if self.daily_positions is None:
            raise ValueError("请先运行 run_backtest()")
        
        if len(self.daily_positions) == 0:
            return pd.DataFrame()
        
        # 创建透视表：日期 x 资产
        position_matrix = self.daily_positions.pivot_table(
            index='date', 
            columns='asset', 
            values='weight', 
            fill_value=0
        )
        
        return position_matrix
    
    def get_position_changes(self) -> pd.DataFrame:
        """
        获取每日标的权重变化
        
        Returns:
        --------
        pd.DataFrame
            行索引为日期，列为资产代码，值为权重变化量
            正值表示增持，负值表示减持
        """
        if self.daily_positions is None:
            raise ValueError("请先运行 run_backtest()")
        
        position_matrix = self.get_position_matrix()
        
        if len(position_matrix) == 0:
            return pd.DataFrame()
        
        # 计算每日权重变化
        position_changes = position_matrix.diff()
        
        return position_changes
    
    def print_metrics(self) -> None:
        """
        打印全部策略表现指标
        
        将所有性能指标按分类美观地打印出来，包括：
        - 收益指标
        - 风险指标
        - 风险调整指标
        - 交易指标
        - 相对基准指标（如有）
        """
        if self.metrics is None:
            raise ValueError("请先运行 run_backtest()")
        
        print("\n" + "=" * 80)
        print("策略表现指标报告".center(80))
        print("=" * 80)
        
        # 1. 收益指标
        print("\n* 收益指标")
        print("-" * 80)
        if '累计收益率' in self.metrics:
            print(f"  累计收益率:        {self.metrics['累计收益率']:>12.2%}")
        if '年化收益率' in self.metrics:
            print(f"  年化收益率:        {self.metrics['年化收益率']:>12.2%}")
        
        # 2. 风险指标
        print("\nWARNING  风险指标")
        print("-" * 80)
        if '年化波动率' in self.metrics:
            print(f"  年化波动率:        {self.metrics['年化波动率']:>12.2%}")
        if '最大回撤' in self.metrics:
            print(f"  最大回撤:          {self.metrics['最大回撤']:>12.2%}")
        if '最大回撤开始日期' in self.metrics and self.metrics['最大回撤开始日期'] is not None:
            print(f"  最大回撤开始日期:  {str(self.metrics['最大回撤开始日期'])[:10]:>12}")
        if '最大回撤结束日期' in self.metrics and self.metrics['最大回撤结束日期'] is not None:
            print(f"  最大回撤结束日期:  {str(self.metrics['最大回撤结束日期'])[:10]:>12}")
        if '最大回撤持续天数' in self.metrics:
            print(f"  最大回撤持续天数:  {self.metrics['最大回撤持续天数']:>12.0f} 天")
        if 'VaR (95%)' in self.metrics:
            print(f"  VaR (95%):         {self.metrics['VaR (95%)']:>12.2%}")
        if 'CVaR (95%)' in self.metrics:
            print(f"  CVaR (95%):        {self.metrics['CVaR (95%)']:>12.2%}")
        
        # 3. 风险调整指标
        print("\n* 风险调整指标")
        print("-" * 80)
        if '夏普比率' in self.metrics:
            print(f"  夏普比率:          {self.metrics['夏普比率']:>12.4f}")
        if '索提诺比率' in self.metrics:
            print(f"  索提诺比率:        {self.metrics['索提诺比率']:>12.4f}")
        if '卡玛比率' in self.metrics:
            print(f"  卡玛比率:          {self.metrics['卡玛比率']:>12.4f}")
        if '胜率' in self.metrics:
            print(f"  胜率:              {self.metrics['胜率']:>12.2%}")
        
        # 4. 交易指标
        print("\n* 交易指标")
        print("-" * 80)
        if '交易次数' in self.metrics:
            print(f"  交易次数:          {self.metrics['交易次数']:>12.0f}")
        if '平均换手率' in self.metrics:
            print(f"  平均换手率:        {self.metrics['平均换手率']:>12.2%}")
        if '累计换手率' in self.metrics:
            print(f"  累计换手率:        {self.metrics['累计换手率']:>12.2%}")
        
        # 5. 相对基准指标（如果有）
        benchmark_metrics = ['基准累计收益率', '基准年化收益率', '超额收益', 
                           '年化超额收益', '信息比率', '跟踪误差']
        has_benchmark = any(metric in self.metrics for metric in benchmark_metrics)
        
        if has_benchmark:
            print("\n* 相对基准指标")
            print("-" * 80)
            if '基准累计收益率' in self.metrics:
                print(f"  基准累计收益率:    {self.metrics['基准累计收益率']:>12.2%}")
            if '基准年化收益率' in self.metrics:
                print(f"  基准年化收益率:    {self.metrics['基准年化收益率']:>12.2%}")
            if '超额收益' in self.metrics:
                print(f"  超额收益:          {self.metrics['超额收益']:>12.2%}")
            if '年化超额收益' in self.metrics:
                print(f"  年化超额收益:      {self.metrics['年化超额收益']:>12.2%}")
            if '信息比率' in self.metrics:
                print(f"  信息比率:          {self.metrics['信息比率']:>12.4f}")
            if '跟踪误差' in self.metrics:
                print(f"  跟踪误差:          {self.metrics['跟踪误差']:>12.2%}")
        
        print("\n" + "=" * 80 + "\n")
    
    def run_backtest_ETF(self,
        etf_db_config: dict,
        weights_data: pd.DataFrame,
        buy_price: str = 'open',
        sell_price: str = 'open',
        transaction_cost: List[float] = [0.001, 0.001],
        rebalance_threshold: float = 0.0,
        slippage: float = 0.0,
        position_ratio_col: Optional[str] = None,
        benchmark_weights: pd.DataFrame = None,
        benchmark_name: str = "Benchmark"
        ):
        from quantchdb import ClickHouseDatabase

        db = ClickHouseDatabase(config=etf_db_config, terminal_log=False)

        sql = f"""
        WITH
            prices as (
                SELECT code, date, open, close, amount/vol as vwap, adj_factor
                FROM etf.etf_day daily
                WHERE and(date>='{self.start_date.strftime(format='%Y-%m-%d')}',
                          date<='{self.end_date.strftime(format='%Y-%m-%d')}')
                ORDER BY code, date
            )

        SELECT * FROM prices
        """

        price_data = db.fetch(sql)
        price_data['date'] = pd.to_datetime(price_data['date'])

        results = self.run_backtest(
            weights_data=weights_data,
            price_data=price_data,
            buy_price=buy_price,
            sell_price=sell_price,
            adj_factor_col='adj_factor',
            close_price_col='close',
            rebalance_threshold=rebalance_threshold,
            slippage=slippage,
            position_ratio_col=position_ratio_col,
            transaction_cost=transaction_cost,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name
            )

        return results

    def run_backtest_stock(self,
        stock_db_config: dict,
        weights_data: pd.DataFrame,
        buy_price: str = 'open',
        sell_price: str = 'close',
        transaction_cost: List[float] = [0.001, 0.001],
        rebalance_threshold: float = 0.0,
        slippage: float = 0.0,
        position_ratio_col: Optional[str] = None,
        benchmark_weights: pd.DataFrame = None,
        benchmark_name: str = "Benchmark"
        ):
        from quantchdb import ClickHouseDatabase

        db = ClickHouseDatabase(config=stock_db_config, terminal_log=False)


        sql = f"""
        WITH
            prices as (
                SELECT  * 
                FROM stocks.daily_adj_tushare2
                WHERE and(date>='{self.start_date.strftime(format='%Y-%m-%d')}', 
                          date<='{self.end_date.strftime(format='%Y-%m-%d')}')
                ORDER BY code, date
            )

        SELECT * FROM prices
        """

        price_data = db.fetch(sql)
        price_data['date'] = pd.to_datetime(price_data['date'])

        results = self.run_backtest(
            weights_data=weights_data,
            price_data=price_data,
            buy_price=buy_price,
            sell_price=sell_price,
            adj_factor_col='adj_factor',
            close_price_col='close',
            rebalance_threshold=rebalance_threshold,
            slippage=slippage,
            position_ratio_col=position_ratio_col,
            transaction_cost=transaction_cost,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name
            )

        return results

    def run_backtest_with_cash(
        self,
        weights_data: pd.DataFrame,
        price_data: pd.DataFrame,
        initial_capital: float,
        buy_price: str,
        sell_price: str,
        close_price_col: str,
        date_col: str = 'date',
        asset_col: str = 'code',
        weight_col: str = 'weight',
        lot_size: int = 100,
        trade_critic: str = 'weight_desc',
        transaction_cost: List[float] = [0.001, 0.001],
        slippage: float = 0.0,
        benchmark_weights: Optional[pd.DataFrame] = None,
        benchmark_name: str = "Benchmark"
    ) -> Dict:
        """
        现金仓位回测：考虑实际股票价格和最小交易单位（手）的回测

        与 run_backtest 不同，此方法：
        1. 追踪实际的股票持仓数量（股数）和现金余额
        2. 考虑最小交易单位（每手 lot_size 股）
        3. 交易可能因现金不足而无法完全执行

        Parameters:
        -----------
        weights_data : pd.DataFrame
            目标权重数据，包含 date_col, asset_col, weight_col
        price_data : pd.DataFrame
            日线价格数据，包含 date_col, asset_col, buy_price, sell_price, close_price_col
        initial_capital : float
            初始资金（现金），如 1000000 表示100万
        buy_price : str
            买入价格字段名
        sell_price : str
            卖出价格字段名
        close_price_col : str
            收盘价字段名
        date_col : str
            日期列名
        asset_col : str
            资产代码列名
        weight_col : str
            权重列名
        lot_size : int
            每手股数，默认100（A股）
        trade_critic : str
            交易优先级策略：
            - 'weight_desc': 按目标权重从大到小交易（优先保证大权重标的）
            - 'weight_asc': 按目标权重从小到大交易
            - 'amount_max': 动态选择能够使用最多现金的标的优先交易（最大化资金利用率）
        transaction_cost : list of float
            交易成本 [买入费率, 卖出费率]
        slippage : float
            滑点率
        benchmark_weights : pd.DataFrame, optional
            基准权重数据
        benchmark_name : str
            基准名称

        Returns:
        --------
        dict
            回测结果，包含：
            - nav_series: 每日净值
            - cash_series: 每日现金余额
            - positions_df: 每日持仓（股数）
            - trade_records: 交易记录
            - metrics: 性能指标
        """
        print("=" * 60)
        print("Start Cash-Based Backtesting...")
        print("=" * 60)

        self.benchmark_name = benchmark_name

        # 数据验证和预处理
        validate_data(weights_data, [date_col, asset_col, weight_col], "weights_data")
        validate_data(price_data, [date_col, asset_col, buy_price, sell_price, close_price_col], "price_data")

        weights_data = weights_data.copy()
        price_data = price_data.copy()
        weights_data[date_col] = pd.to_datetime(weights_data[date_col])
        price_data[date_col] = pd.to_datetime(price_data[date_col])

        # 筛选回测时间范围
        weights_data = weights_data[
            (weights_data[date_col] >= self.start_date) &
            (weights_data[date_col] <= self.end_date)
        ]
        price_data = price_data[
            (price_data[date_col] >= self.start_date) &
            (price_data[date_col] <= self.end_date)
        ]

        # 权重归一化到1
        weights_sum = weights_data.groupby(date_col)[weight_col].transform('sum')
        weights_data[weight_col] = np.where(weights_sum == 0, 0.0, weights_data[weight_col] / weights_sum)

        # 获取所有交易日和调仓日
        all_dates = sorted(price_data[date_col].unique())
        rebalance_dates = sorted(weights_data[date_col].unique())

        print(f"  - Total trading days: {len(all_dates)}")
        print(f"  - Total rebalance days: {len(rebalance_dates)}")

        # 创建价格透视表
        p_close = price_data.pivot(index=date_col, columns=asset_col, values=close_price_col)
        p_buy = price_data.pivot(index=date_col, columns=asset_col, values=buy_price)
        p_sell = price_data.pivot(index=date_col, columns=asset_col, values=sell_price)

        # 初始化状态
        cash = initial_capital
        holdings = {}  # {asset: shares} 持仓股数

        # 记录
        nav_dict = {}
        cash_dict = {}
        trade_records = []
        positions_records = []
        turnover_records = []  # 换手率记录

        for i, date in enumerate(all_dates):
            daily_close = p_close.loc[date]
            is_rebalance = date in rebalance_dates

            if is_rebalance:
                # 获取目标权重
                target_weights = weights_data[weights_data[date_col] == date].set_index(asset_col)[weight_col].to_dict()

                # 计算调仓前总资产
                portfolio_value = cash
                for asset, shares in holdings.items():
                    if asset in daily_close.index and not pd.isna(daily_close[asset]):
                        portfolio_value += shares * daily_close[asset]

                # 计算目标持仓（股数）
                target_holdings = {}
                for asset, weight in target_weights.items():
                    if asset in p_buy.columns and date in p_buy.index:
                        price = p_buy.loc[date, asset]
                        if not pd.isna(price) and price > 0:
                            target_value = portfolio_value * weight
                            target_shares = int(target_value / price / lot_size) * lot_size
                            if target_shares > 0:
                                target_holdings[asset] = target_shares

                # 第一步：卖出（先卖出不在目标持仓中的，或需要减仓的）
                sell_list = []
                for asset, shares in holdings.items():
                    target = target_holdings.get(asset, 0)
                    if shares > target:
                        sell_list.append((asset, shares - target))

                for asset, sell_shares in sell_list:
                    if asset in p_sell.columns and date in p_sell.index:
                        price = p_sell.loc[date, asset]
                        if not pd.isna(price) and price > 0:
                            exec_price = price * (1 - slippage)
                            proceeds = sell_shares * exec_price
                            cost = proceeds * transaction_cost[1]
                            cash += proceeds - cost

                            trade_records.append({
                                'date': date,
                                'asset': asset,
                                'action': 'sell',
                                'shares': sell_shares,
                                'price': exec_price,
                                'amount': proceeds,
                                'commission': cost
                            })

                            holdings[asset] = holdings.get(asset, 0) - sell_shares
                            if holdings[asset] <= 0:
                                del holdings[asset]

                # 第二步：按顺序买入
                # 构建待买入列表
                buy_list = []
                for asset in target_holdings.keys():
                    target = target_holdings[asset]
                    current = holdings.get(asset, 0)
                    buy_shares_needed = target - current
                    if buy_shares_needed > 0 and asset in p_buy.columns and date in p_buy.index:
                        price = p_buy.loc[date, asset]
                        if not pd.isna(price) and price > 0:
                            buy_list.append({
                                'asset': asset,
                                'target': target,
                                'buy_shares_needed': buy_shares_needed,
                                'price': price,
                                'weight': target_weights.get(asset, 0)
                            })

                # 根据 trade_critic 策略确定交易顺序
                if trade_critic == 'weight_desc':
                    # 按权重从大到小排序
                    buy_list.sort(key=lambda x: x['weight'], reverse=True)
                    for item in buy_list:
                        asset = item['asset']
                        buy_shares = item['buy_shares_needed']
                        price = item['price']
                        exec_price = price * (1 + slippage)

                        required_cash = buy_shares * exec_price * (1 + transaction_cost[0])
                        if required_cash > cash:
                            affordable_shares = int(cash / (exec_price * (1 + transaction_cost[0])) / lot_size) * lot_size
                            buy_shares = affordable_shares

                        if buy_shares > 0:
                            amount = buy_shares * exec_price
                            cost = amount * transaction_cost[0]
                            cash -= (amount + cost)
                            trade_records.append({
                                'date': date, 'asset': asset, 'action': 'buy',
                                'shares': buy_shares, 'price': exec_price,
                                'amount': amount, 'commission': cost
                            })
                            holdings[asset] = holdings.get(asset, 0) + buy_shares

                elif trade_critic == 'weight_asc':
                    # 按权重从小到大排序
                    buy_list.sort(key=lambda x: x['weight'])
                    for item in buy_list:
                        asset = item['asset']
                        buy_shares = item['buy_shares_needed']
                        price = item['price']
                        exec_price = price * (1 + slippage)

                        required_cash = buy_shares * exec_price * (1 + transaction_cost[0])
                        if required_cash > cash:
                            affordable_shares = int(cash / (exec_price * (1 + transaction_cost[0])) / lot_size) * lot_size
                            buy_shares = affordable_shares

                        if buy_shares > 0:
                            amount = buy_shares * exec_price
                            cost = amount * transaction_cost[0]
                            cash -= (amount + cost)
                            trade_records.append({
                                'date': date, 'asset': asset, 'action': 'buy',
                                'shares': buy_shares, 'price': exec_price,
                                'amount': amount, 'commission': cost
                            })
                            holdings[asset] = holdings.get(asset, 0) + buy_shares

                elif trade_critic == 'amount_max':
                    # 最大化总成交金额策略
                    # 通过尝试多种排列组合，找到使总成交金额最大的交易顺序
                    from itertools import permutations

                    def calculate_total_traded(order, available_cash):
                        """模拟按给定顺序交易，返回总成交金额"""
                        temp_cash = available_cash
                        total_amount = 0
                        for item in order:
                            price = item['price']
                            exec_price = price * (1 + slippage)
                            buy_shares = item['buy_shares_needed']

                            required = buy_shares * exec_price * (1 + transaction_cost[0])
                            if required > temp_cash:
                                buy_shares = int(temp_cash / (exec_price * (1 + transaction_cost[0])) / lot_size) * lot_size

                            if buy_shares > 0:
                                amount = buy_shares * exec_price
                                cost = amount * transaction_cost[0]
                                temp_cash -= (amount + cost)
                                total_amount += amount
                        return total_amount

                    # 根据股票数量选择优化策略
                    if len(buy_list) <= 8:
                        # 股票数量较少时，尝试所有排列
                        best_order = buy_list
                        best_total = calculate_total_traded(buy_list, cash)
                        for perm in permutations(buy_list):
                            total = calculate_total_traded(perm, cash)
                            if total > best_total:
                                best_total = total
                                best_order = list(perm)
                    else:
                        # 股票数量较多时，使用多种启发式排序，取最优
                        candidates = [
                            sorted(buy_list, key=lambda x: x['buy_shares_needed'] * x['price'], reverse=True),  # 按金额降序
                            sorted(buy_list, key=lambda x: x['buy_shares_needed'] * x['price']),  # 按金额升序
                            sorted(buy_list, key=lambda x: x['price'], reverse=True),  # 按价格降序
                            sorted(buy_list, key=lambda x: x['price']),  # 按价格升序
                        ]
                        best_order = candidates[0]
                        best_total = 0
                        for cand in candidates:
                            total = calculate_total_traded(cand, cash)
                            if total > best_total:
                                best_total = total
                                best_order = cand

                    # 按最优顺序执行交易
                    for item in best_order:
                        asset = item['asset']
                        buy_shares = item['buy_shares_needed']
                        price = item['price']
                        exec_price = price * (1 + slippage)

                        required_cash = buy_shares * exec_price * (1 + transaction_cost[0])
                        if required_cash > cash:
                            buy_shares = int(cash / (exec_price * (1 + transaction_cost[0])) / lot_size) * lot_size

                        if buy_shares > 0:
                            amount = buy_shares * exec_price
                            cost = amount * transaction_cost[0]
                            cash -= (amount + cost)
                            trade_records.append({
                                'date': date, 'asset': asset, 'action': 'buy',
                                'shares': buy_shares, 'price': exec_price,
                                'amount': amount, 'commission': cost
                            })
                            holdings[asset] = holdings.get(asset, 0) + buy_shares

                # 计算当日换手率（基于交易金额）
                daily_trades = [t for t in trade_records if t['date'] == date]
                total_traded_amount = sum(t['amount'] for t in daily_trades)
                if portfolio_value > 0:
                    turnover = (total_traded_amount / 2) / portfolio_value  # 单边换手率
                    turnover_records.append({'date': date, 'turnover': turnover})

            # 计算当日净值（现金 + 持仓市值）
            portfolio_value = cash
            for asset, shares in holdings.items():
                if asset in daily_close.index and not pd.isna(daily_close[asset]):
                    portfolio_value += shares * daily_close[asset]

            nav_dict[date] = portfolio_value
            cash_dict[date] = cash

            # 记录持仓
            for asset, shares in holdings.items():
                if shares > 0:
                    positions_records.append({
                        'date': date,
                        'asset': asset,
                        'shares': shares,
                        'market_value': shares * (daily_close[asset] if asset in daily_close.index and not pd.isna(daily_close[asset]) else 0)
                    })

        # 转换结果
        self.daily_nav = pd.Series(nav_dict, name='nav').sort_index()
        cash_series = pd.Series(cash_dict, name='cash').sort_index()
        self.cash_series = cash_series  # 保存现金序列供绘图使用
        self.daily_positions = pd.DataFrame(positions_records)
        self.trade_records = pd.DataFrame(trade_records)
        self.turnover_records = pd.DataFrame(turnover_records)  # 换手率记录

        print(f"  - Final NAV: {self.daily_nav.iloc[-1]:,.2f}")
        print(f"  - Final Cash: {cash_series.iloc[-1]:,.2f}")
        print(f"  - Total trades: {len(self.trade_records)}")

        # 计算基准（如果提供）
        benchmark_nav = None
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights[date_col] = pd.to_datetime(benchmark_weights[date_col])
            benchmark_weights = benchmark_weights[
                (benchmark_weights[date_col] >= self.start_date) &
                (benchmark_weights[date_col] <= self.end_date)
            ]
            # 基准使用简单的权重回测（不考虑现金约束）
            bench_sum = benchmark_weights.groupby(date_col)[weight_col].transform('sum')
            benchmark_weights[weight_col] = np.where(bench_sum == 0, 0.0, benchmark_weights[weight_col] / bench_sum)

            # 简化的基准计算
            bench_nav = initial_capital
            bench_nav_dict = {}
            bench_positions = {}
            bench_rebalance_dates = sorted(benchmark_weights[date_col].unique())

            for i, date in enumerate(all_dates):
                daily_close = p_close.loc[date]
                is_rebalance = date in bench_rebalance_dates

                if is_rebalance:
                    target = benchmark_weights[benchmark_weights[date_col] == date].set_index(asset_col)[weight_col]
                    bench_positions = target.to_dict()

                if i > 0 and bench_positions:
                    prev_date = all_dates[i - 1]
                    prev_close = p_close.loc[prev_date]
                    port_ret = 0
                    for asset, w in bench_positions.items():
                        if asset in daily_close.index and asset in prev_close.index:
                            if not pd.isna(daily_close[asset]) and not pd.isna(prev_close[asset]) and prev_close[asset] > 0:
                                ret = (daily_close[asset] - prev_close[asset]) / prev_close[asset]
                                port_ret += w * ret
                    bench_nav *= (1 + port_ret)

                bench_nav_dict[date] = bench_nav

            benchmark_nav = pd.Series(bench_nav_dict, name='benchmark').sort_index()

        # 计算评价指标
        self.metrics = calculate_all_metrics(
            nav_series=self.daily_nav,
            benchmark_nav=benchmark_nav,
            trade_dates=rebalance_dates
        )

        # 添加现金相关指标
        self.metrics['最终现金余额'] = cash_series.iloc[-1]
        self.metrics['最终现金占比'] = cash_series.iloc[-1] / self.daily_nav.iloc[-1]
        self.metrics['平均现金占比'] = (cash_series / self.daily_nav).mean()

        # 整理回测结果
        self.backtest_results = {
            'nav_series': self.daily_nav,
            'cash_series': cash_series,
            'positions_df': self.daily_positions,
            'trade_records': self.trade_records,
            'metrics': self.metrics,
            'benchmark_nav': benchmark_nav
        }

        print("\n" + "=" * 60)
        print("Cash-Based Backtest Complete")
        print("=" * 60)

        return self.backtest_results


    # ==================== 可视化方法 ====================
    
    def _set_plotting_style(self):
        """
        设置专业的绘图风格
        """
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'bmh')
        
        # 自定义 rcParams 以获得更好的视觉效果
        plt.rcParams.update({
            'font.family': ['sans-serif'],
            'font.sans-serif': ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'], # 优先使用英文字体，中文后备
            'axes.unicode_minus': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 150, # 提高清晰度
            'lines.linewidth': 1.5
        })
        
    def plot_nav_curve(self, figsize: Tuple[int, int] = (14, 8),
                       title: str = "Strategy Performance Analysis",
                       log_scale: bool = False,
                       save_path: str = None) -> None:
        """
        绘制累计净值曲线 (包含回撤子图)

        Parameters:
        -----------
        figsize : Tuple[int, int]
            图表大小
        title : str
            图表标题
        log_scale : bool
            是否使用对数坐标轴，默认 False。
            当净值变化较大时（如多年回测），使用对数坐标可以更好地展示相对变化。
        save_path : str
            保存路径，如果为 None 则不保存
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")

        self._set_plotting_style()

        # 准备数据
        nav = self.daily_nav
        benchmark = self.backtest_results.get('benchmark_nav')

        # 计算回撤
        dd_info = calculate_max_drawdown(nav)
        drawdown = dd_info['drawdown_series']

        # 创建画布
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

        # --- 子图1: 净值曲线 ---
        ax1 = fig.add_subplot(gs[0:2])
        ax1.plot(nav.index, nav.values, label='Strategy', color='#1f77b4', linewidth=2)

        if benchmark is not None:
            ax1.plot(benchmark.index, benchmark.values, label=self.benchmark_name, color='#7f7f7f', linewidth=1.5, alpha=0.8, linestyle='--')

        # 设置对数坐标轴
        if log_scale:
            ax1.set_yscale('log')
            title_suffix = " (Log Scale)"
        else:
            title_suffix = ""

        # 标记最大回撤区间
        if dd_info['drawdown_start'] is not None and dd_info['drawdown_end'] is not None:
            ax1.axvspan(dd_info['drawdown_start'], dd_info['drawdown_end'],
                       color='red', alpha=0.1, label='Max Drawdown Period')

        ax1.set_ylabel('Net Asset Value' + (' (Log)' if log_scale else ''))
        ax1.set_title(title + title_suffix)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        # 添加关键指标文本框
        metrics_text = (
            f"Total Return: {self.metrics.get('累计收益率', 0):.2%}\n"
            f"Annual Return: {self.metrics.get('年化收益率', 0):.2%}\n"
            f"Sharpe Ratio: {self.metrics.get('夏普比率', 0):.2f}\n"
            f"Max Drawdown: {self.metrics.get('最大回撤', 0):.2%}"
        )
        # 放在图表左上角内部
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgrey')
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

        # --- 子图2: 回撤曲线 ---
        ax2 = fig.add_subplot(gs[2], sharex=ax1)
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='#d62728', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='#d62728', linewidth=1, label='Drawdown')
        
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend(loc='lower left')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # 格式化Y轴为百分比
        import matplotlib.ticker as mtick
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        _display_plot(save_path)

    def plot_nav_curve_dual(self, figsize: Tuple[int, int] = (14, 12),
                            title: str = "Strategy Performance Analysis",
                            save_path: str = None) -> None:
        """
        同时绘制普通净值曲线和 Log Scale 净值曲线

        上半部分为普通坐标净值曲线，下半部分为对数坐标净值曲线，
        方便对比观察不同尺度下的收益走势。
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")

        self._set_plotting_style()

        nav = self.daily_nav
        benchmark = self.backtest_results.get('benchmark_nav')
        dd_info = calculate_max_drawdown(nav)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.35)

        # --- 子图1: 普通净值曲线 ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(nav.index, nav.values, label='Strategy', color='#1f77b4', linewidth=2)
        if benchmark is not None:
            ax1.plot(benchmark.index, benchmark.values, label=self.benchmark_name, color='#7f7f7f', linewidth=1.5, alpha=0.8, linestyle='--')
        if dd_info['drawdown_start'] is not None and dd_info['drawdown_end'] is not None:
            ax1.axvspan(dd_info['drawdown_start'], dd_info['drawdown_end'], color='red', alpha=0.1, label='Max Drawdown Period')
        ax1.set_ylabel('NAV')
        ax1.set_title(title)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        # --- 子图2: Log Scale 净值曲线 ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(nav.index, nav.values, label='Strategy (Log)', color='#1f77b4', linewidth=2)
        if benchmark is not None:
            ax2.plot(benchmark.index, benchmark.values, label=self.benchmark_name, color='#7f7f7f', linewidth=1.5, alpha=0.8, linestyle='--')
        ax2.set_yscale('log')
        ax2.set_ylabel('NAV (Log Scale)')
        ax2.set_title(title + ' - Log Scale')
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)

        # --- 子图3: 回撤曲线 ---
        drawdown = dd_info['drawdown_series']
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.fill_between(drawdown.index, drawdown.values, 0, color='#d62728', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, color='#d62728', linewidth=1, label='Drawdown')
        ax3.set_ylabel('Drawdown')
        ax3.set_xlabel('Date')
        ax3.legend(loc='lower left')
        ax3.grid(True, which='both', linestyle='--', alpha=0.5)
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.tight_layout()
        _display_plot(save_path)

    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (10, 6), save_path: str = None) -> None:
        """
        绘制月度收益率热力图
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
            
        self._set_plotting_style()
        
        # 计算月度收益
        monthly_nav = self.daily_nav.resample('ME').last()
        monthly_rets = monthly_nav.pct_change()
        
        # 构建透视表：Year x Month
        monthly_rets_df = pd.DataFrame(monthly_rets)
        monthly_rets_df['Year'] = monthly_rets_df.index.year
        monthly_rets_df['Month'] = monthly_rets_df.index.month
        
        pivot_table = monthly_rets_df.pivot(index='Year', columns='Month', values='nav')
        
        # 补全月份（如果某些年份没有12个月）
        all_months = range(1, 13)
        for m in all_months:
            if m not in pivot_table.columns:
                pivot_table[m] = np.nan
        pivot_table = pivot_table[sorted(pivot_table.columns)]
        
        # 计算年度总收益 (作为最后一列)
        yearly_rets = self.daily_nav.resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
        yearly_rets.index = yearly_rets.index.year
        pivot_table['Yearly'] = yearly_rets
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用 imshow 绘制热力图
        # 注意：处理NaN值，避免绘图报错
        data_values = pivot_table.values
        # 创建掩码
        mask = np.isnan(data_values)
        
        # 绘制热力图
        im = ax.imshow(data_values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Return', rotation=-90, va="bottom")
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Yearly']
        ax.set_xticklabels(month_labels)
        ax.set_yticklabels(pivot_table.index)
        
        # 在每个格子里添加数值文本
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = data_values[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.05 else "black"
                    text = ax.text(j, i, f"{val:.1%}",
                                   ha="center", va="center", color=text_color, fontsize=9)
                    
        ax.set_title("Monthly Returns Heatmap")
        fig.tight_layout()
        
        _display_plot(save_path)

    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        绘制回撤曲线 (独立)
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
            
        self._set_plotting_style()
        
        dd_info = calculate_max_drawdown(self.daily_nav)
        drawdown_series = dd_info['drawdown_series']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(drawdown_series.index, drawdown_series.values, 0,
                        color='#d62728', alpha=0.3)
        ax.plot(drawdown_series.index, drawdown_series.values,
               linewidth=2, color='#d62728', label='Drawdown')
        
        # 标记最大回撤
        max_dd_end = dd_info['drawdown_end']
        max_dd_value = dd_info['max_drawdown']
        
        if max_dd_end in drawdown_series.index:
            ax.scatter([max_dd_end], [drawdown_series[max_dd_end]], 
                      color='black', s=50, zorder=5, label=f'Max DD: {max_dd_value:.2%}')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Strategy Drawdown')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        plt.show()
    
    def plot_nav_vs_benchmark(self, figsize: Tuple[int, int] = (12, 6), save_path: str = None) -> None:
        """
        Strategy vs Benchmark Comparison (Independent)
        """
        if self.daily_nav is None:
            raise ValueError("Please run run_backtest() first")
        
        self._set_plotting_style()
        
        if self.backtest_results is None or self.backtest_results.get('benchmark_nav') is None:
            print("No benchmark data provided, cannot plot comparison")
            return
            
        benchmark_nav = self.backtest_results['benchmark_nav']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        
        # Top: NAV Comparison
        # Normalize to start at 1.0 for better comparison
        strategy_norm = self.daily_nav / self.daily_nav.iloc[0]
        benchmark_norm = benchmark_nav / benchmark_nav.iloc[0]
        
        ax1.plot(strategy_norm.index, strategy_norm.values, label='Strategy', color='#1f77b4', linewidth=2)
        ax1.plot(benchmark_norm.index, benchmark_norm.values, label=self.benchmark_name, color='#7f7f7f', linestyle='--', linewidth=1.5)
        ax1.set_ylabel('Normalized NAV')
        ax1.set_title('Strategy vs Benchmark')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Relative Strength (Strategy / Benchmark)
        # Or Excess Return (Strategy - Benchmark) - Let's use Cumulative Excess Return as it's more standard
        s_ret = self.daily_nav.pct_change().fillna(0)
        b_ret = benchmark_nav.pct_change().fillna(0)
        excess_ret = s_ret - b_ret
        cum_excess = (1 + excess_ret).cumprod() - 1
        
        ax2.plot(cum_excess.index, cum_excess.values, color='#2ca02c', label='Cumulative Excess Return', linewidth=1.5)
        ax2.axhline(0.0, color='black', linestyle='-', linewidth=0.5)
        ax2.fill_between(cum_excess.index, cum_excess.values, 0, where=(cum_excess.values >= 0), color='#2ca02c', alpha=0.3)
        ax2.fill_between(cum_excess.index, cum_excess.values, 0, where=(cum_excess.values < 0), color='#d62728', alpha=0.3)
        
        ax2.set_ylabel('Excess Return')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        import matplotlib.ticker as mtick
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        _display_plot(save_path)

    def plot_metrics_table(self, save_path: str = None) -> None:
        """
        Render performance metrics as a table image (English)
        """
        if self.metrics is None:
            raise ValueError("Please run run_backtest() first")
            
        # Mapping keys to English if they are in Chinese
        key_map = {
            '累计收益率': 'Total Return',
            '年化收益率': 'Annual Return',
            '年化波动率': 'Annual Volatility',
            '夏普比率': 'Sharpe Ratio',
            '索提诺比率': 'Sortino Ratio',
            '卡玛比率': 'Calmar Ratio',
            '胜率': 'Win Rate',
            '最大回撤': 'Max Drawdown',
            '最大回撤开始日期': 'Max DD Start',
            '最大回撤结束日期': 'Max DD End',
            '最大回撤持续天数': 'Max DD Duration (Days)',
            '交易次数': 'Total Trades',
            '平均换手率': 'Avg Turnover',
            '累计换手率': 'Total Turnover',
            '基准累计收益率': 'Benchmark Total Return',
            '基准年化收益率': 'Benchmark Annual Return',
            '超额收益': 'Excess Return',
            '年化超额收益': 'Annual Excess Return',
            '信息比率': 'Information Ratio',
            '跟踪误差': 'Tracking Error',
            'VaR (95%)': 'VaR (95%)',
            'CVaR (95%)': 'CVaR (95%)'
        }
        
        # Filter and format metrics
        display_metrics = []
        for k, v in self.metrics.items():
            eng_key = key_map.get(k, k)
            
            # Format values
            if isinstance(v, (float, np.float64)):
                if 'Ratio' in eng_key or 'Days' in eng_key or 'Trades' in eng_key:
                    val_str = f"{v:.2f}"
                else:
                    val_str = f"{v:.2%}"
            elif isinstance(v, pd.Timestamp):
                val_str = v.strftime('%Y-%m-%d')
            else:
                val_str = str(v)
                
            display_metrics.append((eng_key, val_str))
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, len(display_metrics) * 0.5 + 1))
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=display_metrics,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.5, 0.4]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Header style
        for i in range(2):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
            
        # Alternating row colors
        for i in range(1, len(display_metrics) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f2f2f2')
                    
        plt.title('Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Metrics table saved to {save_path}")
        plt.show()

    
    def plot_excess_returns(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Cumulative Excess Return Curve
        """
        if self.daily_nav is None:
            raise ValueError("Please run run_backtest() first")
        
        if self.backtest_results is None or self.backtest_results.get('benchmark_nav') is None:
            print("No benchmark data provided, cannot plot excess returns")
            return
        
        benchmark_nav = self.backtest_results['benchmark_nav']
        
        # Calculate excess returns
        strategy_returns = calculate_returns(self.daily_nav)
        benchmark_returns = calculate_returns(benchmark_nav)
        
        # Align
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        excess_returns = aligned_strategy - aligned_benchmark
        
        # Cumulative excess return
        cumulative_excess = (1 + excess_returns).cumprod() - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(cumulative_excess.index, cumulative_excess.values,
               linewidth=2, color='#06A77D', label='Cumulative Excess Return')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(cumulative_excess.index, cumulative_excess.values, 0,
                        where=(cumulative_excess.values > 0), alpha=0.3, color='green')
        ax.fill_between(cumulative_excess.index, cumulative_excess.values, 0,
                        where=(cumulative_excess.values < 0), alpha=0.3, color='red')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Excess Return', fontsize=12)
        ax.set_title('Cumulative Excess Return', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_points(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Trade Points Analysis
        """
        if self.trade_records is None or len(self.trade_records) == 0:
            print("No trade records")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Top: NAV + Trade Points
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.daily_nav.index, self.daily_nav.values,
                linewidth=2, color='#2E86AB', label='NAV')
        
        trade_dates = self.trade_records['date']
        trade_navs = [self.daily_nav[d] for d in trade_dates if d in self.daily_nav.index]
        ax1.scatter(trade_dates, trade_navs, color='red', s=10,
                   zorder=5, label='Rebalance Point', alpha=0.6)
        
        ax1.set_ylabel('NAV', fontsize=12)
        ax1.set_title('Trade Points Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Transaction Cost
        ax2 = fig.add_subplot(gs[1])
        if 'commission' in self.trade_records.columns:
            ax2.bar(self.trade_records['date'], self.trade_records['commission'],
                   color='#F18F01', alpha=0.7, label='Transaction Cost')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Cost', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))

        
        plt.tight_layout()
        plt.show()
    
    def plot_position_heatmap(self, figsize: Tuple[int, int] = (14, 8), save_path: str = None) -> None:
        """
        Position Heatmap
        Prioritizes assets with high weight and long holding duration.
        """
        if self.daily_positions is None or len(self.daily_positions) == 0:
            print("No position data")
            return
        
        self._set_plotting_style()
        
        # Pivot: Date x Asset
        positions_pivot = self.daily_positions.pivot_table(
            index='date', columns='asset', values='weight', fill_value=0
        )
        
        # Sort assets by "Importance" = Sum of daily weights (captures both weight and duration)
        # If an asset is held for 100 days at 10%, sum is 10.
        # If an asset is held for 10 days at 10%, sum is 1.
        asset_importance = positions_pivot.sum().sort_values(ascending=False)
        
        # If too many assets, show top 20
        if positions_pivot.shape[1] > 20:
            top_assets = asset_importance.head(20).index
            positions_pivot = positions_pivot[top_assets]
        else:
            # Still sort by importance for better visualization
            positions_pivot = positions_pivot[asset_importance.index]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(positions_pivot.T.values, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest', vmin=0, vmax=positions_pivot.max().max())
        
        # Ticks
        ax.set_yticks(range(len(positions_pivot.columns)))
        ax.set_yticklabels(positions_pivot.columns)
        
        # Date ticks
        date_indices = list(range(0, len(positions_pivot), max(1, len(positions_pivot) // 10)))
        ax.set_xticks(date_indices)
        ax.set_xticklabels([positions_pivot.index[i].strftime('%Y-%m-%d') 
                           for i in date_indices], rotation=45, ha='right')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.set_title('Position Weights Heatmap (Top Assets)', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight', fontsize=12)
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        _display_plot(save_path)
    
    def plot_turnover_analysis(self, figsize: Tuple[int, int] = (12, 6), save_path: str = None) -> None:
        """
        Turnover Analysis
        """
        if self.turnover_records is None or len(self.turnover_records) == 0:
            print("No turnover data")
            return
        
        self._set_plotting_style()

        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(self.turnover_records['date'], self.turnover_records['turnover'],
              color='#C73E1D', alpha=0.7, label='Turnover Rate')
        
        # Add average lines
        avg_turnover = self.turnover_records['turnover'].mean()
        # avg_turnover_trade = self.turnover_records['turnover'].replace(0, np.nan).mean()
        ax.axhline(y=avg_turnover, color='blue', linestyle='--',
                  linewidth=2, label=f'Avg Turnover (Period): {avg_turnover:.2%}')
        # ax.axhline(y=avg_turnover_trade, color='green', linestyle='--',
        #           linewidth=2, label=f'Avg Turnover (Trade): {avg_turnover_trade:.2%}')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Turnover Rate', fontsize=12)
        ax.set_title('Turnover Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        _display_plot(save_path)
    
    def plot_dashboard(self, save_path: str = None) -> None:
        """
        Comprehensive Dashboard
        Includes: NAV, Excess Return (if benchmark), Drawdown, Key Metrics, Turnover

        NAV display:
        - run_backtest: Normalized NAV starting from 1
        - run_backtest_with_cash: Actual NAV starting from initial_capital
        """
        if self.daily_nav is None:
            raise ValueError("Please run run_backtest() first")

        self._set_plotting_style()

        benchmark_nav = self.backtest_results.get('benchmark_nav')
        has_benchmark = benchmark_nav is not None
        cash_series = getattr(self, 'cash_series', None)
        has_cash = cash_series is not None  # 判断是否为现金仓位回测

        # Calculate Drawdown Info early for use in NAV plot
        dd_info = calculate_max_drawdown(self.daily_nav)

        # Layout configuration
        # If benchmark exists: 4 rows (NAV, Excess, Drawdown, Metrics/Turnover)
        # If no benchmark: 3 rows (NAV, Drawdown, Metrics/Turnover)

        if has_benchmark:
            fig = plt.figure(figsize=(18, 14))
            gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1.5], hspace=0.4)
        else:
            fig = plt.figure(figsize=(18, 10))
            gs = GridSpec(3, 2, height_ratios=[2, 1, 1.5], hspace=0.4)

        # 1. NAV Curve (Top Row)
        ax_nav = fig.add_subplot(gs[0, :])

        if has_cash:
            # 现金仓位回测：显示实际净值（从 initial_capital 开始）
            ax_nav.plot(self.daily_nav.index, self.daily_nav.values, label='Strategy NAV', color='#1f77b4', linewidth=2)
            if has_benchmark:
                ax_nav.plot(benchmark_nav.index, benchmark_nav.values, label=self.benchmark_name, color='#7f7f7f', linestyle='--', alpha=0.7)
            ax_nav.set_title('Portfolio Net Asset Value', fontsize=14)
            ax_nav.set_ylabel('NAV')
            # 格式化 Y 轴为带千位分隔符的数字
            ax_nav.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            # 标准回测：显示归一化净值（从 1 开始）
            nav_norm = self.daily_nav / self.daily_nav.iloc[0]
            ax_nav.plot(nav_norm.index, nav_norm.values, label='Strategy', color='#1f77b4', linewidth=2)
            if has_benchmark:
                bench_norm = benchmark_nav / benchmark_nav.iloc[0]
                ax_nav.plot(bench_norm.index, bench_norm.values, label=self.benchmark_name, color='#7f7f7f', linestyle='--', alpha=0.7)
            ax_nav.set_title('Cumulative Returns', fontsize=14)
            ax_nav.set_ylabel('Normalized NAV')
            
        # Highlight Max Drawdown Period
        if dd_info['drawdown_start'] is not None and dd_info['drawdown_end'] is not None:
            ax_nav.axvspan(dd_info['drawdown_start'], dd_info['drawdown_end'],
                           color='red', alpha=0.1, label='Max Drawdown Period')

        ax_nav.legend(loc='upper left')
        ax_nav.grid(True, alpha=0.3)

        current_row = 1

        # 2. Excess Return (If benchmark)
        if has_benchmark:
            ax_excess = fig.add_subplot(gs[current_row, :], sharex=ax_nav)
            
            s_ret = self.daily_nav.pct_change().fillna(0)
            b_ret = benchmark_nav.pct_change().fillna(0)
            excess_ret = s_ret - b_ret
            cum_excess = (1 + excess_ret).cumprod() - 1
            
            ax_excess.plot(cum_excess.index, cum_excess.values, color='#2ca02c', label='Cumulative Excess Return', linewidth=1.5)
            ax_excess.axhline(0.0, color='black', linestyle='-', linewidth=0.5)
            ax_excess.fill_between(cum_excess.index, cum_excess.values, 0, where=(cum_excess.values >= 0), color='#2ca02c', alpha=0.3)
            ax_excess.fill_between(cum_excess.index, cum_excess.values, 0, where=(cum_excess.values < 0), color='#d62728', alpha=0.3)
            
            ax_excess.set_ylabel('Excess Return')
            ax_excess.legend(loc='upper left')
            ax_excess.grid(True, alpha=0.3)
            ax_excess.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            current_row += 1
            
        # 3. Drawdown
        ax_dd = fig.add_subplot(gs[current_row, :], sharex=ax_nav)
        # dd_info already calculated
        drawdown = dd_info['drawdown_series']
        
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color='#d62728', alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown.values, color='#d62728', linewidth=1)
        
        # Mark Max Drawdown
        max_dd_end = dd_info['drawdown_end']
        max_dd_val = dd_info['max_drawdown']
        if max_dd_end in drawdown.index:
             ax_dd.scatter([max_dd_end], [drawdown[max_dd_end]], color='red', s=20, zorder=5, 
                           label=f'Max DD: {max_dd_val:.2%}')
             ax_dd.legend(loc='lower left')

        ax_dd.set_ylabel('Drawdown')
        ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_dd.grid(True, alpha=0.3)
        
        current_row += 1
        
        # 4. Metrics Table (Bottom Left)
        ax_metrics = fig.add_subplot(gs[current_row, 0])
        ax_metrics.axis('off')
        
        key_metrics = [
            ['Metric', 'Value'],
            ['Total Return', f"{self.metrics.get('累计收益率', 0):.2%}"],
            ['Annual Return', f"{self.metrics.get('年化收益率', 0):.2%}"],
            ['Annual Volatility', f"{self.metrics.get('年化波动率', 0):.2%}"],
            ['Sharpe Ratio', f"{self.metrics.get('夏普比率', 0):.2f}"],
            ['Max Drawdown', f"{self.metrics.get('最大回撤', 0):.2%}"],
            ['Calmar Ratio', f"{self.metrics.get('卡玛比率', 0):.2f}"],
            ['Win Rate', f"{self.metrics.get('胜率', 0):.2%}"]
        ]
        
        if has_benchmark:
             key_metrics.append(['Excess Return', f"{self.metrics.get('超额收益', 0):.2%}"])
             key_metrics.append(['Info Ratio', f"{self.metrics.get('信息比率', 0):.2f}"])

        # 如果有现金仓位数据，添加现金相关指标
        if has_cash:
            key_metrics.append(['Final Cash Ratio', f"{self.metrics.get('最终现金占比', 0):.2%}"])
            key_metrics.append(['Avg Cash Ratio', f"{self.metrics.get('平均现金占比', 0):.2%}"])

        table = ax_metrics.table(cellText=key_metrics, loc='center', cellLoc='left', colWidths=[0.5, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        
        # Header style
        for i in range(2):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
            
        # 5. Turnover (Bottom Right)
        ax_turnover = fig.add_subplot(gs[current_row, 1])
        if self.turnover_records is not None and not self.turnover_records.empty:
            ax_turnover.bar(self.turnover_records['date'], self.turnover_records['turnover'], 
                           color='#ff7f0e', alpha=0.6, width=2)
            ax_turnover.set_title('Turnover')
            ax_turnover.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax_turnover.grid(True, alpha=0.3)
        else:
            ax_turnover.text(0.5, 0.5, 'No Turnover Data', ha='center')
            ax_turnover.axis('off')

        plt.suptitle('Quantitative Strategy Backtest Report', fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        plt.show()

    def plot_all(self, save_path: str = None) -> None:
        """
        Comprehensive Dashboard (Alias for plot_dashboard)
        """
        self.plot_dashboard(save_path=save_path)

