"""
通用量化策略回测框架
支持灵活的调仓时间、向量化计算、丰富的性能指标和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, List, Tuple
import warnings


from .utils import (
    validate_data, align_dates, calculate_all_metrics,
    calculate_returns, calculate_max_drawdown, calculate_turnover,
    calculate_transaction_costs, calculate_monthly_returns, calculate_adjusted_weights
)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


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
        rebalance_threshold: float = 0.005,
        transaction_cost: List[float] = [0.001, 0.001],
        initial_capital: float = 1.0,
        benchmark_weights: Optional[pd.DataFrame] = None
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
        rebalance_threshold : float
            调仓阈值，如果某只标的的仓位变化绝对值不超过rebalance_threshold，则不进行调仓
        transaction_cost : list of float
            交易成本，格式为 [买入成本, 卖出成本]
        initial_capital : float
            初始资金
        benchmark_weights : pd.DataFrame, optional
            基准权重数据，格式与weights_data相同
            
        Returns:
        --------
        dict
            回测结果字典
        """
        print("=" * 60)
        print("Start Backtesting...")
        print("=" * 60)
        
        # 1. 数据预处理和验证
        ## 验证字段是否齐全
        weights_data, price_data = self._preprocess_data(
            weights_data, price_data, date_col, asset_col, weight_col,
            buy_price, sell_price, adj_factor_col, close_price_col
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
            rebalance_threshold, transaction_cost, initial_capital
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
                date_col, asset_col, weight_col, buy_price, sell_price,
                adj_factor_col, rebalance_threshold, [0, 0],  # 基准不考虑交易成本
                initial_capital
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
        close_price_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        数据预处理和验证
        """
        # 验证数据
        validate_data(weights_data, [date_col, asset_col, weight_col], "weights_data")
        validate_data(
            price_data, 
            [date_col, asset_col, buy_price, sell_price, adj_factor_col, close_price_col], 
            "price_data"
        )
        
        # 转换日期格式
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
        
        # 计算复权价格（用于收益率计算）
        price_data['adj_buy_price'] = price_data[buy_price] * price_data[adj_factor_col]
        price_data['adj_sell_price'] = price_data[sell_price] * price_data[adj_factor_col]
        price_data['adj_close_price'] = price_data[close_price_col] * price_data[adj_factor_col]

        
        # 权重归一化（确保每个日期的权重和为1）
        weights_sum = weights_data.groupby(date_col)[weight_col].transform('sum')
        weights_data[weight_col] = np.where(weights_sum == 0, 0.0, weights_data[weight_col] / weights_sum)
        return weights_data, price_data
    
    def _calculate_daily_nav(
        self,
        weights_data: pd.DataFrame,
        price_data: pd.DataFrame,
        rebalance_dates: List,
        date_col: str,
        asset_col: str,
        weight_col: str,
        rebalance_threshold: float,
        transaction_cost: List[float],
        initial_capital: float
    ) -> Dict:
        """
        计算每日净值（向量化）
        
        核心逻辑：
        1. 在调仓日，根据目标权重调仓，扣除交易成本
        2. 调仓日当天，还要考虑从买入价到收盘价的收益
        3. 非调仓日，根据复权价格变化计算净值
        """
        # 获取所有交易日
        all_dates = sorted(price_data[date_col].unique())
        
        # 初始化结果
        nav_dict = {}
        positions_records = []
        trade_records = []
        turnover_records = []
        
        # 当前净值和持仓
        current_nav = initial_capital
        current_positions = {}  # {asset: weight}
        
        # 遍历所有交易日
        for i, date in enumerate(all_dates):
            is_rebalance_day = date in rebalance_dates
            
            # 获取当日价格数据
            day_prices = price_data[price_data[date_col] == date].set_index(asset_col)
            
            if is_rebalance_day:
                # ===== 调仓日 =====
                
                # 获取目标权重
                target_weights = weights_data[weights_data[date_col] == date].set_index(asset_col)[weight_col]
                
                # 调仓前的持仓（Series）
                before_weights = pd.Series(current_positions)

                target_weights = calculate_adjusted_weights(weight_after=target_weights, 
                                                            weight_before=before_weights, 
                                                            rebalance_threshold=rebalance_threshold)
                
                # 计算交易成本
                cost = calculate_transaction_costs(
                    before_weights, target_weights,
                    transaction_cost[0], transaction_cost[1]
                )
                
                # 扣除交易成本
                current_nav = current_nav * (1 - cost)
                
                # 更新持仓为目标权重
                current_positions = target_weights.to_dict()
                
                # 记录交易
                trade_records.append({
                    'date': date,
                    'nav_before': current_nav / (1 - cost) if cost < 1 else current_nav,
                    'nav_after': current_nav,
                    'cost': cost,
                    'cost_amount': current_nav / (1 - cost) * cost if cost < 1 else 0
                })
                
                # 计算换手率
                turnover = abs(target_weights.reindex(before_weights.index.union(target_weights.index), fill_value=0) - 
                              before_weights.reindex(before_weights.index.union(target_weights.index), fill_value=0)).sum()
                turnover_records.append({
                    'date': date,
                    'turnover': turnover
                })
                
                # ===== 重要：计算调仓当日从买入价到收盘价的收益 =====
                # 这是用户特别提醒需要考虑的部分
                
                intraday_return = 0.0
                for asset, weight in current_positions.items():
                    if asset in day_prices.index:
                        # 买入价（复权）
                        buy_adj_price = day_prices.loc[asset, 'adj_buy_price']
                        # 收盘价（复权），用于计算当日收益
                        close_adj_price = day_prices.loc[asset, 'adj_close_price']
                        
                        # 资产当日收益率
                        if buy_adj_price > 0:
                            asset_return = (close_adj_price - buy_adj_price) / buy_adj_price
                            intraday_return += weight * asset_return
                
                # 应用当日收益到净值
                current_nav = current_nav * (1 + intraday_return)
                
            else:
                # ===== 非调仓日 =====
                
                # 计算持仓收益（基于复权价格变化）
                if i > 0 and len(current_positions) > 0:
                    prev_date = all_dates[i - 1]
                    prev_prices = price_data[price_data[date_col] == prev_date].set_index(asset_col)
                    
                    daily_return = 0.0
                    for asset, weight in current_positions.items():
                        if asset in day_prices.index and asset in prev_prices.index:
                            # 使用复权后的卖出价计算收益率
                            prev_adj_price = prev_prices.loc[asset, 'adj_close_price']
                            curr_adj_price = day_prices.loc[asset, 'adj_close_price']
                            
                            if prev_adj_price > 0:
                                asset_return = (curr_adj_price - prev_adj_price) / prev_adj_price
                                daily_return += weight * asset_return
                    
                    # 更新净值
                    current_nav = current_nav * (1 + daily_return)
            
            # 记录当日净值
            nav_dict[date] = current_nav
            
            # 记录当日持仓
            for asset, weight in current_positions.items():
                positions_records.append({
                    'date': date,
                    'asset': asset,
                    'weight': weight
                })
        
        # 转换为 DataFrame 和 Series
        nav_series = pd.Series(nav_dict)
        nav_series.index = pd.to_datetime(nav_series.index)
        nav_series = nav_series.sort_index()
        
        positions_df = pd.DataFrame(positions_records)
        if len(positions_df) > 0:
            positions_df['date'] = pd.to_datetime(positions_df['date'])
        
        trade_records_df = pd.DataFrame(trade_records)
        if len(trade_records_df) > 0:
            trade_records_df['date'] = pd.to_datetime(trade_records_df['date'])
        
        turnover_records_df = pd.DataFrame(turnover_records)
        if len(turnover_records_df) > 0:
            turnover_records_df['date'] = pd.to_datetime(turnover_records_df['date'])
        
        return {
            'nav_series': nav_series,
            'positions_df': positions_df,
            'trade_records': trade_records_df,
            'turnover_records': turnover_records_df
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
    
    def run_backtest_ETF(self, 
        weights_data: pd.DataFrame,
        buy_price: str = 'OpenPrice',
        sell_price: str = 'ClosePrice',
        transaction_cost: List[float] = [0.001, 0.001],
        rebalance_threshold = 0.01,
        benchmark_weights: pd.DataFrame = None,
        ):
        from quantchdb import ClickHouseDatabase
        from .db_config import Agent_db_config

        db = ClickHouseDatabase(config=Agent_db_config, terminal_log=False)

        sql = f"""
        WITH
            cp as (
                SELECT Symbol, argMin(ComparablePrice, TradingDate) as init_cp
                FROM etf.etf_daily
                WHERE ComparablePrice IS NOT NULL
                GROUP BY Symbol
            ),

            prices as (
                SELECT Symbol as code, TradingDate as date, OpenPrice, HighPrice, LowPrice, ClosePrice, ComparablePrice, ComparablePrice/init_cp as adj_factor
                FROM etf.etf_daily daily
                LEFT JOIN cp ON cp.Symbol == daily.Symbol
                WHERE and(TradingDate>='{self.start_date.strftime(format='%Y-%m-%d')}', 
                          TradingDate<='{self.end_date.strftime(format='%Y-%m-%d')}',
                          Filling=0)
                ORDER BY Symbol, TradingDate
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
            rebalance_threshold = rebalance_threshold,
            transaction_cost=transaction_cost,
            benchmark_weights=benchmark_weights
            )

        return results

    def run_backtest_stock(self, 
        weights_data: pd.DataFrame,
        buy_price: str = 'open',
        sell_price: str = 'close',
        transaction_cost: List[float] = [0.001, 0.001],
        rebalance_threshold = 0.01,
        benchmark_weights: pd.DataFrame = None,
        ):
        from quantchdb import ClickHouseDatabase
        from .db_config import Agent_db_config

        db = ClickHouseDatabase(config=Agent_db_config, terminal_log=False)

        sql = f"""
        WITH
            prices as (
                SELECT Stkcd as code, Trddt as date, Opnprc as open, Hiprc as high, Loprc as low, Clsprc as close,
                  Adjprcwd as adj_price, adj_factor_f as adj_factor
                FROM stocks.stocks_daily 
                WHERE and(Trddt>='{self.start_date.strftime(format='%Y-%m-%d')}', 
                          Trddt<='{self.end_date.strftime(format='%Y-%m-%d')}')
                ORDER BY Stkcd, Trddt
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
            rebalance_threshold = rebalance_threshold,
            transaction_cost=transaction_cost,
            benchmark_weights=benchmark_weights
            )

        return results
        
    
    # ==================== 可视化方法 ====================
    
    def plot_nav_curve(self, figsize: Tuple[int, int] = (12, 6), 
                       title: str = "策略累计净值曲线") -> None:
        """
        绘制累计净值曲线
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.daily_nav.index, self.daily_nav.values, 
                linewidth=2, label='策略净值', color='#2E86AB')
        
        # 标记调仓点
        if len(self.trade_records) > 0:
            trade_dates = self.trade_records['date']
            trade_navs = [self.daily_nav[d] for d in trade_dates if d in self.daily_nav.index]
            ax.scatter(trade_dates, trade_navs, color='red', s=10, 
                      zorder=5, label='调仓点', alpha=0.6)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        绘制回撤曲线
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
        
        dd_info = calculate_max_drawdown(self.daily_nav)
        drawdown_series = dd_info['drawdown_series']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(drawdown_series.index, drawdown_series.values, 0,
                        color='#A23B72', alpha=0.3)
        ax.plot(drawdown_series.index, drawdown_series.values,
               linewidth=2, color='#A23B72', label='回撤')
        
        # 标记最大回撤
        max_dd_start = dd_info['drawdown_start']
        max_dd_end = dd_info['drawdown_end']
        max_dd_value = dd_info['max_drawdown']
        
        if max_dd_end in drawdown_series.index:
            ax.scatter([max_dd_end], [drawdown_series[max_dd_end]], 
                      color='red', s=100, zorder=5, label=f'最大回撤: {max_dd_value:.2%}')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('回撤', fontsize=12)
        ax.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_nav_vs_benchmark(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        策略与基准对比
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
        
        if self.backtest_results is None or self.backtest_results.get('benchmark_nav') is None:
            print("未提供基准数据，无法绘制对比图")
            return
        
        benchmark_nav = self.backtest_results['benchmark_nav']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 归一化到相同起点
        strategy_normalized = self.daily_nav / self.daily_nav.iloc[0]
        benchmark_normalized = benchmark_nav / benchmark_nav.iloc[0]
        
        ax.plot(strategy_normalized.index, strategy_normalized.values,
               linewidth=2, label='策略', color='#2E86AB')
        ax.plot(benchmark_normalized.index, benchmark_normalized.values,
               linewidth=2, label='基准', color='#F18F01', linestyle='--')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('归一化净值', fontsize=12)
        ax.set_title('策略 vs 基准', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_excess_returns(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        超额收益曲线
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
        
        if self.backtest_results is None or self.backtest_results.get('benchmark_nav') is None:
            print("未提供基准数据，无法绘制超额收益")
            return
        
        benchmark_nav = self.backtest_results['benchmark_nav']
        
        # 计算超额收益
        strategy_returns = calculate_returns(self.daily_nav)
        benchmark_returns = calculate_returns(benchmark_nav)
        
        # 对齐
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        excess_returns = aligned_strategy - aligned_benchmark
        
        # 累计超额收益
        cumulative_excess = (1 + excess_returns).cumprod() - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(cumulative_excess.index, cumulative_excess.values,
               linewidth=2, color='#06A77D', label='累计超额收益')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(cumulative_excess.index, cumulative_excess.values, 0,
                        where=(cumulative_excess.values > 0), alpha=0.3, color='green')
        ax.fill_between(cumulative_excess.index, cumulative_excess.values, 0,
                        where=(cumulative_excess.values < 0), alpha=0.3, color='red')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('超额收益', fontsize=12)
        ax.set_title('累计超额收益曲线', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_points(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        交易点位分析
        """
        if self.trade_records is None or len(self.trade_records) == 0:
            print("没有交易记录")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # 上图：净值曲线 + 交易点
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.daily_nav.index, self.daily_nav.values,
                linewidth=2, color='#2E86AB', label='净值')
        
        trade_dates = self.trade_records['date']
        trade_navs = [self.daily_nav[d] for d in trade_dates if d in self.daily_nav.index]
        ax1.scatter(trade_dates, trade_navs, color='red', s=10,
                   zorder=5, label='调仓点', alpha=0.6)
        
        ax1.set_ylabel('净值', fontsize=12)
        ax1.set_title('交易点位分析', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 下图：每次调仓的成本
        ax2 = fig.add_subplot(gs[1])
        if 'cost' in self.trade_records.columns:
            ax2.bar(self.trade_records['date'], self.trade_records['cost'],
                   color='#F18F01', alpha=0.7, label='交易成本')
            ax2.set_xlabel('日期', fontsize=12)
            ax2.set_ylabel('交易成本', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_position_heatmap(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        持仓热力图
        """
        if self.daily_positions is None or len(self.daily_positions) == 0:
            print("没有持仓数据")
            return
        
        # 透视表：日期 x 资产
        positions_pivot = self.daily_positions.pivot_table(
            index='date', columns='asset', values='weight', fill_value=0
        )
        
        # 如果资产太多，只显示前20个
        if positions_pivot.shape[1] > 20:
            # 按平均权重排序，取前20
            avg_weights = positions_pivot.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(20).index
            positions_pivot = positions_pivot[top_assets]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(positions_pivot.T.values, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest')
        
        # 设置刻度
        ax.set_yticks(range(len(positions_pivot.columns)))
        ax.set_yticklabels(positions_pivot.columns)
        
        # 日期刻度（采样显示）
        date_indices = list(range(0, len(positions_pivot), max(1, len(positions_pivot) // 10)))
        ax.set_xticks(date_indices)
        ax.set_xticklabels([positions_pivot.index[i].strftime('%Y-%m-%d') 
                           for i in date_indices], rotation=45, ha='right')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('资产', fontsize=12)
        ax.set_title('持仓权重热力图', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('权重', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_turnover_analysis(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        换手率分析
        """
        if self.turnover_records is None or len(self.turnover_records) == 0:
            print("没有换手率数据")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(self.turnover_records['date'], self.turnover_records['turnover'],
              color='#C73E1D', alpha=0.7, label='换手率')
        
        # 添加平均线
        avg_turnover = self.turnover_records['turnover'].mean()
        avg_turnover_trade = self.turnover_records['turnover'].replace(0, np.nan).mean()
        ax.axhline(y=avg_turnover, color='blue', linestyle='--',
                  linewidth=2, label=f'期间平均换手率: {avg_turnover:.2%}')
        ax.axhline(y=avg_turnover_trade, color='green', linestyle='--',
                  linewidth=2, label=f'平均换手率: {avg_turnover_trade:.2%}')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('换手率', fontsize=12)
        ax.set_title('换手率分析', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_all(self, figsize: Tuple[int, int] = (20, 16)) -> None:
        """
        综合展示面板
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_backtest()")
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. 净值曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.daily_nav.index, self.daily_nav.values,
                linewidth=2, color='#2E86AB', label='策略净值')
        if len(self.trade_records) > 0:
            trade_dates = self.trade_records['date']
            trade_navs = [self.daily_nav[d] for d in trade_dates if d in self.daily_nav.index]
            ax1.scatter(trade_dates, trade_navs, color='red', s=30,
                       zorder=5, alpha=0.6)
        ax1.set_ylabel('净值', fontsize=10)
        ax1.set_title('策略累计净值曲线', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤曲线
        ax2 = fig.add_subplot(gs[1, 0])
        dd_info = calculate_max_drawdown(self.daily_nav)
        drawdown_series = dd_info['drawdown_series']
        ax2.fill_between(drawdown_series.index, drawdown_series.values, 0,
                        color='#A23B72', alpha=0.3)
        ax2.plot(drawdown_series.index, drawdown_series.values,
                linewidth=1.5, color='#A23B72')
        
        # 标记最大回撤
        max_dd_end = dd_info['drawdown_end']
        max_dd_value = dd_info['max_drawdown']
        
        if max_dd_end in drawdown_series.index:
            ax2.scatter([max_dd_end], [drawdown_series[max_dd_end]], 
                      color='red', s=10, alpha=0.5, zorder=5, label=f'最大回撤: {max_dd_value:.2%}')
            
        ax2.set_ylabel('回撤', fontsize=10)
        ax2.set_title('回撤曲线', fontsize=11, fontweight='bold')
        # 显示图例（如果有 label 被设置）
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 3. 日度收益分布
        ax3 = fig.add_subplot(gs[1, 1])
        monthly_stats = calculate_monthly_returns(self.daily_nav)
        daily_returns = calculate_returns(self.daily_nav)
        # monthly_nav = self.daily_nav.resample('ME').last()
        # monthly_returns = monthly_nav.pct_change().dropna()
        daily_returns_avg = daily_returns.mean()
        
        if len(daily_returns) > 0:
            ax3.hist(daily_returns, bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
            ax3.axvline(x=daily_returns_avg, color='red', linestyle='--', linewidth=1, label=f'日度平均收益率:{daily_returns_avg:.2%}')
            ax3.legend(loc='best', fontsize=8)
            ax3.set_xlabel('日度收益率', fontsize=10)
            ax3.set_ylabel('频数', fontsize=10)
            ax3.set_title('日度收益率分布', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', labelsize=8)
        
        # 4. 换手率
        ax4 = fig.add_subplot(gs[2, 0])
        if len(self.turnover_records) > 0:
            ax4.bar(self.turnover_records['date'], self.turnover_records['turnover'],
                   color='#C73E1D', alpha=0.7, label='换手率')
            avg_turnover = self.turnover_records['turnover'].mean()
            avg_turnover_trade = self.turnover_records['turnover'].replace(0, np.nan).mean()
            ax4.axhline(y=avg_turnover, color='blue', linestyle='--',
                    linewidth=1.5, label=f'期间平均换手率: {avg_turnover:.2%}')
            ax4.axhline(y=avg_turnover_trade, color='green', linestyle='--',
                    linewidth=1.5, label=f'平均换手率: {avg_turnover_trade:.2%}')
            ax4.set_ylabel('换手率', fontsize=10)
            ax4.set_title('换手率', fontsize=11, fontweight='bold')
            # 显示图例（小图需要显式调用 legend）
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', labelsize=8)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 5. 性能指标表
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # 选择关键指标显示
        key_metrics = [
            ('累计收益率', f"{self.metrics.get('累计收益率', 0):.2%}"),
            ('年化收益率', f"{self.metrics.get('年化收益率', 0):.2%}"),
            ('年化波动率', f"{self.metrics.get('年化波动率', 0):.2%}"),
            ('夏普比率', f"{self.metrics.get('夏普比率', 0):.2f}"),
            ('最大回撤', f"{self.metrics.get('最大回撤', 0):.2%}"),
            ('胜率', f"{self.metrics.get('胜率', 0):.2%}"),
        ]
        
        table_data = [[k, v] for k, v in key_metrics]
        table = ax5.table(cellText=table_data, colLabels=['指标', '值'],
                         cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title('关键指标', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('策略回测综合分析', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
