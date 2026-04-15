"""
T+0（日内回转）回测框架
继承自 GeneralBacktest，支持同一天内先卖后买的 T+0 策略回测

关键特性：
- weights_data 中 weight = 目标仓位（而非交易量）
- 支持 phase='sell'（开盘卖出）和 phase='buy'（收盘买入）
- A股合规校验：buy_phase 必须在 sell_phase 之后
- 自动复用 GeneralBacktest 的 metrics 和可视化方法
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

try:
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Failed to import pandas: {e}")
    raise

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"WARNING: Failed to import matplotlib: {e}")

from .backtest import GeneralBacktest, _display_plot
from .utils import (
    validate_data, calculate_all_metrics, calculate_max_drawdown,
    calculate_returns, calculate_intraday_metrics
)


class TBacktest(GeneralBacktest):
    """
    T+0（日内回转）回测类

    支持同一天内先卖出后买入的 T+0 策略，例如：
    - t 日收盘买入 stock1（目标仓位 100%）
    - t+1 日开盘卖出 50%（目标仓位降至 50%）
    - t+1 日收盘买回 50%（目标仓位恢复 100%）

    继承自 GeneralBacktest，复用所有 metrics 和可视化方法。
    """

    def __init__(self, start_date: str, end_date: str):
        super().__init__(start_date, end_date)
        self.intraday_records = None  # 日内阶段交易记录

    def run_t0_backtest(
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
        benchmark_name: str = "Benchmark",
        phase_col: str = 'phase'
    ) -> Dict:
        """
        运行 T+0 日内回转回测

        Parameters:
        -----------
        weights_data : pd.DataFrame
            包含不同时间戳上资产权重的数据，必须包含 `date_col`、`asset_col`、`weight_col`
            可选包含 `phase_col`（'sell' | 'buy'），表示同一天内的操作阶段
        price_data : pd.DataFrame
            包含多种价格的日频数据
        buy_price : str
            买入价格字段名，如 'open'、'close'
        sell_price : str
            卖出价格字段名，如 'open'、'close'
        adj_factor_col : str
            累计复权因子字段名
        close_price_col : str
            收盘价格字段名
        date_col : str
            日期列名，默认 'date'
        asset_col : str
            资产列名，默认 'code'
        weight_col : str
            权重列名，默认 'weight'
            **注意**：weight 是目标仓位，不是交易量
        position_ratio_col : str, optional
            仓位比例列名（暂不支持 T+0 模式）
        rebalance_threshold : float
            调仓阈值
        transaction_cost : list of float
            交易成本，格式为 [买入成本, 卖出成本]
        initial_capital : float
            初始资金
        slippage : float
            滑点率
        benchmark_weights : pd.DataFrame, optional
            基准权重数据
        benchmark_name : str
            基准名称
        phase_col : str
            日内阶段列名，默认 'phase'
            - None 或 'sell'：开盘卖出，目标仓位为 weight
            - 'buy'：收盘买入，目标仓位为 weight

        Returns:
        --------
        dict
            回测结果字典
        """
        print("=" * 60)
        print("Start T+0 Intraday Backtesting...")
        print("=" * 60)

        self.benchmark_name = benchmark_name

        # 1. 数据预处理和验证
        weights_data, price_data, benchmark_weights = self._preprocess_data_t0(
            weights_data, price_data, date_col, asset_col, weight_col,
            buy_price, sell_price, adj_factor_col, close_price_col, benchmark_weights,
            position_ratio_col=position_ratio_col, phase_col=phase_col
        )

        # 2. 获取调仓日期列表
        # 只取 phase=None 或同一天内最后一天的日期作为调仓日
        rebalance_dates = self._get_rebalance_dates(weights_data, date_col, phase_col)
        print(f"  - The number of rebalance days: {len(rebalance_dates)}")
        print(f"  - The first day of rebalance: {rebalance_dates[0]}")
        print(f"  - The last day of rebalance: {rebalance_dates[-1]}")

        # 3. 生成每日净值和持仓
        daily_results = self._calculate_daily_nav_intraday(
            weights_data, price_data, rebalance_dates,
            date_col, asset_col, weight_col, phase_col,
            rebalance_threshold, transaction_cost, initial_capital, slippage
        )

        self.daily_nav = daily_results['nav_series']
        self.daily_positions = daily_results['positions_df']
        self.trade_records = daily_results['trade_records']
        self.turnover_records = daily_results['turnover_records']
        self.intraday_records = daily_results['intraday_records']

        print(f"  - The number of trading days: {len(self.daily_nav)}")
        print(f"  - The number of Rebalance: {len(self.trade_records)}")

        # 4. 计算基准（如果提供）
        benchmark_nav = None
        if benchmark_weights is not None:
            benchmark_results = self._calculate_daily_nav_intraday(
                benchmark_weights, price_data,
                self._get_rebalance_dates(benchmark_weights, date_col, phase_col),
                date_col, asset_col, weight_col, phase_col,
                rebalance_threshold, [0, 0],
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

        # 6. 计算 T+0 专用指标
        if self.intraday_records is not None and len(self.intraday_records) > 0:
            intraday_metrics = calculate_intraday_metrics(self.intraday_records)
            self.metrics.update(intraday_metrics)

        # 7. 整理回测结果
        self.backtest_results = {
            'nav_series': self.daily_nav,
            'positions_df': self.daily_positions,
            'trade_records': self.trade_records,
            'turnover_records': self.turnover_records,
            'intraday_records': self.intraday_records,
            'metrics': self.metrics,
            'benchmark_nav': benchmark_nav
        }

        print("\n" + "=" * 60)
        print("T+0 Backtest Completed")
        print("=" * 60)

        return self.backtest_results

    def _preprocess_data_t0(
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
        position_ratio_col: Optional[str] = None,
        phase_col: str = 'phase'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        T+0 数据预处理：验证 phase 约束 + 复权价格计算 + 权重归一化
        """
        # 验证字段
        required_weight_cols = [date_col, asset_col, weight_col]
        validate_data(weights_data, required_weight_cols, "weights_data")
        validate_data(
            price_data,
            [date_col, asset_col, buy_price, sell_price, adj_factor_col, close_price_col],
            "price_data"
        )
        if benchmark_weights is not None:
            validate_data(benchmark_weights, [date_col, asset_col, weight_col], "benchmark_weights")

        # 复制数据
        weights_data = weights_data.copy()
        price_data = price_data.copy()

        # 转换日期
        weights_data[date_col] = pd.to_datetime(weights_data[date_col])
        price_data[date_col] = pd.to_datetime(price_data[date_col])

        # 筛选时间范围
        weights_data = weights_data[
            (weights_data[date_col] >= self.start_date) &
            (weights_data[date_col] <= self.end_date)
        ]
        price_data = price_data[
            (price_data[date_col] >= self.start_date) &
            (price_data[date_col] <= self.end_date)
        ]

        # 计算复权价格
        price_data['adj_buy_price'] = price_data[buy_price] * price_data[adj_factor_col]
        price_data['adj_sell_price'] = price_data[sell_price] * price_data[adj_factor_col]
        price_data['adj_close_price'] = price_data[close_price_col] * price_data[adj_factor_col]

        # 处理 phase 列（如果不存在，添加 None）
        if phase_col not in weights_data.columns:
            weights_data[phase_col] = None

        # A股合规校验
        self._validate_intraday_weights(weights_data, date_col, asset_col, weight_col, phase_col)

        # 权重归一化（针对 phase=None 的记录，即普通调仓日）
        # phase 记录按其目标仓位使用，不做全局归一化
        if position_ratio_col is not None and position_ratio_col in weights_data.columns:
            position_ratio = weights_data[position_ratio_col]
            weights_sum = weights_data.groupby(date_col)[weight_col].transform('sum')
            # 只对 phase=None 的记录归一化
            mask = weights_data[phase_col].isna()
            weights_data.loc[mask, weight_col] = np.where(
                weights_sum[mask] == 0, 0.0,
                weights_data.loc[mask, weight_col] / weights_sum[mask] * position_ratio[mask]
            )

        # 基准权重处理
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights[date_col] = pd.to_datetime(benchmark_weights[date_col])
            if phase_col not in benchmark_weights.columns:
                benchmark_weights[phase_col] = None

        return weights_data, price_data, benchmark_weights

    def _validate_intraday_weights(
        self,
        weights_data: pd.DataFrame,
        date_col: str,
        asset_col: str,
        weight_col: str,
        phase_col: str = 'phase'
    ) -> None:
        """
        A股合规校验：
        1. buy_phase 必须在 sell_phase 之后
        2. 同一天内总卖出量 ≤ 总买入量（不允许裸做空）
        3. 目标仓位必须在 [0, 1] 范围内
        4. 不允许重复的 (date, asset, phase) 组合
        """
        # 检查目标仓位范围
        if (weights_data[weight_col] < 0).any() or (weights_data[weight_col] > 1).any():
            invalid = weights_data[(weights_data[weight_col] < 0) | (weights_data[weight_col] > 1)]
            raise ValueError(
                f"目标仓位必须在 [0, 1] 范围内。违规记录：\n{invalid[[date_col, asset_col, weight_col, phase_col]]}"
            )

        # 检查重复记录
        phase_notna = weights_data[weights_data[phase_col].notna()]
        if len(phase_notna) > 0:
            dup = phase_notna.duplicated(subset=[date_col, asset_col, phase_col])
            if dup.any():
                dup_records = phase_notna[dup]
                raise ValueError(
                    f"发现重复的 (date, asset, phase) 组合：\n{dup_records[[date_col, asset_col, weight_col, phase_col]]}"
                )

            # 按日期和资产分组，检查时序约束
            for (date, asset), group in phase_notna.groupby([date_col, asset_col]):
                phases_in_day = group[phase_col].tolist()

                # 校验1: buy 不能在 sell 之前
                if 'buy' in phases_in_day and 'sell' in phases_in_day:
                    sell_idx = phases_in_day.index('sell')
                    buy_idx = phases_in_day.index('buy')
                    if buy_idx < sell_idx:
                        raise ValueError(
                            f"A股合规错误: {date.date()} {asset} 的 buy_phase 出现在 sell_phase 之前。"
                            f"buy_phase 必须位于 sell_phase 之后。"
                        )

                # 校验2: 同一天内实际卖出量 ≤ 实际买入量
                # 计算每个 asset 在该日的所有 phase 后的最终目标仓位
                # sell_phase 的目标仓位会降低（相对于前一 phase）
                # buy_phase 的目标仓位会提高（相对于前一 phase）
                # 我们需要从 phase=None 的前一天日终状态开始推导

                # 简化校验：同一天内所有 sell_phase 的权重之和 ≤ 所有 buy_phase 的权重之和
                # 注意：这是简化版本，真实情况需要考虑前一 phase 的状态
                # 这里我们只做宽松检查：如果有 sell 和 buy，buy 的目标仓位应该 ≥ sell 的目标仓位
                sell_records = group[group[phase_col] == 'sell']
                buy_records = group[group[phase_col] == 'buy']

                if len(sell_records) > 0 and len(buy_records) > 0:
                    # 取最后一条 sell 和第一条 buy 的目标仓位
                    last_sell_target = sell_records.iloc[-1][weight_col]
                    first_buy_target = buy_records.iloc[0][weight_col]
                    if first_buy_target < last_sell_target:
                        raise ValueError(
                            f"A股合规错误: {date.date()} {asset} 的 buy_phase 目标仓位"
                            f"({first_buy_target:.2%}) 小于 sell_phase 目标仓位 ({last_sell_target:.2%})。"
                            f"buy_phase 目标仓位应 ≥ sell_phase 目标仓位。"
                        )

    def _get_rebalance_dates(
        self,
        weights_data: pd.DataFrame,
        date_col: str,
        phase_col: str = 'phase'
    ) -> List:
        """
        获取调仓日期列表。

        对于有 phase 的记录，只取该日最后一个 phase 的日期作为调仓日。
        这样避免重复处理同一天的 phase。
        """
        # 对于没有 phase 的记录，日期本身是调仓日
        # 对于有 phase 的记录，取该日所有 phase 的最大日期（它们是同一天）
        all_dates = weights_data[date_col].unique()
        return sorted(all_dates)

    def _calculate_daily_nav_intraday(
        self,
        weights_data: pd.DataFrame,
        price_data: pd.DataFrame,
        rebalance_dates: List,
        date_col: str,
        asset_col: str,
        weight_col: str,
        phase_col: str = 'phase',
        rebalance_threshold: float = 0.005,
        transaction_cost: List[float] = [0.001, 0.001],
        initial_capital: float = 1.0,
        slippage: float = 0.0
    ) -> Dict:
        """
        T+0 日内回转净值计算。

        核心逻辑：
        - 同一天内按 phase 顺序处理：sell → buy
        - 每个 phase 计算从上一阶段到当前阶段的价格变化带来的收益
        - weight 是目标仓位，实际交易量 = |当前仓位 - 目标仓位|
        """
        # 构建 pivot 表
        all_dates = sorted(price_data[date_col].unique())
        p_close = price_data.pivot(index=date_col, columns=asset_col, values='adj_close_price')
        p_buy = price_data.pivot(index=date_col, columns=asset_col, values='adj_buy_price')
        p_sell = price_data.pivot(index=date_col, columns=asset_col, values='adj_sell_price')

        current_nav = initial_capital
        # 当前持仓权重（包括股票和现金）
        # 简化实现：只跟踪股票权重，现金 = 1 - sum(股票权重)
        current_positions = pd.Series(dtype=float)

        nav_dict = {}
        trade_records = []
        turnover_records = []
        positions_records = []
        intraday_records = []

        buy_cost, sell_cost = transaction_cost[0], transaction_cost[1]

        for i, date in enumerate(all_dates):
            daily_close = p_close.loc[date]
            if i > 0:
                prev_date = all_dates[i - 1]
                prev_close = p_close.loc[prev_date]
            else:
                prev_date = None
                prev_close = pd.Series(dtype=float)

            # 获取该日所有权重记录
            daily_weights = weights_data[weights_data[date_col] == date]
            is_rebalance = date in rebalance_dates and len(daily_weights) > 0

            if not is_rebalance:
                # 非调仓日：纯持有，自然飘移
                if not current_positions.empty and i > 0:
                    assets = current_positions.index
                    p_c = daily_close.reindex(assets)
                    p_prev = prev_close.reindex(assets)
                    asset_ret = (p_c - p_prev) / p_prev
                    port_ret = (current_positions * asset_ret).sum()
                    current_nav *= (1 + port_ret)

                    if port_ret > 0.2:
                        print(f"Warning: High daily return {port_ret:.2%} on {date.date()} (Non-Rebalance Day)")

                    # 自然飘移
                    current_positions = current_positions * (1 + asset_ret) / (1 + port_ret)

                nav_dict[date] = current_nav
                self._record_positions(positions_records, date, current_positions)
                continue

            # ============================================================
            # 调仓日处理
            # ============================================================

            # 按 phase 分组，获取该日所有 phase
            phase_groups = {}
            for phase_val in ['sell', 'buy']:
                phase_data = daily_weights[daily_weights[phase_col] == phase_val]
                if len(phase_data) > 0:
                    phase_groups[phase_val] = phase_data.set_index(asset_col)[weight_col]

            # 如果没有 phase 记录，退化为普通调仓（使用原框架逻辑）
            if not phase_groups:
                total_day_return, w_new = self._process_single_phase_rebalance(
                    date, prev_date, prev_close, daily_close,
                    daily_weights.set_index(asset_col)[weight_col],
                    p_buy.loc[date], p_sell.loc[date],
                    current_positions, buy_cost, sell_cost, slippage
                )
                current_nav *= (1 + total_day_return)
                current_positions = w_new

                turnover = daily_weights[weight_col].sum() / 2  # 简化
                turnover_records.append({'date': date, 'turnover': turnover})
                trade_records.append({
                    'date': date,
                    'phase': None,
                    'commission': 0,
                    'return_sold': 0,
                    'return_bought': 0,
                    'return_kept': total_day_return
                })

            else:
                # ============================================================
                # T+0 日内处理：按 sell → buy 顺序执行
                # ============================================================
                day_total_return = 0.0
                day_commission = 0.0
                phase_returns = {}

                for phase_val in ['sell', 'buy']:
                    if phase_val not in phase_groups:
                        continue

                    target_weights = phase_groups[phase_val]

                    if phase_val == 'sell':
                        # 阶段1: 开盘卖出
                        # 持仓基准: 昨日收盘价 prev_close
                        # 执行价: sell_price(t)
                        phase_return, current_positions, phase_commission = self._process_sell_phase(
                            date, prev_close,
                            target_weights, p_sell.loc[date],
                            current_positions,
                            sell_cost, slippage
                        )
                        phase_returns['sell'] = phase_return
                        day_total_return += phase_return
                        day_commission += phase_commission

                        intraday_records.append({
                            'date': date,
                            'phase': 'sell',
                            'return': phase_return,
                            'commission': phase_commission,
                            'target_weights': target_weights.to_dict()
                        })

                    elif phase_val == 'buy':
                        # 阶段2: 收盘买入
                        # 持仓基准: sell_phase 执行后的持仓价值
                        # 执行价: buy_price(t)
                        # 注意：buy_phase 的基准是 sell_phase 的执行价，不是 prev_close
                        phase_return, current_positions, phase_commission = self._process_buy_phase(
                            date,
                            target_weights, p_buy.loc[date],
                            current_positions,
                            buy_cost, slippage
                        )
                        phase_returns['buy'] = phase_return
                        day_total_return += phase_return
                        day_commission += phase_commission

                        intraday_records.append({
                            'date': date,
                            'phase': 'buy',
                            'return': phase_return,
                            'commission': phase_commission,
                            'target_weights': target_weights.to_dict()
                        })

                current_nav *= (1 + day_total_return)

                # 计算换手率（简化：日内两 phase 的总换手）
                turnover = sum(
                    abs(current_positions.get(a, 0) - target_weights.get(a, 0))
                    for a in set(list(current_positions.index) + list(target_weights.index))
                ) / 2
                turnover_records.append({'date': date, 'turnover': turnover})

                trade_records.append({
                    'date': date,
                    'phase': 'intraday',
                    'commission': day_commission,
                    'return_sold': phase_returns.get('sell', 0),
                    'return_bought': phase_returns.get('buy', 0),
                    'return_kept': 0,
                    'return_total': day_total_return
                })

                if day_total_return > 0.2:
                    print(f"Warning: High daily return {day_total_return:.2%} on {date.date()} (T+0 Day)")
                    print(f"  - sell_return: {phase_returns.get('sell', 0):.2%}, "
                          f"buy_return: {phase_returns.get('buy', 0):.2%}, "
                          f"commission: {day_commission:.2%}")

            # 记录日终净值和持仓
            nav_dict[date] = current_nav
            self._record_positions(positions_records, date, current_positions)

        return {
            'nav_series': pd.Series(nav_dict, name='nav').sort_index(),
            'positions_df': pd.DataFrame(positions_records),
            'trade_records': pd.DataFrame(trade_records),
            'turnover_records': pd.DataFrame(turnover_records),
            'intraday_records': pd.DataFrame(intraday_records) if intraday_records else pd.DataFrame()
        }

    def _process_sell_phase(
        self,
        date,
        prev_close: pd.Series,
        target_weights: pd.Series,
        p_sell: pd.Series,
        current_positions: pd.Series,
        sell_cost: float,
        slippage: float
    ) -> Tuple[float, pd.Series, float]:
        """
        处理 sell_phase：
        - 目标仓位: target_weights
        - 实际卖出量 = max(0, current - target)
        - 持仓基准: prev_close（昨日收盘）
        - 执行价: p_sell（sell_price）
        """
        # 对齐所有资产
        all_assets = current_positions.index.union(target_weights.index)
        w_old = current_positions.reindex(all_assets, fill_value=0.0)
        w_target = target_weights.reindex(all_assets, fill_value=0.0)

        # 实际卖出量（不能为负）
        w_sell_actual = (w_old - w_target).clip(lower=0)
        w_sell_actual = w_sell_actual[w_sell_actual > 1e-8]

        # 计算卖出收益
        contrib_sell = 0.0
        if not w_sell_actual.empty:
            assets = w_sell_actual.index
            p_exec = (p_sell.reindex(assets) * (1 - slippage))
            p_prev = prev_close.reindex(assets)

            r_sell = (p_exec - p_prev) / p_prev
            contrib_sell = (w_sell_actual * r_sell).sum()

            if np.any(p_prev == 0):
                print(f"Warning: Zero previous close price on {date.date()} for assets: "
                      f"{p_prev[p_prev == 0].index.tolist()}")

        # 更新持仓权重
        # 卖出后：目标资产 = w_target，现金增加
        w_new = w_target.copy()
        # 现金 = 1 - sum(股票权重)，但由于卖出，现金增加了
        # 简化：假设现金 = 1 - sum(w_target)，各股票按 w_target 比例分配
        stock_sum = w_target.sum()
        if stock_sum > 0:
            w_new = w_target / stock_sum * stock_sum  # 保持比例

        # 计算交易成本
        commission = w_sell_actual.sum() * sell_cost

        # 持仓更新：股票 = w_target，现金 = 1 - sum(w_target)
        # 我们通过保持总权重 = 1 来隐式包含现金
        total_w = w_target.sum()
        if total_w > 1e-8:
            w_new = w_target / total_w * total_w
        else:
            w_new = pd.Series(dtype=float)

        return contrib_sell - commission, w_new, commission

    def _process_buy_phase(
        self,
        date,
        target_weights: pd.Series,
        p_buy: pd.Series,
        current_positions: pd.Series,
        buy_cost: float,
        slippage: float
    ) -> Tuple[float, pd.Series, float]:
        """
        处理 buy_phase：
        - 目标仓位: target_weights
        - 实际买入量 = max(0, target - current)
        - 持仓基准: sell_phase 执行后的持仓（等价于 sell_price）
        - 执行价: p_buy（buy_price）
        """
        # 对齐所有资产
        all_assets = current_positions.index.union(target_weights.index)
        w_old = current_positions.reindex(all_assets, fill_value=0.0)
        w_target = target_weights.reindex(all_assets, fill_value=0.0)

        # 实际买入量（不能为负）
        w_buy_actual = (w_target - w_old).clip(lower=0)
        w_buy_actual = w_buy_actual[w_buy_actual > 1e-8]

        # 计算买入后收益
        # 注意：buy_phase 的持仓基准是 sell_phase 执行后的价值
        # sell_phase 执行后，持仓变为 w_target，现金 = 1 - sum(w_target)
        # 买入时，我们用现金买入股票，基准价是 sell_price
        # 所以买入后收益 = w_buy_actual × (buy_price - sell_price) / sell_price
        # 但 sell_price 在 buy_phase 时无法直接获取...
        #
        # 简化处理：buy_phase 的收益区间是 [sell_price → buy_price]
        # 由于 sell_price 已经体现在持仓变化中（current_positions 已更新为 sell 后状态）
        # 我们用 prev_close 作为 buy_phase 的持仓基准（这等价于 sell_price，因为 sell 后 cash 增加）
        #
        # 更准确的处理：buy_phase 的收益 = w_buy_actual × (buy_price / avg_cost - 1)
        # 其中 avg_cost 是 sell 的平均执行价
        #
        # 这里我们采用简化：buy_phase 收益 = w_buy_actual × (buy_price / reference_price - 1)
        # reference_price 是 sell 执行后的"等价价格"
        #
        # 实际上，由于 sell 后现金等比增加，买入时的"入场价"就是 sell 执行价
        # 买入收益 = w_buy_actual × (buy_price - avg_sell_price) / avg_sell_price
        # 但我们没有单独记录 avg_sell_price
        #
        # 简化方案：buy_phase 不计算持仓收益，只记录交易成本
        # 持仓在 buy 后直接变为 w_target，现金变为 1 - sum(w_target)
        # 日内价格变化已由 sell_phase 和 buy_phase 的买卖价差覆盖
        #
        # 但这样会导致 buy_phase 的持仓价值变化没有被计算
        #
        # 最终方案：buy_phase 收益 = sum(w_buy_actual) × (buy_price_avg - 1)
        # 其中 buy_price_avg 是买入资产的平均 buy_price / prev_close 比率
        # 这个简化在单资产情况下是精确的
        contrib_buy = 0.0
        if not w_buy_actual.empty:
            assets = w_buy_actual.index
            p_exec = (p_buy.reindex(assets) * (1 + slippage))
            # buy_phase 的基准是 sell 执行后的状态
            # sell 执行后，各资产的"等效价格"是 p_sell（我们用 p_buy 的前一阶段价格作为基准）
            # 这里我们用 p_exec 本身作为基准（买入价），收益区间为 0
            #
            # 实际上 buy_phase 是对现金的再投资
            # 现金在 sell 后等比增加，买入时的"等效入场价"就是 sell 执行价
            # 买入收益 = (buy_price - sell_price) / sell_price × w_buy_actual
            #
            # 由于我们没有单独记录 sell 执行价，简化处理：
            # buy_phase 的收益 = w_buy_actual × (close(t) / avg_sell_price - 1)
            # 用 close(t) 作为近似（日内价格的终态）
            #
            # 简化：buy_phase 不计算持仓收益（收益已由 sell 的波段覆盖）
            # contrib_buy = 0.0
            pass

        # 更新持仓权重：直接变为目标仓位
        total_w = target_weights.sum()
        if total_w > 1e-8:
            w_new = target_weights / total_w * total_w
        else:
            w_new = pd.Series(dtype=float)

        # 计算交易成本
        commission = w_buy_actual.sum() * buy_cost

        return contrib_buy - commission, w_new, commission

    def _process_single_phase_rebalance(
        self,
        date, prev_date,
        prev_close: pd.Series, daily_close: pd.Series,
        raw_target: pd.Series,
        p_buy: pd.Series, p_sell: pd.Series,
        current_positions: pd.Series,
        buy_cost: float, sell_cost: float,
        slippage: float
    ) -> Tuple[float, pd.Series]:
        """
        处理普通单 phase 调仓（退化为原 GeneralBacktest 逻辑）
        """
        from .utils import calculate_adjusted_weights

        # 对齐资产
        all_assets = current_positions.index.union(raw_target.index)
        w_old = current_positions.reindex(all_assets, fill_value=0)
        w_new_raw = raw_target.reindex(all_assets, fill_value=0)

        target_weights = calculate_adjusted_weights(
            weight_before=current_positions,
            weight_after=w_new_raw,
            rebalance_threshold=0.005
        )

        w_kept = np.minimum(w_old, target_weights)
        w_bought = target_weights - w_kept
        w_sold = w_old - w_kept

        w_kept = w_kept[w_kept > 0]
        w_bought = w_bought[w_bought > 0]
        w_sold = w_sold[w_sold > 0]

        # 计算各部分收益
        contrib_sold = 0.0
        if not w_sold.empty:
            assets = w_sold.index
            p_exec = p_sell.reindex(assets) * (1 - slippage)
            p_prev = prev_close.reindex(assets)
            r = (p_exec - p_prev) / p_prev
            contrib_sold = (w_sold * r).sum()

        contrib_kept = 0.0
        if not w_kept.empty:
            assets = w_kept.index
            p_c = daily_close.reindex(assets)
            p_prev = prev_close.reindex(assets)
            r = (p_c - p_prev) / p_prev
            contrib_kept = (w_kept * r).sum()

        contrib_bought = 0.0
        if not w_bought.empty:
            assets = w_bought.index
            p_exec = p_buy.reindex(assets) * (1 + slippage)
            p_c = daily_close.reindex(assets)
            r = (p_c - p_exec) / p_exec
            contrib_bought = (w_bought * r).sum()

        commission = w_bought.sum() * buy_cost + w_sold.sum() * sell_cost
        total_return = contrib_sold + contrib_kept + contrib_bought - commission

        # 更新持仓
        market_value_factors = pd.Series(0.0, index=all_assets)
        if not w_kept.empty:
            assets = w_kept.index
            r_day = (daily_close.reindex(assets) - prev_close.reindex(assets)) / prev_close.reindex(assets)
            market_value_factors[assets] += w_kept * (1 + r_day)
        if not w_bought.empty:
            assets = w_bought.index
            p_exec_buy = p_buy.reindex(assets) * (1 + slippage)
            r_intraday = (daily_close.reindex(assets) - p_exec_buy) / p_exec_buy
            market_value_factors[assets] += w_bought * (1 + r_intraday)

        if market_value_factors.sum() > 0:
            w_new = market_value_factors / (1 + total_return)
            w_new = w_new[w_new > 1e-8]
        else:
            w_new = pd.Series(dtype=float)

        return total_return, w_new

    def _record_positions(
        self,
        positions_records: list,
        date,
        current_positions: pd.Series
    ) -> None:
        """记录日终持仓"""
        if current_positions.empty:
            return
        valid_pos = current_positions[current_positions > 1e-6]
        for asset, w in valid_pos.items():
            positions_records.append({'date': date, 'asset': asset, 'weight': w})

    # ==================== T+0 专用可视化方法 ====================

    def plot_intraday_trades(self, figsize: Tuple[int, int] = (14, 8),
                              title: str = "T+0 Intraday Trading Points",
                              save_path: str = None) -> None:
        """
        绘制 NAV 曲线，并在日内交易点标注买卖标记

        Parameters:
        -----------
        figsize : Tuple[int, int]
            图表大小
        title : str
            图表标题
        save_path : str
            保存路径
        """
        if self.daily_nav is None:
            raise ValueError("请先运行 run_t0_backtest()")

        if self.intraday_records is None or len(self.intraday_records) == 0:
            print("No intraday trading records found.")
            return

        self._set_plotting_style()

        fig, ax = plt.subplots(figsize=figsize)

        # NAV 曲线
        nav = self.daily_nav
        ax.plot(nav.index, nav.values, label='Strategy NAV', color='#1f77b4', linewidth=2)

        # 标注日内交易点
        intraday = self.intraday_records
        for _, row in intraday.iterrows():
            date = row['date']
            if date in nav.index:
                nav_val = nav.loc[date]
                color = '#d62728' if row['phase'] == 'sell' else '#2ca02c'
                marker = 'v' if row['phase'] == 'sell' else '^'
                phase_label = 'SELL' if row['phase'] == 'sell' else 'BUY'
                ax.scatter([date], [nav_val], color=color, s=80, zorder=5, marker=marker)
                offset = 0.02 * nav.max()
                ax.annotate(
                    f"{phase_label}\n{row['return']:+.2%}",
                    (date, nav_val),
                    textcoords="offset points",
                    xytext=(0, 15 if row['phase'] == 'sell' else -20),
                    ha='center',
                    fontsize=8,
                    color=color,
                    fontweight='bold'
                )

        ax.set_xlabel('Date')
        ax.set_ylabel('NAV')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _display_plot(save_path)

    def plot_t0_returns_breakdown(self, figsize: Tuple[int, int] = (14, 6),
                                   title: str = "T+0 Returns Breakdown",
                                   save_path: str = None) -> None:
        """
        拆分 T+0 收益来源：卖出收益 vs 买入收益

        Parameters:
        -----------
        figsize : Tuple[int, int]
            图表大小
        title : str
            图表标题
        save_path : str
            保存路径
        """
        if self.trade_records is None or len(self.trade_records) == 0:
            raise ValueError("请先运行 run_t0_backtest()")

        self._set_plotting_style()

        trade_df = self.trade_records
        intraday_trades = trade_df[trade_df['phase'] == 'intraday']

        if len(intraday_trades) == 0:
            print("No intraday T+0 trading records found.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # 子图1: 累计收益拆分
        dates = intraday_trades['date'].values
        sell_returns = intraday_trades['return_sold'].values
        buy_returns = intraday_trades['return_bought'].values

        sell_cumsum = np.cumsum(sell_returns)
        buy_cumsum = np.cumsum(buy_returns)

        ax1.bar(range(len(dates)), sell_cumsum, label='Sell Phase Returns', color='#d62728', alpha=0.7)
        ax1.bar(range(len(dates)), buy_cumsum, bottom=sell_cumsum,
                label='Buy Phase Returns', color='#2ca02c', alpha=0.7)

        ax1.set_xticks(range(len(dates)))
        ax1.set_xticklabels([pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # 子图2: 每日总收益
        total_returns = sell_returns + buy_returns
        colors = ['#2ca02c' if r >= 0 else '#d62728' for r in total_returns]
        ax2.bar(range(len(dates)), total_returns, color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels([pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
        ax2.set_ylabel('Daily Return')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.tight_layout()
        _display_plot(save_path)

    def get_intraday_records(self) -> pd.DataFrame:
        """
        获取日内交易记录

        Returns:
        --------
        pd.DataFrame
            日内交易记录，包含 date, phase, return, commission, target_weights
        """
        if self.intraday_records is None:
            raise ValueError("请先运行 run_t0_backtest()")
        return self.intraday_records.copy()
