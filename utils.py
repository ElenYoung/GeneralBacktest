"""
通用回测框架辅助函数模块
包含性能指标计算、向量化数据处理等功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


# ==================== 性能指标计算函数（向量化） ====================

def calculate_returns(nav_series: pd.Series) -> pd.Series:
    """
    计算收益率序列
    
    Parameters:
    -----------
    nav_series : pd.Series
        净值序列
        
    Returns:
    --------
    pd.Series
        收益率序列
    """
    return nav_series.pct_change().fillna(0)


def calculate_annualized_return(nav_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化收益率
    
    Parameters:
    -----------
    nav_series : pd.Series
        净值序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        年化收益率
    """
    if len(nav_series) < 2:
        return 0.0
    
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_periods = len(nav_series) - 1
    
    if n_periods == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return annualized


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化波动率
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        年化波动率
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series, 
                          periods_per_year: int = 252) -> float:
    """
    计算夏普比率
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        夏普比率
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns/ periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(nav_series: pd.Series) -> Dict[str, any]:
    """
    计算最大回撤及相关信息
    
    Parameters:
    -----------
    nav_series : pd.Series
        净值序列，index为日期
        
    Returns:
    --------
    dict
        包含最大回撤、回撤开始日期、回撤结束日期、回撤持续天数等信息
    """
    if len(nav_series) < 2:
        return {
            'max_drawdown': 0.0,
            'drawdown_start': None,
            'drawdown_end': None,
            'drawdown_duration': 0,
            'drawdown_series': pd.Series(0, index=nav_series.index)
        }
    
    # 计算累计最高净值
    cummax = nav_series.cummax()
    
    # 计算回撤序列
    drawdown = (nav_series - cummax) / cummax
    
    # 最大回撤
    max_dd = drawdown.min()
    
    # 找到最大回撤的位置
    max_dd_end = drawdown.idxmin()
    
    # 找到最大回撤开始的位置（最大回撤结束前的最高点）
    max_dd_start = nav_series.loc[:max_dd_end].idxmax()
    
    # 计算持续天数
    duration = len(nav_series.loc[max_dd_start:max_dd_end]) - 1
    
    return {
        'max_drawdown': abs(max_dd),
        'drawdown_start': max_dd_start,
        'drawdown_end': max_dd_end,
        'drawdown_duration': duration,
        'drawdown_series': drawdown
    }


def calculate_calmar_ratio(nav_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算卡玛比率（Calmar Ratio）= 年化收益率 / 最大回撤
    
    Parameters:
    -----------
    nav_series : pd.Series
        净值序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        卡玛比率
    """
    ann_return = calculate_annualized_return(nav_series, periods_per_year)
    max_dd = calculate_max_drawdown(nav_series)['max_drawdown']
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / max_dd


def calculate_sortino_ratio(returns: pd.Series,
                           periods_per_year: int = 252) -> float:
    """
    计算索提诺比率（Sortino Ratio）- 只考虑下行波动
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        索提诺比率
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns/periods_per_year
    
    # 只考虑负收益的波动
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    
    return sortino


def calculate_win_rate(returns: pd.Series) -> float:
    """
    计算胜率（正收益天数占比）
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
        
    Returns:
    --------
    float
        胜率（0-1之间）
    """
    if len(returns) == 0:
        return 0.0
    
    # 排除第一个NaN或0值
    valid_returns = returns[returns != 0]
    
    if len(valid_returns) == 0:
        return 0.0
    
    win_rate = (valid_returns > 0).sum() / len(valid_returns)
    return win_rate


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                                periods_per_year: int = 252) -> float:
    """
    计算信息比率（Information Ratio）= 年化超额收益 / 跟踪误差
    
    Parameters:
    -----------
    returns : pd.Series
        策略收益率序列
    benchmark_returns : pd.Series
        基准收益率序列
    periods_per_year : int
        每年的周期数，默认252（交易日）
        
    Returns:
    --------
    float
        信息比率
    """
    if len(returns) < 2:
        return 0.0
    
    # 对齐两个序列
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
    
    if len(aligned_returns) < 2:
        return 0.0
    
    # 超额收益
    excess_returns = aligned_returns - aligned_benchmark
    
    # 跟踪误差
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    # 年化超额收益
    ann_excess_return = excess_returns.mean() * periods_per_year
    
    ir = ann_excess_return / tracking_error
    return ir


def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    计算风险价值（Value at Risk）
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    confidence_level : float
        置信水平，默认95%
        
    Returns:
    --------
    float
        VaR值（负数表示损失）
    """
    if len(returns) < 2:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    计算条件风险价值（Conditional Value at Risk / Expected Shortfall）
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    confidence_level : float
        置信水平，默认95%
        
    Returns:
    --------
    float
        CVaR值（负数表示损失）
    """
    if len(returns) < 2:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    
    # CVaR是所有小于VaR的收益的平均值
    cvar = returns[returns <= var].mean()
    
    return cvar


def calculate_turnover(weights_df: pd.DataFrame, date_col: str = 'date',
                      asset_col: str = 'code', weight_col: str = 'weight') -> pd.DataFrame:
    """
    计算换手率（向量化）
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        权重数据，包含date_col, asset_col, weight_col
    date_col : str
        日期列名
    asset_col : str
        资产列名
    weight_col : str
        权重列名
        
    Returns:
    --------
    pd.DataFrame
        每个调仓日的换手率
    """
    # 透视表：日期 x 资产
    weights_pivot = weights_df.pivot_table(
        index=date_col, 
        columns=asset_col, 
        values=weight_col, 
        fill_value=0
    )
    
    # 计算权重变化
    weight_changes = weights_pivot.diff().abs()
    
    # 换手率 = 权重变化的总和 / 2
    turnover = weight_changes.sum(axis=1) / 2
    
    return turnover.to_frame('turnover')


# ==================== 向量化数据处理函数 ====================

def validate_data(df: pd.DataFrame, required_cols: list, name: str = "DataFrame") -> None:
    """
    验证数据框是否包含必需的列
    
    Parameters:
    -----------
    df : pd.DataFrame
        要验证的数据框
    required_cols : list
        必需的列名列表
    name : str
        数据框名称，用于错误信息
        
    Raises:
    -------
    ValueError
        如果缺少必需的列
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} lacking of `{missing_cols}`")


def align_dates(df1: pd.DataFrame, df2: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐两个数据框的日期
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        要对齐的数据框
    date_col : str
        日期列名
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        对齐后的数据框
    """
    common_dates = set(df1[date_col].unique()) & set(df2[date_col].unique())
    
    df1_aligned = df1[df1[date_col].isin(common_dates)].copy()
    df2_aligned = df2[df2[date_col].isin(common_dates)].copy()
    
    return df1_aligned, df2_aligned


def fill_nav_between_rebalances(price_data: pd.DataFrame, rebalance_dates: list,
                                date_col: str, asset_col: str, 
                                adj_price_col: str) -> pd.DataFrame:
    """
    在调仓日之间填充净值曲线（向量化）
    
    该函数处理在两个调仓日之间的净值计算，基于复权价格的变化
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据，包含日期、资产、复权价格
    rebalance_dates : list
        调仓日期列表
    date_col : str
        日期列名
    asset_col : str
        资产列名
    adj_price_col : str
        复权价格列名
        
    Returns:
    --------
    pd.DataFrame
        包含每日价格变化率的数据框
    """
    # 计算每个资产的日收益率（基于复权价格）
    price_data = price_data.sort_values([asset_col, date_col])
    price_data['price_change_pct'] = price_data.groupby(asset_col)[adj_price_col].pct_change()
    
    return price_data


def calculate_position_drift(positions: pd.DataFrame, price_changes: pd.Series) -> pd.Series:
    """
    计算持仓漂移（向量化）
    
    Parameters:
    -----------
    positions : pd.DataFrame
        持仓权重
    price_changes : pd.Series
        价格变化率
        
    Returns:
    --------
    pd.Series
        漂移后的持仓权重
    """
    # 向量化计算持仓价值变化
    new_values = positions * (1 + price_changes)
    total_value = new_values.sum()
    
    if total_value == 0:
        return positions
    
    # 归一化得到新权重
    new_weights = new_values / total_value
    
    return new_weights


def calculate_transaction_costs(weight_before: pd.Series, weight_after: pd.Series,
                                buy_cost: float, sell_cost: float) -> float:
    """
    计算交易成本
    
    Parameters:
    -----------
    weight_before : pd.Series
        调仓前权重
    weight_after : pd.Series
        调仓后权重
    buy_cost : float
        买入成本率
    sell_cost : float
        卖出成本率
        
    Returns:
    --------
    float
        总交易成本（相对于总资产的比例）
    """
    # 对齐两个权重序列
    all_assets = weight_before.index.union(weight_after.index)
    weight_before = weight_before.reindex(all_assets, fill_value=0)
    weight_after = weight_after.reindex(all_assets, fill_value=0)
    
    # 权重变化
    weight_change = weight_after - weight_before
    
    # 买入成本
    buy_amount = weight_change[weight_change > 0].sum()
    buy_total_cost = buy_amount * buy_cost
    
    # 卖出成本
    sell_amount = abs(weight_change[weight_change < 0].sum())
    sell_total_cost = sell_amount * sell_cost
    
    total_cost = buy_total_cost + sell_total_cost
    
    return total_cost

def calculate_adjusted_weights(weight_before: pd.Series, 
                               weight_after: pd.Series, 
                               rebalance_threshold: float) -> pd.Series:
    """
    计算考虑交易门槛（绝对阈值）后的调整权重。
    
    参数:
    - rebalance_threshold: 绝对阈值 (例如 0.01 代表 1% 的仓位变动)
    
    逻辑:
    1. 如果 abs(weight_after - weight_before) < threshold，则保持原权重。
    2. 剩余的资金空间 (1 - 锁定的原权重之和) 分配给需要交易的标的。
    """
    
    # 1. 对齐并填充缺失值
    weight_before_assets = weight_before.index
    weight_after_assets = weight_after.index
    all_assets = weight_before_assets.union(weight_after_assets)
    
    w_old = weight_before.reindex(all_assets, fill_value=0.0)
    w_new = weight_after.reindex(all_assets, fill_value=0.0)
    
    # 2. 计算绝对权重变化 (Absolute Weight Change)
    weight_diff = (w_new - w_old).abs()
    
    # 3. 判定哪些资产需要保持不动 (Keep)
    # 条件：绝对变化量小于阈值
    keep_mask = weight_diff < rebalance_threshold
    
    # 4. 初始化最终权重序列
    final_weights = pd.Series(0.0, index=all_assets)
    
    # 4.1 对于不调仓的资产，锁定使用原权重
    final_weights[keep_mask] = w_old[keep_mask]
    
    # 4.2 计算剩余可用仓位 (Remaining Capacity)
    # 目标总仓位通常是 w_new 的总和 (一般为 1.0)
    target_total_leverage = w_new.sum()
    used_capacity = final_weights[keep_mask].sum()
    remaining_capacity = target_total_leverage - used_capacity
    
    # 5. 对需要调仓的资产进行再分配 (Re-allocate)
    trade_mask = ~keep_mask
    
    # 如果没有需要交易的资产，直接返回
    if trade_mask.sum() == 0:
        return final_weights
        
    # 获取需要交易资产的目标权重总和
    target_weight_sum_for_traders = w_new[trade_mask].sum()
    
    if target_weight_sum_for_traders == 0:
        # 特殊情况：所有需要动的仓位目标都是0（全部清仓），不需要分配剩余空间
        final_weights[trade_mask] = 0.0
    else:
        # 归一化因子：将剩余仓位按比例分配给需要交易的资产
        # 这一步保证了 final_weights 的总和依然等于 target_total_leverage
        scale_factor = remaining_capacity / target_weight_sum_for_traders
        final_weights[trade_mask] = w_new[trade_mask] * scale_factor
        
    return final_weights

    




def calculate_monthly_returns(nav_series: pd.Series) -> pd.DataFrame:
    """
    计算月度收益率统计
    
    Parameters:
    -----------
    nav_series : pd.Series
        净值序列，index为日期
        
    Returns:
    --------
    pd.DataFrame
        月度收益率统计
    """
    # 重采样到月度
    monthly_nav = nav_series.resample('ME').last()
    monthly_returns = monthly_nav.pct_change().dropna()
    
    stats = {
        'mean': monthly_returns.mean(),
        'std': monthly_returns.std(),
        'min': monthly_returns.min(),
        'max': monthly_returns.max(),
        'positive_months': (monthly_returns > 0).sum(),
        'negative_months': (monthly_returns < 0).sum(),
        'total_months': len(monthly_returns)
    }
    
    return pd.DataFrame([stats])


def calculate_all_metrics(nav_series: pd.Series, benchmark_nav: Optional[pd.Series] = None,
                         trade_dates: Optional[list] = None, turnover_series: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.03, periods_per_year: int = 252) -> Dict:
    """
    计算所有性能指标（一站式函数）
    
    Parameters:
    -----------
    nav_series : pd.Series
        策略净值序列
    benchmark_nav : pd.Series, optional
        基准净值序列
    trade_dates : list, optional
        交易日期列表
    turnover_series : pd.Series, optional
        换手率序列
    risk_free_rate : float
        无风险利率
    periods_per_year : int
        每年的周期数
        
    Returns:
    --------
    dict
        包含所有性能指标的字典
    """
    # 计算收益率
    returns = calculate_returns(nav_series)
    
    # 基础指标
    metrics = {
        '累计收益率': nav_series.iloc[-1] / nav_series.iloc[0] - 1,
        '年化收益率': calculate_annualized_return(nav_series, periods_per_year),
        '年化波动率': calculate_annualized_volatility(returns, periods_per_year),
        '夏普比率': calculate_sharpe_ratio(returns, periods_per_year),
        '索提诺比率': calculate_sortino_ratio(returns, periods_per_year),
        '卡玛比率': calculate_calmar_ratio(nav_series, periods_per_year),
        '胜率': calculate_win_rate(returns),
        'VaR (95%)': calculate_var(returns, 0.95),
        'CVaR (95%)': calculate_cvar(returns, 0.95),
    }
    
    # 最大回撤相关
    dd_info = calculate_max_drawdown(nav_series)
    metrics['最大回撤'] = dd_info['max_drawdown']
    metrics['最大回撤开始日期'] = dd_info['drawdown_start']
    metrics['最大回撤结束日期'] = dd_info['drawdown_end']
    metrics['最大回撤持续天数'] = dd_info['drawdown_duration']
    
    # 交易相关
    if trade_dates is not None:
        metrics['交易次数'] = len(trade_dates)
    
    if turnover_series is not None:
        metrics['平均换手率'] = turnover_series.mean()
        metrics['累计换手率'] = turnover_series.sum()
    
    # 相对基准的指标
    if benchmark_nav is not None:
        benchmark_returns = calculate_returns(benchmark_nav)
        
        # 对齐
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        metrics['基准累计收益率'] = benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
        metrics['基准年化收益率'] = calculate_annualized_return(benchmark_nav, periods_per_year)
        metrics['超额收益'] = metrics['累计收益率'] - metrics['基准累计收益率']
        metrics['年化超额收益'] = metrics['年化收益率'] - metrics['基准年化收益率']
        metrics['信息比率'] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        
        # 跟踪误差
        excess_returns = aligned_returns - aligned_benchmark
        metrics['跟踪误差'] = excess_returns.std() * np.sqrt(periods_per_year)
    
    return metrics
