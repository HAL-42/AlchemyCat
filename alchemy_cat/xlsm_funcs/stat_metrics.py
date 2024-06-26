#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/22 14:12
@File    : stat_metrics.py
@Software: PyCharm
@Desc    : 
"""
from math import sqrt
from functools import partial

import xlwings as xw
import numpy as np
import pandas as pd
import scipy.stats as st


@xw.func(async_mode='threading')
@xw.arg('X', np.ndarray, dtype=np.float64, doc='所有样本。')
@xw.ret(numbers=int)
def 样本数(X: np.ndarray) -> int:
    return X.size


@xw.func(async_mode='threading')
@xw.arg('X', np.ndarray, dtype=np.float64, doc='所有样本。')
@xw.arg('ddof', numbers=int, doc='归一化时，自由度的减数。')
@xw.ret(numbers=float)
def 标准差(X: np.ndarray, ddof: int = 1) -> float:
    return X.std(ddof=ddof)


@xw.func(async_mode='threading')
@xw.arg('X', np.ndarray, dtype=np.float64, doc='所有样本。')
@xw.ret(numbers=float)
def 均值(X: np.ndarray) -> float:
    return X.mean()


@xw.func(async_mode='threading')
@xw.arg('X', np.ndarray, doc='所有样本。')
@xw.ret(transpose=True)
def 最小最大值(X: np.ndarray) -> np.ndarray:
    return np.array([np.amin(X), np.amax(X)])


@xw.func(async_mode='threading')
@xw.arg('alpha', numbers=float, doc='1 - 置信度。')
@xw.arg('X', np.ndarray, dtype=np.float64, doc='所有样本。')
@xw.arg('sigma', numbers=float, doc='样本所来自分布的标准差，若缺省，则用t分布计算置信区间。')
@xw.ret(transpose=True)
def 置信区间(alpha: float, X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        return np.array(st.t.interval(alpha=1 - alpha, df=X.size - 1,
                                      loc=X.mean(), scale=st.sem(X, axis=None)), dtype=np.float64)
    else:
        return np.array(st.norm.interval(alpha=1 - alpha, loc=X.mean(),
                                         scale=sigma / (X.size ** 0.5)), dtype=np.float64)


@xw.func(async_mode='threading')
@xw.arg('alpha', numbers=float, doc='1 - 置信度。')
@xw.arg('X', np.ndarray, dtype=np.float64, doc='所有样本。')
@xw.arg('sigma', numbers=float, doc='样本所来自分布的标准差，若缺省，则用t分布计算置信区间。')
@xw.ret(transpose=True)
def 单侧置信区间(alpha: float, X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    return 置信区间(alpha * 2, X, sigma)


@xw.func(async_mode='threading')
@xw.arg('metrics', pd.DataFrame, dtype=np.float64, index=0, header=1, doc='所有样本的各种指标。')
@xw.arg('alpha', numbers=float, doc='1 - 置信度。')
@xw.arg('sigma', numbers=float, ndim=1, doc='样本所来自分布的标准差，若缺省，则用t分布计算置信区间。')
@xw.ret(index=True, header=True)
def 总结多次实验(metrics: pd.DataFrame, alpha: float=.05, sigma: list | None = None) -> pd.DataFrame:
    if sigma is not None:
        assert len(sigma) == len(metrics.columns)
    else:
        sigma = [sigma] * len(metrics.columns)

    index = ['样本数', '均值', '无偏标准差', '有偏标准差', '最小值', '最大值',
             '置信区间下限', '置信区间上限', '置信区间半径', '单侧置信下限', '单侧置信上限']
    conclude = pd.DataFrame(data=np.full((len(index), len(metrics.columns)), np.nan, dtype=np.float64),
                            index=index, columns=metrics.columns)

    for column, s in zip(conclude.columns, sigma):
        con, X = conclude[column], metrics[column].values
        con['样本数'] = 样本数(X)
        con['均值'] = 均值(X)
        con['无偏标准差'] = 标准差(X, ddof=1)
        con['有偏标准差'] = 标准差(X, ddof=0)
        con['最小值'], con['最大值'] = 最小最大值(X)
        con['置信区间下限'], con['置信区间上限'] = 置信区间(alpha, X, s)
        con['置信区间半径'] = (con['置信区间上限'] - con['置信区间下限']) / 2
        con['单侧置信下限'], con['单侧置信上限'] = 单侧置信区间(alpha, X, s)

    return conclude


@xw.func(async_mode='threading')
@xw.arg('alpha', numbers=float, doc='1 - 置信度。')
@xw.arg('n_X', numbers=int, doc='总体X的样本总数。')
@xw.arg('n_Y', numbers=int, doc='总体Y的样本总数。')
@xw.arg('avg_X', numbers=float, doc='总体X的样本均值。')
@xw.arg('avg_Y', numbers=float, doc='总体Y的样本均值。')
@xw.arg('sigma_X', numbers=float, doc='总体X所来自分布的标准差，或样本的标准差。')
@xw.arg('sigma_Y', numbers=float, doc='总体Y所来自分布的标准差，或样本的标准差。')
@xw.arg('use_t', numbers=bool, doc='是否使用t分布计算置信区间，默认为True。')
@xw.ret(transpose=True)
def 总体间均值差置信区间(alpha: float,
                         n_X: int, n_Y: int,
                         avg_X: float, avg_Y: float,
                         sigma_X: float, sigma_Y: float,
                         use_t: bool = True) -> np.ndarray:
    loc = avg_X - avg_Y

    if not use_t:
        scale = sqrt((sigma_X ** 2 / n_X) + (sigma_Y ** 2 / n_Y))
    else:
        df = n_X + n_Y - 2
        scale = sqrt((((n_X - 1) * sigma_X ** 2) + ((n_Y - 1) * sigma_Y ** 2)) / df)
        scale *= sqrt((1 / n_X) + (1 / n_Y))

    if not use_t:
        return np.array(st.norm.interval(alpha=1 - alpha, loc=loc, scale=scale), dtype=np.float64)
    else:
        return np.array(st.t.interval(alpha=1 - alpha, df=df, loc=loc, scale=scale), dtype=np.float64)


@xw.func(async_mode='threading')
@xw.arg('alpha', numbers=float, doc='1 - 置信度。')
@xw.arg('n_X', numbers=int, doc='总体X的样本总数。')
@xw.arg('n_Y', numbers=int, doc='总体Y的样本总数。')
@xw.arg('avg_X', numbers=float, doc='总体X的样本均值。')
@xw.arg('avg_Y', numbers=float, doc='总体Y的样本均值。')
@xw.arg('sigma_X', numbers=float, doc='总体X所来自分布的标准差，或样本的标准差。')
@xw.arg('sigma_Y', numbers=float, doc='总体Y所来自分布的标准差，或样本的标准差。')
@xw.arg('use_t', numbers=bool, doc='是否使用t分布计算置信区间，默认为True。')
@xw.ret(transpose=True)
def 总体间均值差单侧置信区间(alpha: float,
                             n_X: int, n_Y: int,
                             avg_X: float, avg_Y: float,
                             sigma_X: float, sigma_Y: float,
                             use_t: bool = True) -> np.ndarray:
    return 总体间均值差置信区间(alpha * 2, n_X, n_Y, avg_X, avg_Y, sigma_X, sigma_Y, use_t)


def _单个总体检验统计量(hypo_X: float, n_X: int, avg_X: float, sigma_X: float) -> dict[str, float]:
    ret = {'loc': hypo_X, 'scale': sigma_X / sqrt(n_X)}
    ret['statistics'] = (avg_X - hypo_X) / ret['scale']
    return ret


@xw.func(async_mode='threading')
@xw.arg('hypo_X', numbers=float, doc='假设均值。')
@xw.arg('n_X', numbers=int, doc='总体X的样本总数。')
@xw.arg('avg_X', numbers=float, doc='总体X的样本均值。')
@xw.arg('sigma_X', numbers=float, doc='总体X所来自分布的标准差，或样本的标准差。')
@xw.arg('alternative', doc='备择假设；gt：样本均值大于假设，lt：样本均值小于假设，'
                           'eq：样本均值等于假设，ned：样本均值不等于假设。')
@xw.arg('use_t', numbers=bool, doc='是否使用t分布计算P值，默认为True。')
def 单个总体P值(hypo_X: float,
                n_X: int, avg_X: float, sigma_X: float,
                alternative: str | int,
                use_t: bool = True) -> float:
    统计量 = _单个总体检验统计量(hypo_X, n_X, avg_X, sigma_X)
    loc, scale, statistics = 统计量['loc'], 统计量['scale'], 统计量['statistics']
    sf = partial(st.t.sf, df=n_X - 1) if use_t else st.norm.sf
    cdf = partial(st.t.cdf, df=n_X - 1) if use_t else st.norm.cdf
    match alternative:
        case 'gt' | 0:
            p_val = sf(statistics)
        case 'lt' | 1:
            p_val = cdf(statistics)
        case 'eq' | 2:
            p_val = 1 - 2 * sf(abs(statistics))
        case 'neq' | 3:
            p_val = 2 * sf(abs(statistics))
        case _:
            raise ValueError(f"不支持的{alternative=}。")
    return p_val


def _两个总体检验统计量(hypo_X_sub_Y: float,
                        n_X: int, n_Y: int,
                        avg_X: float, avg_Y: float,
                        sigma_X: float, sigma_Y: float,
                        use_t: bool = True
                        ) -> dict[str, float]:
    ret = {'loc': hypo_X_sub_Y}
    if not use_t:
        ret['scale'] = sqrt((sigma_X ** 2 / n_X) + (sigma_Y ** 2 / n_Y))
    else:
        df = n_X + n_Y - 2
        ret['scale'] = sqrt((((n_X - 1) * sigma_X ** 2) + ((n_Y - 1) * sigma_Y ** 2)) / df)
        ret['scale'] *= sqrt((1 / n_X) + (1 / n_Y))
    ret['statistics'] = (avg_X - avg_Y - hypo_X_sub_Y) / ret['scale']
    return ret


xw.func(async_mode='threading')


@xw.arg('hypo_X_sub_Y', numbers=float, doc='假设均值之差。')
@xw.arg('n_X', numbers=int, doc='总体X的样本总数。')
@xw.arg('n_Y', numbers=int, doc='总体Y的样本总数。')
@xw.arg('avg_X', numbers=float, doc='总体X的样本均值。')
@xw.arg('avg_Y', numbers=float, doc='总体Y的样本均值。')
@xw.arg('sigma_X', numbers=float, doc='总体X所来自分布的标准差，或样本的标准差。')
@xw.arg('sigma_Y', numbers=float, doc='总体Y所来自分布的标准差，或样本的标准差。')
@xw.arg('alternative', doc='备择假设；gt：样本均值差大于假设，lt：样本均值差小于假设，'
                           'eq：样本均值差等于假设，ned：样本均值差不等于假设。')
@xw.arg('use_t', numbers=bool, doc='是否使用t分布计算P值，默认为True。')
def 两个总体P值(hypo_X_sub_Y: float,
                n_X: int, n_Y: int,
                avg_X: float, avg_Y: float,
                sigma_X: float, sigma_Y: float,
                alternative: str | int,
                use_t: bool = True) -> float:
    统计量 = _两个总体检验统计量(hypo_X_sub_Y,
                                 n_X, n_Y,
                                 avg_X, avg_Y,
                                 sigma_X, sigma_Y,
                                 use_t=use_t)
    loc, scale, statistics = 统计量['loc'], 统计量['scale'], 统计量['statistics']
    sf = partial(st.t.sf, df=n_X + n_Y - 2) if use_t else st.norm.sf
    cdf = partial(st.t.cdf, df=n_X + n_Y - 2) if use_t else st.norm.cdf
    match alternative:
        case 'gt' | 0:
            p_val = sf(statistics)
        case 'lt' | 1:
            p_val = cdf(statistics)
        case 'eq' | 2:
            p_val = 1 - 2 * sf(abs(statistics))
        case 'neq' | 3:
            p_val = 2 * sf(abs(statistics))
        case _:
            raise ValueError(f"不支持的{alternative=}。")
    return p_val


xw.func(async_mode='threading')


@xw.arg('avg_X', numbers=float, doc='总体X的样本均值。')
@xw.arg('radius_X', numbers=float, doc='样本均值置信区间半径。')
@xw.arg('mul', numbers=float, doc='显示时乘以因子。')
@xw.arg('ndigits', numbers=int, doc='保留几位小数。')
def 格式化均值加减区间半径(avg_X: float, radius_X: float, mul: float = 100., ndigits: int = 2) -> str:
    return f'{avg_X * mul:.{ndigits}f}±{radius_X * mul:.{ndigits}f}'


if __name__ == '__main__':
    xw.serve()
