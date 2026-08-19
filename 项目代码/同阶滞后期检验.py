# -*- coding: utf-8 -*-


import re
import ast
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.api import VAR
import statsmodels.api as sm


def parse_chinese_date(s):
    """将“YYYY年M月D日 ...”标准化为 datetime"""
    s = str(s)
    s = re.sub(r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3', s)
    try:
        return parser.parse(s)
    except:
        return pd.NaT


def safe_parse_vector(cell):
    """安全解析 attribute_cooccurrence_matrix 的字符串为 list[float]。"""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    if isinstance(cell, str):
        try:
            val = ast.literal_eval(cell)
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)
        except Exception:
            return None
    return None

def pad_vector_to_len(vec, target_len):

    if vec is None:
        return [0.0]*target_len
    out = list(vec)[:target_len]
    if len(out) < target_len:
        out += [0.0]*(target_len - len(out))
    return [float(x) if (x is not None and x == x) else 0.0 for x in out]

def expand_matrix_equal_width(df, col="attribute_cooccurrence_matrix", target_len=None):

    parsed = df[col].apply(safe_parse_vector)
    if target_len is None:
        max_len = int(parsed.map(lambda v: len(v) if v is not None else 0).max())
    else:
        max_len = int(target_len)
    if max_len == 0:
        raise ValueError(f"{col} 解析失败：没有有效向量。")
    padded = parsed.map(lambda v: pad_vector_to_len(v, max_len))
    mat = pd.DataFrame(padded.tolist(), columns=[f"attr_{i}" for i in range(max_len)])
    out = pd.concat([df[['time_bin']], mat], axis=1)
    return out, max_len

def union_time_align(dfs):

    all_times = sorted(set().union(*[set(d['time_bin']) for d in dfs]))
    return pd.DataFrame({'time_bin': all_times})

def reindex_fill_zero(df, full_times, width):

    out = pd.merge(full_times, df, on='time_bin', how='left')
    attr_cols = [f'attr_{i}' for i in range(width)]
    out[attr_cols] = out[attr_cols].fillna(0.0)
    return out


def var_scan_both_criteria(src_series, tgt_series, max_lag=12):

    df2 = pd.DataFrame({
        'src': np.asarray(src_series, dtype=float),
        'tgt': np.asarray(tgt_series, dtype=float)
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    T = len(df2)
    eff_maxlag = int(min(max_lag, max(1, T // 3), max(1, T - 2)))
    if T < 10 or eff_maxlag < 1:
        return dict(best_lag_aic=np.nan, best_lag_bic=np.nan,
                    aic=np.nan, bic=np.nan, r2_aic=np.nan, r2_bic=np.nan,
                    n_eff_aic=0, n_eff_bic=0)


    if np.nanvar(df2['src'].values) < 1e-12 and np.nanvar(df2['tgt'].values) < 1e-12:
        return dict(best_lag_aic=np.nan, best_lag_bic=np.nan,
                    aic=np.nan, bic=np.nan, r2_aic=np.nan, r2_bic=np.nan,
                    n_eff_aic=0, n_eff_bic=0)

    lag_list, aics, bics, fitted_cache = [], [], [], {}

    for L in range(1, eff_maxlag + 1):
        try:
            res = VAR(df2).fit(L)
            lag_list.append(L)
            aics.append(float(res.aic))
            bics.append(float(res.bic))
            fitted_cache[L] = res
        except Exception:
            continue

    if len(lag_list) == 0:
        return dict(best_lag_aic=np.nan, best_lag_bic=np.nan,
                    aic=np.nan, bic=np.nan, r2_aic=np.nan, r2_bic=np.nan,
                    n_eff_aic=0, n_eff_bic=0)

    lags_arr = np.array(lag_list)
    aics_arr = np.array(aics)
    bics_arr = np.array(bics)

    idx_aic = int(np.argmin(aics_arr))
    idx_bic = int(np.argmin(bics_arr))
    best_lag_aic = int(lags_arr[idx_aic])
    best_lag_bic = int(lags_arr[idx_bic])

    def r2_target(results, L):
        fitted = results.fittedvalues
        y_hat = fitted['tgt'].values if 'tgt' in fitted.columns else fitted.iloc[:, 1].values
        y_true = df2['tgt'].iloc[L:].values
        n_eff = len(y_true)
        if n_eff > 1 and np.var(y_true) > 0:
            r2 = 1.0 - float(np.sum((y_true - y_hat)**2)) / float(np.sum((y_true - np.mean(y_true))**2))
        else:
            r2 = 0.0
        return r2, int(n_eff)

    res_aic = fitted_cache[best_lag_aic]
    r2_aic, n_eff_aic = r2_target(res_aic, best_lag_aic)
    aic_best = float(res_aic.aic)

    res_bic = fitted_cache[best_lag_bic]
    r2_bic, n_eff_bic = r2_target(res_bic, best_lag_bic)
    bic_best = float(res_bic.bic)

    return dict(best_lag_aic=best_lag_aic, best_lag_bic=best_lag_bic,
                aic=aic_best, bic=bic_best,
                r2_aic=r2_aic, r2_bic=r2_bic,
                n_eff_aic=n_eff_aic, n_eff_bic=n_eff_bic)


def build_ols_lag_design(src, tgt, L):

    s = pd.DataFrame({'src': np.asarray(src, dtype=float),
                      'tgt': np.asarray(tgt, dtype=float)}).replace([np.inf, -np.inf], np.nan)

    for j in range(1, L+1):
        s[f'tgt_l{j}'] = s['tgt'].shift(j)
        s[f'src_l{j}'] = s['src'].shift(j)
    y = s['tgt']
    X = s[[f'tgt_l{j}' for j in range(1, L+1)] + [f'src_l{j}' for j in range(1, L+1)]]
    data = pd.concat([y, X], axis=1).dropna()
    if data.empty:
        return None, None
    y2 = data['tgt'].values.astype(float)
    X2 = sm.add_constant(data.drop(columns=['tgt']).values.astype(float))
    return y2, X2

def ols_lag_scan_both_criteria(src_series, tgt_series, max_lag=12, min_n=12):

    src = np.asarray(src_series, dtype=float)
    tgt = np.asarray(tgt_series, dtype=float)
  
    m1 = np.isfinite(src); src = np.where(m1, src, 0.0)
    m2 = np.isfinite(tgt); tgt = np.where(m2, tgt, 0.0)

    T = len(src)
    eff_maxlag = int(min(max_lag, max(1, T // 3), max(1, T - 2)))
    if T < (min_n + 2) or eff_maxlag < 1:
        return dict(best_lag_aic=np.nan, best_lag_bic=np.nan,
                    aic=np.nan, bic=np.nan, r2_aic=np.nan, r2_bic=np.nan,
                    n_eff_aic=0, n_eff_bic=0)

    candidates = []
    for L in range(1, eff_maxlag+1):
        y, X = build_ols_lag_design(src, tgt, L)
        if y is None or len(y) < min_n:
            continue
        try:
            model = sm.OLS(y, X).fit()
            candidates.append({
                'L': L,
                'aic': float(model.aic),
                'bic': float(model.bic),
                'r2': float(model.rsquared),
                'n_eff': int(len(y)),
                'model': model
            })
        except Exception:
            continue

    if not candidates:
        return dict(best_lag_aic=np.nan, best_lag_bic=np.nan,
                    aic=np.nan, bic=np.nan, r2_aic=np.nan, r2_bic=np.nan,
                    n_eff_aic=0, n_eff_bic=0)


    dfc = pd.DataFrame([{k:v for k,v in c.items() if k!='model'} for c in candidates])

    idx_aic = int(dfc['aic'].idxmin())
    best_aic = candidates[idx_aic]

    idx_bic = int(dfc['bic'].idxmin())
    best_bic = candidates[idx_bic]

    return dict(best_lag_aic=int(best_aic['L']), best_lag_bic=int(best_bic['L']),
                aic=float(best_aic['aic']), bic=float(best_bic['bic']),
                r2_aic=float(best_aic['r2']), r2_bic=float(best_bic['r2']),
                n_eff_aic=int(best_aic['n_eff']), n_eff_bic=int(best_bic['n_eff']))


def run_path_VAR_both(df_src, df_tgt, src_label, tgt_label, max_lag=12):

    src_cols = [c for c in df_src.columns if c.startswith("attr_")]
    tgt_cols = [c for c in df_tgt.columns if c.startswith("attr_")]
    k = min(len(src_cols), len(tgt_cols))


    merged = pd.merge(
        df_src[['time_bin'] + [f'attr_{i}' for i in range(k)]],
        df_tgt[['time_bin'] + [f'attr_{i}' for i in range(k)]],
        on='time_bin', how='inner', suffixes=('__src','__tgt')
    ).sort_values('time_bin')

    rows = []
    pbar = tqdm(range(k), total=k, desc=f"{src_label}→{tgt_label} VAR(AIC&SC)")
    for i in pbar:
        src_i = merged[f'attr_{i}__src'] if f'attr_{i}__src' in merged else pd.Series(dtype=float)
        tgt_i = merged[f'attr_{i}__tgt'] if f'attr_{i}__tgt' in merged else pd.Series(dtype=float)


        m_var = var_scan_both_criteria(src_i, tgt_i, max_lag=max_lag)


        method_aic = 'VAR' if pd.notna(m_var['best_lag_aic']) else 'OLS-lag'
        method_bic = 'VAR' if pd.notna(m_var['best_lag_bic']) else 'OLS-lag'

        if method_aic == 'OLS-lag' or method_bic == 'OLS-lag':
            m_ols = ols_lag_scan_both_criteria(src_i, tgt_i, max_lag=max_lag)
        else:
            m_ols = None


        best_lag_aic = m_var['best_lag_aic'] if method_aic=='VAR' else m_ols['best_lag_aic']
        best_lag_bic = m_var['best_lag_bic'] if method_bic=='VAR' else m_ols['best_lag_bic']
        aic_best = m_var['aic'] if method_aic=='VAR' else m_ols['aic']
        bic_best = m_var['bic'] if method_bic=='VAR' else m_ols['bic']
        r2_aic  = m_var['r2_aic'] if method_aic=='VAR' else m_ols['r2_aic']
        r2_bic  = m_var['r2_bic'] if method_bic=='VAR' else m_ols['r2_bic']
        n_eff_aic = m_var['n_eff_aic'] if method_aic=='VAR' else m_ols['n_eff_aic']
        n_eff_bic = m_var['n_eff_bic'] if method_bic=='VAR' else m_ols['n_eff_bic']

        rows.append({
            '路径': f'{src_label}→{tgt_label}',
            '特征': f'attr_{i}',
            '方法_AIC': method_aic,
            '方法_SC':  method_bic,
            '最优阶_AIC(半小时)': best_lag_aic,
            '最优阶_SC(半小时)':  best_lag_bic,
            'AIC@最优': aic_best,
            'SC@最优':  bic_best,
            'R²_AIC(目标)': r2_aic,
            'R²_SC(目标)':  r2_bic,
            '有效样本_AIC': n_eff_aic,
            '有效样本_SC':  n_eff_bic,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":

    public_raw = pd.read_excel("111.xlsx")
    leader_raw = pd.read_excel("211.xlsx")
    media_raw  = pd.read_excel("311.xlsx")


    for df in (public_raw, leader_raw, media_raw):
        df['time_bin'] = df['time_bin'].apply(parse_chinese_date)
        df.sort_values('time_bin', inplace=True)


    all_parsed = []
    for df in (public_raw, leader_raw, media_raw):
        all_parsed += list(df['attribute_cooccurrence_matrix'].apply(safe_parse_vector))
    global_max_len = int(max((len(v) if v is not None else 0) for v in all_parsed))
    if global_max_len == 0:
        raise ValueError("三源的 attribute_cooccurrence_matrix 均无法解析，请检查数据。")

    public_mat, _ = expand_matrix_equal_width(public_raw, "attribute_cooccurrence_matrix", target_len=global_max_len)
    leader_mat, _ = expand_matrix_equal_width(leader_raw, "attribute_cooccurrence_matrix", target_len=global_max_len)
    media_mat,  _ = expand_matrix_equal_width(media_raw,  "attribute_cooccurrence_matrix", target_len=global_max_len)


    full_times = union_time_align([public_mat, leader_mat, media_mat])
    public_aln = reindex_fill_zero(public_mat, full_times, global_max_len)
    leader_aln = reindex_fill_zero(leader_mat, full_times, global_max_len)
    media_aln  = reindex_fill_zero(media_mat,  full_times, global_max_len)


    max_lag = 12


    res_PL = run_path_VAR_both(public_aln, leader_aln, '公众',   '意见领袖', max_lag=max_lag)
    res_PM = run_path_VAR_both(public_aln, media_aln,  '公众',   '媒体',     max_lag=max_lag)
    res_LM = run_path_VAR_both(leader_aln, media_aln,  '意见领袖','媒体',    max_lag=max_lag)

    out = pd.concat([res_PL, res_PM, res_LM], ignore_index=True)
    out.to_excel("跨网络VAR滞后分析_AIC与SC.xlsx", index=False)

    print("✅ 完成：跨网络VAR滞后分析_AIC与SC.xlsx 已生成（含 AIC/SC 最优阶、R²、并标注使用 VAR 或 OLS-lag）")
