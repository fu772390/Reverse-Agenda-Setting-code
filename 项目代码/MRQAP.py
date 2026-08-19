# -*- coding: utf-8 -*-


import ast
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import norm

np.random.seed(42)

# ===================== 参数区 =====================
COLUMN_NAME = "attribute_cooccurrence_matrix"
SHEET_NAME = "主题属性矩阵"
TIME_COL = "time_bin"
TIME_FREQ = "30min"


FILL_MISSING_WINDOWS_WITH_ZERO = True


AGG_METHOD = "mean"

NORMALIZE_BY_TOTAL_EDGE_WEIGHT = True


def restore_matrix(flat_value):
    """把 Excel 中保存的扁平矩阵还原为 N×N 方阵。"""
    if isinstance(flat_value, str):
        arr = np.array(ast.literal_eval(flat_value), dtype=float)
    else:
        arr = np.array(flat_value, dtype=float)
    n = int(np.sqrt(len(arr)))
    if n * n != len(arr):
        raise ValueError(f"矩阵长度 {len(arr)} 不能还原为方阵。")
    return arr.reshape(n, n)


def load_all_matrices_with_time(file_path, column_name=COLUMN_NAME, sheet_name=SHEET_NAME):
 
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if column_name not in df.columns:
        raise KeyError(f"文件 {file_path} 中未找到列 {column_name}，实际列名：{list(df.columns)}")
    if TIME_COL not in df.columns:
        raise KeyError(f"文件 {file_path} 中未找到时间列 {TIME_COL}，实际列名：{list(df.columns)}")

    df = df[[TIME_COL, column_name]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, column_name]).sort_values(TIME_COL).reset_index(drop=True)
    df["matrix"] = df[column_name].apply(restore_matrix)

    if df.empty:
        raise ValueError(f"文件 {file_path} 没有可用矩阵。")

    return df[[TIME_COL, "matrix"]]


def make_continuous_matrix_series(df, freq=TIME_FREQ, fill_zero=True):

    df = df.copy().sort_values(TIME_COL)
    n = df["matrix"].iloc[0].shape[0]
    zero_matrix = np.zeros((n, n), dtype=float)


    grouped = df.groupby(TIME_COL)["matrix"].apply(lambda mats: np.mean(np.stack(list(mats)), axis=0))
    grouped = grouped.sort_index()

    if not fill_zero:
        return list(grouped.values), grouped.index

    full_index = pd.date_range(grouped.index.min().floor(freq), grouped.index.max().ceil(freq), freq=freq)
    matrices = []
    for t in full_index:
        if t in grouped.index:
            matrices.append(grouped.loc[t])
        else:
            matrices.append(zero_matrix.copy())
    return matrices, full_index


def aggregate_matrices(matrices, method=AGG_METHOD, normalize=NORMALIZE_BY_TOTAL_EDGE_WEIGHT):

    stack = np.stack(matrices)
    if method == "mean":
        A = stack.mean(axis=0)
    elif method == "sum":
        A = stack.sum(axis=0)
    else:
        raise ValueError("AGG_METHOD 只能是 'mean' 或 'sum'。")

    if normalize:
        total = np.nansum(A)
        if total > 0:
            A = A / total
    return A


def load_aggregate_matrix_from_excel(file_path):
 
    df = load_all_matrices_with_time(file_path)
    matrices, time_index = make_continuous_matrix_series(
        df,
        freq=TIME_FREQ,
        fill_zero=FILL_MISSING_WINDOWS_WITH_ZERO
    )
    A = aggregate_matrices(matrices)
    return A, {
        "file": file_path,
        "start_time": str(time_index.min()),
        "end_time": str(time_index.max()),
        "window_n": len(time_index),
        "observed_matrix_n": len(df),
        "filled_zero_window_n": len(time_index) - len(df) if FILL_MISSING_WINDOWS_WITH_ZERO else 0,
        "matrix_shape": A.shape,
        "agg_method": AGG_METHOD,
        "normalize_total_edge_weight": NORMALIZE_BY_TOTAL_EDGE_WEIGHT,
    }


def get_lower_triangular(M):

    return M[np.tril_indices_from(M, k=-1)]


def standardized_linear(y_vec, X_mat):
    X_mat = np.asarray(X_mat)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)
    Xs = StandardScaler().fit_transform(X_mat)
    ys = StandardScaler().fit_transform(np.asarray(y_vec).reshape(-1, 1)).ravel()
    return LinearRegression().fit(Xs, ys).coef_


def mrqap_multi(y_matrix, x_matrices, n_permutations=1000):
    y_vec = get_lower_triangular(y_matrix)
    X_vec = np.column_stack([get_lower_triangular(x) for x in x_matrices])

    Xs = StandardScaler().fit_transform(X_vec)
    ys = StandardScaler().fit_transform(y_vec.reshape(-1, 1)).ravel()

    model = LinearRegression().fit(Xs, ys)
    betas_std = model.coef_
    r_squared = model.score(Xs, ys)

    betas_perm = np.zeros((n_permutations, Xs.shape[1]))
    for i in tqdm(range(n_permutations), desc="MRQAP置换检验"):
        ys_perm = ys[np.random.permutation(len(ys))]
        betas_perm[i, :] = LinearRegression().fit(Xs, ys_perm).coef_

    p_values = [
        (np.sum(np.abs(betas_perm[:, j]) >= np.abs(betas_std[j])) + 1) / (n_permutations + 1)
        for j in range(Xs.shape[1])
    ]
    return betas_std, p_values, r_squared


def bca_ci(data, observed, alpha=0.05):
    data = np.sort(np.asarray(data))
    if data.size == 0:
        return np.nan, np.nan
    prop_less = np.sum(data < observed) / len(data)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)
    z_alpha = norm.ppf([alpha / 2, 1 - alpha / 2])
    pct = norm.cdf(2 * z0 + z_alpha) * 100
    pct = np.clip(pct, 0, 100)
    return float(np.percentile(data, pct[0])), float(np.percentile(data, pct[1]))


def mediation_mrqap(x_matrix, m_matrix, y_matrix, n_permutations=5000, ci=0.95):
    idx = np.tril_indices_from(x_matrix, k=-1)
    x = x_matrix[idx]
    m = m_matrix[idx]
    y = y_matrix[idx]

    a = standardized_linear(m, x.reshape(-1, 1))[0]
    b_all = standardized_linear(y, np.column_stack([x, m]))
    c_prime = b_all[0]
    b = b_all[1]
    indirect = a * b

    indirect_perm = []
    for _ in tqdm(range(n_permutations), desc="中介置换检验"):
        x_perm = np.random.permutation(x)
        a_p = standardized_linear(m, x_perm.reshape(-1, 1))[0]
        b_p = standardized_linear(y, np.column_stack([x_perm, m]))[1]
        indirect_perm.append(a_p * b_p)

    indirect_perm = np.asarray(indirect_perm)
    p_value = (np.sum(np.abs(indirect_perm) >= np.abs(indirect)) + 1) / (n_permutations + 1)
    lower, upper = bca_ci(indirect_perm, indirect, alpha=1 - ci)
    return {
        "a_公众→意见领袖": a,
        "b_意见领袖→媒体": b,
        "c_prime_公众→媒体_控制意见领袖": c_prime,
        "indirect_ab": indirect,
        "p_value_indirect": p_value,
        f"CI_lower_{int(ci*100)}%": lower,
        f"CI_upper_{int(ci*100)}%": upper,
    }


def run_all():

    files = [
        "1.xlsx",  # 公众 t1
        "2.xlsx",  # 意见领袖 t1
        "3.xlsx",  # 媒体 t1
        "4.xlsx",  # 公众 t2
        "5.xlsx",  # 意见领袖 t2
        "6.xlsx",  # 媒体 t2
    ]

    matrices = []
    diagnostics = []
    for f in files:
        A, info = load_aggregate_matrix_from_excel(f)
        matrices.append(A)
        diagnostics.append(info)
    pd.DataFrame(diagnostics).to_excel("MRQAP_矩阵汇总诊断.xlsx", index=False)

 
    M = matrices

    models = [
        ("模型1_时点1_公众+意见领袖→媒体", M[2], [M[0], M[1]], ["公众", "意见领袖"]),
        ("模型2_时点2_公众+意见领袖→媒体", M[5], [M[3], M[4]], ["公众", "意见领袖"]),
        ("模型3_公众_t1→公众_t2", M[3], [M[0]], ["公众_t1"]),
        ("模型3_意见领袖_t1→意见领袖_t2", M[4], [M[1]], ["意见领袖_t1"]),
        ("模型3_媒体_t1→媒体_t2", M[5], [M[2]], ["媒体_t1"]),
        ("模型4_公众→意见领袖_控制媒体_时点1", M[1], [M[0], M[2]], ["公众", "媒体"]),
        ("模型4_公众→意见领袖_控制媒体_时点2", M[4], [M[3], M[5]], ["公众", "媒体"]),
        ("模型5_公众→意见领袖_不控媒体_时点1", M[1], [M[0]], ["公众"]),
        ("模型5_公众→意见领袖_不控媒体_时点2", M[4], [M[3]], ["公众"]),
    ]

    reg_rows = []
    for label, y, xs, names in models:
        betas, pvals, r2 = mrqap_multi(y, xs, n_permutations=1000)
        for name, beta, p in zip(names, betas, pvals):
            reg_rows.append({
                "模型": label,
                "自变量": name,
                "标准化回归系数Beta": float(beta),
                "p值": float(p),
                "R²": float(r2),
            })
    pd.DataFrame(reg_rows).to_excel("MRQAP_全样本汇总矩阵_多模型回归结果.xlsx", index=False)

    # 中介：时点1与时点2
    med1 = mediation_mrqap(M[0], M[1], M[2], n_permutations=5000)
    med1["timepoint"] = "时点1"
    med2 = mediation_mrqap(M[3], M[4], M[5], n_permutations=5000)
    med2["timepoint"] = "时点2"
    pd.DataFrame([med1, med2]).to_excel("MRQAP_全样本汇总矩阵_中介分析结果.xlsx", index=False)

    print("完成。已输出：")
    print("- MRQAP_矩阵汇总诊断.xlsx")
    print("- MRQAP_全样本汇总矩阵_多模型回归结果.xlsx")
    print("- MRQAP_全样本汇总矩阵_中介分析结果.xlsx")


if __name__ == "__main__":
    run_all()
