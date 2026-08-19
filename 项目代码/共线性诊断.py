import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm

np.random.seed(42)


def load_matrix_from_excel(file_path, column_name="attribute_cooccurrence_matrix"):
    df = pd.read_excel(file_path)
    if column_name not in df.columns:
        raise KeyError(f"文件 {file_path} 中未找到列 '{column_name}'，实际列名有：{list(df.columns)}")
    def restore_matrix(flat_list):
        arr = np.array(eval(flat_list)) if isinstance(flat_list, str) else np.array(flat_list)
        n = int(np.sqrt(len(arr)))
        assert n * n == len(arr), f"矩阵长度 {len(arr)} 不能还原为方阵"
        return arr.reshape(n, n)
    return df[column_name].apply(restore_matrix).tolist()

def get_lower_triangular(matrix):
    return matrix[np.tril_indices_from(matrix, k=-1)]

def standardized_linear(y_vec, X_mat):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_mat)
    y_scaled = scaler_y.fit_transform(y_vec.reshape(-1, 1)).flatten()
    model = LinearRegression().fit(X_scaled, y_scaled)
    return model.coef_


def collinearity_diagnostics(x_matrices, x_names=None, eps=1e-12):
    
    if len(x_matrices) == 0:
        raise ValueError("x_matrices 不能为空")

    if x_names is None:
        x_names = [f"X{i + 1}" for i in range(len(x_matrices))]
    if len(x_names) != len(x_matrices):
        raise ValueError("x_names 的长度必须与 x_matrices 一致")

    X_vec = np.column_stack(
        [get_lower_triangular(np.asarray(x, dtype=float)) for x in x_matrices]
    )
    if not np.all(np.isfinite(X_vec)):
        raise ValueError("共线性诊断发现 NaN 或无穷值，请先清理输入矩阵")

    n_obs, n_pred = X_vec.shape
    stds = np.std(X_vec, axis=0, ddof=1)
    constant_mask = stds <= eps

 
    pair_rows = []
    if n_pred >= 2:
        corr_matrix = np.corrcoef(X_vec, rowvar=False)
        for i in range(n_pred):
            for j in range(i + 1, n_pred):
                r = float(corr_matrix[i, j])
                abs_r = abs(r)
                if not np.isfinite(r):
                    corr_flag = "无法计算（可能含常量变量）"
                elif abs_r >= 0.90:
                    corr_flag = "严重相关"
                elif abs_r >= 0.80:
                    corr_flag = "较高相关"
                else:
                    corr_flag = "未见明显问题"
                pair_rows.append({
                    "变量1": x_names[i],
                    "变量2": x_names[j],
                    "Pearson_r": r,
                    "|r|": abs_r,
                    "相关诊断": corr_flag
                })


    vif_rows = []
    for j, name in enumerate(x_names):
        if constant_mask[j]:
            r2_j = 1.0
            tolerance = 0.0
            vif = np.inf
            vif_flag = "常量变量，无法进入回归"
        elif n_pred == 1:
            r2_j = 0.0
            tolerance = 1.0
            vif = 1.0
            vif_flag = "单一自变量，无共线性"
        else:
            y_j = X_vec[:, j]
            X_other = np.delete(X_vec, j, axis=1)

            r2_j = float(LinearRegression().fit(X_other, y_j).score(X_other, y_j))
            tolerance = max(0.0, 1.0 - r2_j)
            vif = np.inf if tolerance <= eps else 1.0 / tolerance

            if not np.isfinite(vif) or vif >= 10:
                vif_flag = "严重共线性"
            elif vif >= 5:
                vif_flag = "需要关注"
            else:
                vif_flag = "未见明显问题"

        vif_rows.append({
            "自变量": name,
            "被其余自变量解释的R²": r2_j,
            "容忍度": tolerance,
            "VIF": vif,
            "变量标准差": float(stds[j]),
            "是否常量变量": bool(constant_mask[j]),
            "VIF诊断": vif_flag
        })

  
    valid_mask = ~constant_mask
    X_valid = X_vec[:, valid_mask]
    valid_names = [x_names[i] for i in range(n_pred) if valid_mask[i]]

    if X_valid.shape[1] == 0:
        rank = 0
        condition_index = np.inf
    elif X_valid.shape[1] == 1:
        rank = 1
        condition_index = 1.0
    else:
        X_scaled = StandardScaler().fit_transform(X_valid)
        singular_values = np.linalg.svd(X_scaled, compute_uv=False)
        rank = int(np.linalg.matrix_rank(X_scaled))
        if singular_values[-1] <= eps:
            condition_index = np.inf
        else:
            condition_index = float(singular_values[0] / singular_values[-1])

    finite_vifs = [row["VIF"] for row in vif_rows if np.isfinite(row["VIF"])]
    max_vif = max(finite_vifs) if finite_vifs else np.inf
    max_abs_corr = max((row["|r|"] for row in pair_rows), default=np.nan)

    if (
        np.any(constant_mask)
        or not np.isfinite(condition_index)
        or condition_index >= 30
        or max_vif >= 10
        or (np.isfinite(max_abs_corr) and max_abs_corr >= 0.90)
        or rank < len(valid_names)
    ):
        overall_flag = "严重共线性，需要谨慎解释系数"
    elif (
        condition_index >= 15
        or max_vif >= 5
        or (np.isfinite(max_abs_corr) and max_abs_corr >= 0.80)
    ):
        overall_flag = "存在较高共线性，建议报告诊断"
    else:
        overall_flag = "未见明显共线性"

    model_row = {
        "下三角观测数": int(n_obs),
        "自变量数": int(n_pred),
        "非常量自变量数": int(len(valid_names)),
        "设计矩阵秩": int(rank),
        "条件指数": condition_index,
        "最大VIF": max_vif,
        "最大绝对相关": max_abs_corr,
        "整体诊断": overall_flag
    }

    return (
        pd.DataFrame(vif_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame([model_row])
    )


def bca_ci(data, observed, alpha=0.05):
    data = np.sort(np.asarray(data))
    if data.size == 0:
        return (np.nan, np.nan)
    prop_less = np.sum(data < observed) / len(data)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)
    z_alpha = norm.ppf([alpha/2, 1 - alpha/2])
    pct = norm.cdf(2*z0 + z_alpha) * 100
    pct = np.clip(pct, 0, 100)
    return float(np.percentile(data, pct[0])), float(np.percentile(data, pct[1]))


def mrqap_multi_parallel(y_matrix, x_matrices, n_permutations=500):
    y_vec = get_lower_triangular(y_matrix)
    X_vec = np.column_stack([get_lower_triangular(x) for x in x_matrices])
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_vec)
    y_scaled = scaler_y.fit_transform(y_vec.reshape(-1, 1)).flatten()
    model = LinearRegression().fit(X_scaled, y_scaled)
    betas_std = model.coef_
    r_squared = model.score(X_scaled, y_scaled)

    perm_betas = np.zeros((n_permutations, len(x_matrices)))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(permute, X_scaled, y_scaled) for _ in range(n_permutations)]
        for i, future in enumerate(tqdm(futures, desc="置换检验", total=n_permutations)):
            perm_betas[i, :] = future.result()

    p_values = [(np.sum(np.abs(perm_betas[:, i]) >= np.abs(betas_std[i])) + 1) / (n_permutations + 1)
                for i in range(len(x_matrices))]
    return betas_std, p_values, r_squared

def permute(X_scaled, y_scaled):
    perm = np.random.permutation(len(y_scaled))
    y_perm = y_scaled[perm]
    model = LinearRegression().fit(X_scaled, y_perm)
    return model.coef_


def sobel_p_value(a, b, se_a, se_b):
    if not np.isfinite(se_a) or not np.isfinite(se_b) or se_a <= 0 or se_b <= 0:
        return 1.0
    se_indirect = np.sqrt((b*b) * (se_a*se_a) + (a*a) * (se_b*se_b))
    if not np.isfinite(se_indirect) or se_indirect == 0:
        return 1.0
    z = (a * b) / se_indirect
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(p) if np.isfinite(p) else 1.0


def prodclin_ci(a_hat, b_hat, a_samples, b_samples, ci=0.95, n_mc=200000):
    a_samples = np.asarray(a_samples); b_samples = np.asarray(b_samples)
    sa = np.std(a_samples, ddof=1)
    sb = np.std(b_samples, ddof=1)
    rho = np.corrcoef(a_samples, b_samples)[0,1] if a_samples.size > 1 else 0.0
    if not np.isfinite(rho):
        rho = 0.0

    mean = np.array([a_hat, b_hat], dtype=float)
    cov = np.array([[sa*sa, rho*sa*sb],
                    [rho*sa*sb, sb*sb]], dtype=float)

    try:
        ab = np.random.multivariate_normal(mean, cov, size=n_mc)
    except (np.linalg.LinAlgError, ValueError):
        eps = 1e-9
        cov = cov + np.eye(2)*eps
        ab = np.random.multivariate_normal(mean, cov, size=n_mc)

    prod = ab[:,0] * ab[:,1]
    alpha = 1 - ci
    low, up = np.percentile(prod, [alpha/2*100, (1-alpha/2)*100])
    return float(low), float(up)


def delta_se_ab(a_hat, b_hat, a_samples, b_samples):
    a_samples = np.asarray(a_samples); b_samples = np.asarray(b_samples)
    Va = np.var(a_samples, ddof=1)
    Vb = np.var(b_samples, ddof=1)
    Cov = np.cov(a_samples, b_samples, ddof=1)[0,1] if a_samples.size > 1 else 0.0
    se = np.sqrt((b_hat**2)*Va + (a_hat**2)*Vb + 2*a_hat*b_hat*Cov)
    return float(se), float(Va), float(Vb), float(Cov)

def bootstrap_t_ci(ab_hat, a_samples, b_samples, ci=0.95):
    a_samples = np.asarray(a_samples); b_samples = np.asarray(b_samples)
    if a_samples.size == 0 or b_samples.size == 0:
        return (np.nan, np.nan)
    Va = np.var(a_samples, ddof=1)
    Vb = np.var(b_samples, ddof=1)
    Cov = np.cov(a_samples, b_samples, ddof=1)[0,1] if a_samples.size > 1 else 0.0

    ab_b = a_samples * b_samples
    se_b = np.sqrt((b_samples**2)*Va + (a_samples**2)*Vb + 2*a_samples*b_samples*Cov)
    valid = np.isfinite(se_b) & (se_b > 0)
    if not np.any(valid):
        return (np.nan, np.nan)

    t_star = (ab_b[valid] - ab_hat) / se_b[valid]
    alpha = 1 - ci
    q_lo, q_hi = np.percentile(t_star, [ (1-alpha/2)*100, (alpha/2)*100 ])  # 注意反推
    # 用点估计处的 se_hat
    se_hat = np.sqrt((np.mean(b_samples)**2)*Va + (np.mean(a_samples)**2)*Vb + 2*np.mean(a_samples)*np.mean(b_samples)*Cov)
    if not np.isfinite(se_hat) or se_hat <= 0:
        return (np.nan, np.nan)
    ci_lo = ab_hat - q_lo * se_hat
    ci_hi = ab_hat - q_hi * se_hat
    lo, hi = (min(ci_lo, ci_hi), max(ci_lo, ci_hi))
    return float(lo), float(hi)


def rademacher_weights(n):
    return np.where(np.random.rand(n) < 0.5, -1.0, 1.0)

def wild_bootstrap_ci_ab(x_mat, m_mat, y_mat, n_bootstrap=2000, ci=0.95, scheme="rademacher"):
    n = x_mat.shape[0]
    idx = np.tril_indices_from(x_mat, k=-1)
    x_vec = x_mat[idx]; m_vec = m_mat[idx]; y_vec = y_mat[idx]

    ab_wild = []
    for _ in tqdm(range(n_bootstrap), desc="WildBoot", leave=False):
        if scheme == "rademacher":
            w = rademacher_weights(n)
        else:
            # Mammen 权重（备用）
            p = (np.sqrt(5)+1)/(2*np.sqrt(5))
            u = np.random.rand(n)
            w = np.where(u < p, (1-np.sqrt(5))/2, (1+np.sqrt(5))/2)

        W = np.outer(w, w)
        w_vec = W[idx]

        xw = x_vec * w_vec
        mw = m_vec * w_vec
        yw = y_vec * w_vec

        a_b = standardized_linear(mw, xw.reshape(-1,1))[0]
        b_b = standardized_linear(yw, np.column_stack([xw, mw]))[1]
        ab_wild.append(a_b * b_b)

    ab_wild = np.asarray(ab_wild)
    alpha = 1 - ci
    lo, hi = np.percentile(ab_wild, [alpha/2*100, (1-alpha/2)*100])
    return float(lo), float(hi)


def mediation_all_ci(x_mat, m_mat, y_mat, n_bootstrap=5000, ci=0.95, n_mc=200000, n_wild=2000):
    idx = np.tril_indices_from(x_mat, k=-1)
    x_vec = x_mat[idx]
    m_vec = m_mat[idx]
    y_vec = y_mat[idx]


    a_hat = standardized_linear(m_vec, x_vec.reshape(-1, 1))[0]
    b_all = standardized_linear(y_vec, np.column_stack([x_vec, m_vec]))
    b_hat = b_all[1]
    c_prime_hat = b_all[0]
    ab_hat = a_hat * b_hat

   
    ab_dist = []
    a_samples, b_samples = [], []
    for _ in tqdm(range(n_bootstrap), desc="中介Boot", leave=False):
        boot_idx = np.random.choice(len(x_vec), size=len(x_vec), replace=True)
        x_b = x_vec[boot_idx]
        m_b = m_vec[boot_idx]
        y_b = y_vec[boot_idx]
        a_b = standardized_linear(m_b, x_b.reshape(-1, 1))[0]
        b_b_all = standardized_linear(y_b, np.column_stack([x_b, m_b]))
        b_b = b_b_all[1]
        a_samples.append(a_b); b_samples.append(b_b)
        ab_dist.append(a_b * b_b)

    a_samples = np.asarray(a_samples); b_samples = np.asarray(b_samples); ab_dist = np.asarray(ab_dist)


    alpha = 1 - ci

    ci_pct = tuple(np.percentile(ab_dist, [alpha/2*100, (1-alpha/2)*100])) if ab_dist.size else (np.nan, np.nan)

    ci_bca = bca_ci(ab_dist, np.median(ab_dist) if ab_dist.size else ab_hat, alpha=1-ci)

    ci_prodclin = prodclin_ci(a_hat, b_hat, a_samples, b_samples, ci=ci, n_mc=n_mc) if a_samples.size and b_samples.size else (np.nan, np.nan)

    ci_bt = bootstrap_t_ci(ab_hat, a_samples, b_samples, ci=ci) if a_samples.size and b_samples.size else (np.nan, np.nan)
  
    ci_wild = wild_bootstrap_ci_ab(x_mat, m_mat, y_mat, n_bootstrap=n_wild, ci=ci, scheme="rademacher")


    se_a = np.std(a_samples, ddof=1) if a_samples.size else np.nan
    se_b = np.std(b_samples, ddof=1) if b_samples.size else np.nan
    sobel_p = sobel_p_value(a_hat, b_hat, se_a, se_b)

    return {
        "a (raw)": float(a_hat),
        "b (raw)": float(b_hat),
        "c′ (raw)": float(c_prime_hat),
        "ab (raw, a*b)": float(ab_hat),
        "ab_boot_median": float(np.median(ab_dist)) if ab_dist.size else np.nan,
        "Sobel_p": sobel_p,
        f"CI_percentile_{int(ci*100)}%": (float(ci_pct[0]), float(ci_pct[1])),
        f"CI_BCa_{int(ci*100)}%": (float(ci_bca[0]), float(ci_bca[1])),
        f"CI_PRODCLIN_{int(ci*100)}%": (float(ci_prodclin[0]), float(ci_prodclin[1])),
        f"CI_BootstrapT_{int(ci*100)}%": (float(ci_bt[0]), float(ci_bt[1])),
        f"CI_WildRademacher_{int(ci*100)}%": (float(ci_wild[0]), float(ci_wild[1]))
    }


def run_mrqap_all():
    files = ["1.xlsx", "2.xlsx", "3.xlsx", "4.xlsx", "5.xlsx", "6.xlsx"]
    matrices = [load_matrix_from_excel(f)[0] for f in files]

    final_results = []
    vif_results = []
    pair_corr_results = []
    model_diag_results = []

    
    models = [
        (
            "模型1_时点1_公众+意见领袖对媒体",
            matrices[2],
            [matrices[0], matrices[1]],
            ["公众议程网络", "意见领袖议程网络"]
        ),
        (
            "模型2_时点2_公众+意见领袖对媒体",
            matrices[5],
            [matrices[3], matrices[4]],
            ["公众议程网络", "意见领袖议程网络"]
        ),
        (
            "模型3_时点1公众→时点2公众",
            matrices[3],
            [matrices[0]],
            ["时点1公众议程网络"]
        ),
        (
            "模型3_时点1意见领袖→时点2意见领袖",
            matrices[4],
            [matrices[1]],
            ["时点1意见领袖议程网络"]
        ),
        (
            "模型3_时点1媒体→时点2媒体",
            matrices[5],
            [matrices[2]],
            ["时点1媒体议程网络"]
        ),
        (
            "模型4_公众→意见领袖_控制媒体_时点1",
            matrices[1],
            [matrices[0], matrices[2]],
            ["公众议程网络", "媒体议程网络"]
        ),
        (
            "模型4_公众→意见领袖_控制媒体_时点2",
            matrices[4],
            [matrices[3], matrices[5]],
            ["公众议程网络", "媒体议程网络"]
        ),
        (
            "模型5_公众→意见领袖_不控媒体_时点1",
            matrices[1],
            [matrices[0]],
            ["公众议程网络"]
        ),
        (
            "模型5_公众→意见领袖_不控媒体_时点2",
            matrices[4],
            [matrices[3]],
            ["公众议程网络"]
        )
    ]

    for label, y, x_list, x_names in models:
  
        betas, pvals, r2 = mrqap_multi_parallel(y, x_list)
        for name, beta, p in zip(x_names, betas, pvals):
            final_results.append({
                "模型": label,
                "自变量": name,
                "标准化回归系数": float(beta),
                "p值": float(p),
                "R²": float(r2)
            })

   
        vif_df, pair_df, model_df = collinearity_diagnostics(
            x_matrices=x_list,
            x_names=x_names
        )

        vif_df.insert(0, "模型", label)
        vif_results.append(vif_df)

        if not pair_df.empty:
            pair_df.insert(0, "模型", label)
            pair_corr_results.append(pair_df)

        model_df.insert(0, "模型", label)
        model_diag_results.append(model_df)

        diagnosis = model_df.loc[0, "整体诊断"]
        print(
            f"[共线性] {label}: {diagnosis}; "
            f"最大VIF={model_df.loc[0, '最大VIF']:.3f}, "
            f"条件指数={model_df.loc[0, '条件指数']:.3f}"
        )


    pd.DataFrame(final_results).to_excel(
        "MRQAP_多模型回归结果.xlsx",
        index=False
    )


    vif_all = pd.concat(vif_results, ignore_index=True)
    pair_all = (
        pd.concat(pair_corr_results, ignore_index=True)
        if pair_corr_results
        else pd.DataFrame(
            columns=["模型", "变量1", "变量2", "Pearson_r", "|r|", "相关诊断"]
        )
    )
    model_diag_all = pd.concat(model_diag_results, ignore_index=True)

    with pd.ExcelWriter("MRQAP_共线性诊断.xlsx") as writer:
        vif_all.to_excel(writer, sheet_name="VIF_变量层面", index=False)
        pair_all.to_excel(writer, sheet_name="相关_变量对", index=False)
        model_diag_all.to_excel(writer, sheet_name="模型整体诊断", index=False)


    mediation_results = [
        mediation_all_ci(
            matrices[0], matrices[1], matrices[2],
            n_bootstrap=5000, ci=0.95, n_mc=200000, n_wild=2000
        ),
        mediation_all_ci(
            matrices[3], matrices[4], matrices[5],
            n_bootstrap=5000, ci=0.95, n_mc=200000, n_wild=2000
        )
    ]
    pd.DataFrame(
        mediation_results,
        index=["时点1", "时点2"]
    ).to_excel("MRQAP_中介分析结果_全部区间.xlsx")

    print("✅ 多模型回归结果已保存为：MRQAP_多模型回归结果.xlsx")
    print("✅ 共线性诊断已保存为：MRQAP_共线性诊断.xlsx")
    print("✅ 中介分析结果（含 Percentile / BCa / PRODCLIN / Bootstrap-T / Wild）已保存为：MRQAP_中介分析结果_全部区间.xlsx")


if __name__ == "__main__":
    run_mrqap_all()
