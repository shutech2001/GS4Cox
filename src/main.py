import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
from polyagamma import random_polyagamma  # type: ignore

np.random.seed(42)


def simulate_cox_data(n, beta_true, seed=42):
    """
    Cox回帰用のデータシミュレーション
    """
    np.random.seed(seed)
    p = len(beta_true)
    X = np.random.randn(n, p)
    linpred = X.dot(beta_true)
    T = np.random.exponential(scale=1/np.exp(linpred))
    C = np.random.exponential(scale=1.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)
    return X, time, event


def build_risk_sets(time, event):
    """
    各イベント時刻 t に対して，リスク集合 R(t) を作成する．
    イベント発生した被験者を先頭に配置し，残りはソートして連結する．
    """
    event_times = np.sort(np.unique(time[event == 1]))
    risk_sets = {}
    for t in event_times:
        idx = np.where(time >= t)[0]
        event_idx = np.where((time == t) & (event == 1))[0]
        if len(event_idx) > 0:
            non_event_idx = np.setdiff1d(idx, event_idx, assume_unique=True)
            new_order = np.concatenate([event_idx, np.sort(non_event_idx)])
        else:
            new_order = np.sort(idx)
        risk_sets[t] = new_order
    return event_times, risk_sets


def cumulative_logsumexp(s):
    """
    s: 1次元配列 (各被験者のスコア)
    L[i] = logsumexp(s[i:]) を計算する．
    """
    L = np.empty_like(s)
    L[-1] = s[-1]
    for i in range(len(s)-2, -1, -1):
        L[i] = np.logaddexp(s[i], L[i+1])
    return L


def cumulative_weighted_X(X_sub, exp_s):
    """
    X_sub: リスク集合内の特徴行列 (shape: (K, p))
    exp_s: 各被験者の exp(s) 値 (shape: (K,))

    Returns:
      cum_exp: 各位置 i から末尾までの exp(s) の和 (shape: (K,))
      cum_X: 各位置 i から末尾までの exp(s)*X の和 (shape: (K, p))
    """
    K = len(exp_s)
    cum_exp = np.empty_like(exp_s)
    cum_X = np.empty((K, X_sub.shape[1]))
    cum_exp[-1] = exp_s[-1]
    cum_X[-1, :] = X_sub[-1, :] * exp_s[-1]
    for i in range(K-2, -1, -1):
        cum_exp[i] = cum_exp[i+1] + exp_s[i]
        cum_X[i, :] = cum_X[i+1, :] + X_sub[i, :] * exp_s[i]
    return cum_exp, cum_X


def compute_all_eta_gradients(X, beta, risk_sets, event_times):
    """
    各イベント時刻 t において，リスク集合 R(t) 内の各順位 k (k = 0,...,K-2) について，
    以下の値を計算し，すべてのイベント時刻の結果を連結して返す．
      - eta: s[k] - logsumexp(s[k:])
      - offset: c = (weighted average of X)' * beta - logsumexp(s[k:])
      - y: 最初の順位は 1, 他は 0
      - grad: tilde_x = X[i_k] - (weighted average of X in {k,...,K})
    """
    eta_list = []
    offset_list = []
    y_list = []
    grad_list = []
    for t in event_times:
        indices = risk_sets[t]
        K = len(indices)
        if K < 2:
            continue  # リスク集合内に2件未満ならスキップ
        # リスク集合内の各被験者のスコア: s = X[indices] dot beta
        s = X[indices, :].dot(beta)  # shape: (K,)
        L = cumulative_logsumexp(s)  # shape: (K,)
        exp_s = np.exp(s)           # shape: (K,)
        cum_exp, cum_X = cumulative_weighted_X(X[indices, :], exp_s)  # shapes: (K,), (K, p)
        K_eff = K - 1  # 最後の被験者は除外（スティックブレイキング表現の性質より）
        eta_vals = s[:K_eff] - L[:K_eff]  # 各順位の eta
        avg_X = cum_X[:K_eff, :] / cum_exp[:K_eff, np.newaxis]  # 重み付き平均 \bar{x}
        grads = X[indices[:K_eff], :] - avg_X  # 局所線形化における \tilde{x} = x - \bar{x}
        # offset = c = \bar{x}^T beta - logsumexp(s)
        offsets = eta_vals - np.sum(grads * beta, axis=1)
        y_vals = np.zeros(K_eff)
        y_vals[0] = 1  # 最初の順位のみ事象発生とする
        eta_list.append(eta_vals)
        offset_list.append(offsets)
        y_list.append(y_vals)
        grad_list.append(grads)
    if len(eta_list) > 0:
        eta_all = np.concatenate(eta_list)
        offsets_all = np.concatenate(offset_list)
        y_all = np.concatenate(y_list)
        grads_all = np.concatenate(grad_list, axis=0)
    else:
        eta_all = np.array([])
        offsets_all = np.array([])
        y_all = np.array([])
        grads_all = np.empty((0, X.shape[1]))
    return eta_all, offsets_all, y_all, grads_all


def cox_pg_sampler(X, time, event, n_iter=1000, burn_in=0, beta_init=None,
                   prior_mean=None, prior_cov=None):
    """
    Cox-Polya-Gamma Gibbsサンプラー
      - X: 特徴量行列 (n x p)
      - time, event: 生存時間とイベント指示子
      - n_iter: 総反復回数
      - burn_in: burn-in期間（最初の burn_in サンプルは除外）
      - beta_init, prior_mean, prior_cov: 事前設定
    """
    n, p = X.shape
    if prior_mean is None:
        prior_mean = np.zeros(p)
    if prior_cov is None:
        prior_cov = np.eye(p) * 100.0
    if beta_init is None:
        beta = np.zeros(p)
    else:
        beta = beta_init.copy()

    beta_samples = []
    event_times, risk_sets = build_risk_sets(time, event)

    for it in tqdm(range(n_iter)):
        # すべてのリスク集合での eta, offset, y, grad を一括して計算
        eta_all, offsets_all, y_all, grads_all = compute_all_eta_gradients(X, beta, risk_sets, event_times)

        if eta_all.size == 0:
            # イベントがない場合は事前のみの更新
            beta = np.random.multivariate_normal(prior_mean, prior_cov)
            beta_samples.append(beta.copy())
            continue

        # 各 eta に対して Polya-Gamma 補助変数をサンプル
        omega_all = np.array([random_polyagamma(1, eta) for eta in eta_all])
        # 補正項付き kappa: (y - 0.5) - omega * offset
        kappa_all = (y_all - 0.5) - omega_all * offsets_all

        # 勾配項 grads_all に対して，精度行列の寄与を計算
        XtWX = np.einsum('i,ij,ik->jk', omega_all, grads_all, grads_all)
        prec_prior = inv(prior_cov)
        post_prec = prec_prior + XtWX
        post_cov = inv(post_prec)
        sum_kappa_grad = np.einsum('i,ij->j', kappa_all, grads_all)
        post_mean = post_cov.dot(prec_prior.dot(prior_mean) + sum_kappa_grad)

        # β のサンプル更新
        beta = np.random.multivariate_normal(post_mean, post_cov)
        beta_samples.append(beta.copy())

    beta_samples = np.array(beta_samples)
    return beta_samples[burn_in:]


# --- 使用例 ---
if __name__ == '__main__':
    beta_true = np.array([1.0, 0.5])
    n = 100
    X, time, event = simulate_cox_data(n, beta_true, seed=42)
    # burn-in 500回，総反復回数1000回でサンプル取得
    samples = cox_pg_sampler(X, time, event, n_iter=1000, burn_in=500)
    print("推定されたβの事後平均:", samples.mean(axis=0))
