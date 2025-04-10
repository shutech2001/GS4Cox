import numpy as np


def autocorrelation(x, max_lag=None):
    """
    1次元配列 x の自己相関を計算する関数.
    max_lag が指定されていればその値まで自己相関を返す.
    返り値はラグ0からmax_lagまでの自己相関の1次元配列（ラグ0は1）。
    """
    x = np.asarray(x)
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    # 平均を引く
    x = x - np.mean(x)
    var = np.var(x)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            # 有効なデータ点は n - lag 個
            acf[lag] = np.sum(x[:-lag]*x[lag:]) / ((n - lag) * var)
    return acf


def compute_ess(chain, max_lag=None):
    """
    MCMCサンプルチェーンから各パラメータのESSを計算する.

    Parameters
    ----------
    chain : numpy.ndarray
        形状 (n_samples, n_params) のMCMCサンプルチェーン。
    max_lag : int or None
        自己相関の計算で用いる最大ラグ。Noneの場合は (n_samples-1) を用いるが、
        実際には自己相関が正の範囲のみを和に足す（切り捨て）例も多い。

    Returns
    -------
    ess : numpy.ndarray
        各パラメータごとのESSの1次元配列（長さ n_params）。
    """
    chain = np.atleast_2d(chain)
    n_samples, n_params = chain.shape
    ess = np.empty(n_params)

    for i in range(n_params):
        x = chain[:, i]
        # 自己相関関数を計算：ラグ=0は1
        acf = autocorrelation(x, max_lag=max_lag)
        # 自己相関が正であるラグのみを和に足す（または連続した正のみを採用する方法も考えられる）
        positive_acf = acf[1:][acf[1:] > 0]
        if len(positive_acf) == 0:
            summation = 0.0
        else:
            summation = np.sum(positive_acf)
        ess[i] = n_samples / (1.0 + 2.0 * summation)
    return ess


def compute_esr(chain, runtime, max_lag=None):
    """
    chain と実行時間 runtime (秒)から各パラメータのESR (Effective Sampling Rate) を計算する。

    ESR は、ESS を実行時間（秒）で割ったものです。

    Parameters
    ----------
    chain : numpy.ndarray
        形状 (n_samples, n_params) のMCMCサンプルチェーン
    runtime : float
        サンプラーの実行時間（秒）
    max_lag : int or None
        自己相関の計算で用いる最大ラグ（compute_essに渡す）

    Returns
    -------
    esr : numpy.ndarray
        各パラメータごとのESRの1次元配列（単位：【サンプル/秒】）
    """
    ess = compute_ess(chain, max_lag=max_lag)
    return ess / runtime


def compute_dist(chain):
    differences = np.diff(chain, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    avg_dist = np.mean(distances)
    return avg_dist