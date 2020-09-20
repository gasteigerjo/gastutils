import os
import copy
import math
import pickle
import gzip
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns


def get_accuracy(predictions: np.array, labels: np.array):
    return np.mean(predictions == labels)


def get_f1(predictions: np.array, labels: np.array):
    nclasses = np.max(labels) + 1
    avg_precision = 0
    avg_recall = 0
    for i in range(nclasses):
        pred_is_i = predictions == i
        label_is_i = labels == i
        true_pos = np.sum(pred_is_i & label_is_i)
        false_pos = np.sum(pred_is_i & ~label_is_i)
        false_neg = np.sum(~pred_is_i & label_is_i)
        if false_pos == 0:
            avg_precision += 1.
        else:
            avg_precision += true_pos / (true_pos + false_pos)
        if false_neg == 0:
            avg_recall += 1.
        else:
            avg_recall += true_pos / (true_pos + false_neg)
    avg_precision /= nclasses
    avg_recall /= nclasses
    f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall))
    return f1_score


def get_confusion_matrix(predictions: np.array, labels: np.array):
    return pd.crosstab(
            predictions, labels,
            rownames=['Predictions'], colnames=['Labels'])


def pickle_cache(filename, fn, fn_args=[], fn_kwargs={}, compression=False):
    if os.path.isfile(filename):
        if compression:
            with gzip.open(filename, 'rb') as f:
                obj = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
    else:
        obj = fn(*fn_args, **fn_kwargs)
        if compression:
            with gzip.open(filename, 'wb') as f:
                pickle.dump(obj, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
    return obj


def softmax(logits, axis=-1):
    after_exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return after_exp / np.sum(after_exp, axis=axis, keepdims=True)


def entropy(p, axis=-1):
    return -np.sum(p * np.log2(p), axis=axis)


def margin(p, axis=-1):
    sorted_p = np.sort(p, axis=axis)
    return np.take(sorted_p, -1, axis=axis) - np.take(sorted_p, -2, axis=axis)


def calc_uncertainty(values: np.ndarray, n_boot: int = 1000, ci: int = 95) -> dict:
    stats = {}
    stats['mean'] = values.mean()
    boots_series = sns.algorithms.bootstrap(values, func=np.mean, n_boot=n_boot)
    stats['CI'] = sns.utils.ci(boots_series, ci)
    stats['uncertainty'] = np.max(np.abs(stats['CI'] - stats['mean']))
    return stats


def paired_ttest(
        df, top, value, select='Model', exclude=['PPNP', 'APPNP']):
    df_top = df[df[select] == top]
    df_top.index = np.arange(len(df_top))
    df_others = df[(df[select] != top) & ~(df[select].isin(exclude))]
    if df_top[value].mean() < df_others.groupby([select])[value].mean().max():
        return np.nan
    else:
        pvals = []
        contenders = df_others[select].unique()
        for contender in contenders:
            df_contender = df[df[select] == contender]
            df_contender.index = np.arange(len(df_contender))
            pvals.append(stats.ttest_1samp(
                df_top[value] - df_contender[value], 0).pvalue)
        return max(pvals)


def bold_best(
        df, df_text, group_cols, max_is_best=True, exclude_model=[],
        model_col='Model', value_col='value', error_col=None, rtol=1e-5, atol=1e-8):
    df_filtered = df.loc[[model not in exclude_model for model in df[model_col]]]
    if max_is_best:
        bests_idx = df_filtered.groupby(group_cols)[value_col].idxmax()
    else:
        bests_idx = df_filtered.groupby(group_cols)[value_col].idxmin()

    # Get close 2nds
    close_bests_idx = []
    for name, g in df_filtered.groupby(group_cols):
        if error_col is None:
            isclose = np.isclose(g[value_col], df_filtered.loc[bests_idx[name], value_col], rtol=rtol, atol=atol)
        else:
            isclose = np.abs(g[value_col] - df_filtered.loc[bests_idx[name], value_col]) <= (g[error_col] + df_filtered.loc[bests_idx[name], error_col])
        close_bests_idx.extend(g[isclose].index)

    df_bold = df_text.copy()
    for idx in close_bests_idx:
        df_bold.loc[idx, value_col] = f"\\textbf{{{df_text.loc[idx, value_col]}}}"
    return df_bold


def save_latex_table(
        filename, df, col_order=None, row_order=None,
        insert_hlines=[], **kwargs):
    old_pd_colwidth = pd.options.display.max_colwidth
    pd.options.display.max_colwidth = 1_000
    df_output = copy.deepcopy(df)
    if col_order is not None:
        df_output = df_output.reindex(columns=col_order)
    if row_order is not None:
        df_output = df_output.reindex(index=row_order)
    df_output.columns.name = df_output.index.name
    df_output.index.name = ''
    df_output = df_output.fillna('-')
    df_output = df_output.applymap(lambda x: "-" if x == "nan" else x)
    latex = df_output.to_latex(escape=False, **kwargs)
    latex = latex.replace('midrule', 'hline')
    latex_list = latex.splitlines()
    del latex_list[1]
    del latex_list[2]
    del latex_list[-2]
    for line in insert_hlines:
        latex_list.insert(line + 3, '\\hline')
    latex = '\n'.join(latex_list)
    with open(f'{filename}.tex', 'w') as f_output:
        f_output.write(latex)
    pd.options.display.max_colwidth = old_pd_colwidth


def fmt_latex(x, fmt='.2f'):
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    if mant == 1:
        s = f"10^{{{exp}}}"
    else:
        s = f"{{mant:{fmt}}} \cdot 10^{{{{{exp}}}}}".format(mant=mant)
    return s


def to_precision(x, p):
    """
    Returns a string representation of x formatted with a precision of p

    From: http://randlet.com/blog/python-significant-figures-format/
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


def gaussian_filter(xs_grid, xs_data, ys_data, sigma):
    weights = np.exp(-(xs_data - xs_grid[:, None])**2 / (2 * sigma**2))
    return np.sum(weights * ys_data, 1) / np.sum(weights, 1)


def gaussian_filter_bt(xs_grid, xs_data, ys_data, sigma):
    bootstrap_res = sns.algorithms.bootstrap(
            np.column_stack((xs_data, ys_data)),
            func=lambda x: gaussian_filter(xs_grid, x[:, 0], x[:, 1], sigma=sigma),
            n_boot=2000)

    mean = bootstrap_res.mean(0)
    ci = sns.utils.ci(bootstrap_res, axis=0)
    return mean, ci
