import os
import copy
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


def bold_top(df, df_text, value):
    tops = df.loc[df.groupby(['Graph'])[value].idxmax()]
    for graph, model in zip(tops['Graph'], tops['Model']):
        selection = (df_text['Graph'] == graph) & (df_text['Model'] == model)
        entry = df_text.loc[selection, value]
        df_text.loc[selection, value] = f"\\textbf{{{entry.iloc[0]}}}"


def save_latex_table(filename, df, col_order=None, row_order=None, **kwargs):
    df_output = copy.deepcopy(df)
    if col_order is not None:
        df_output = df_output[col_order]
    if row_order is not None:
        df_output = df_output.reindex(index=row_order)
    df_output.columns.name = df_output.index.name
    df_output.index.name = ''
    df_output = df_output.fillna('-')
    latex = df_output.to_latex(escape=False, **kwargs)
    latex = latex.replace('midrule', 'hline')
    latex_list = latex.splitlines()
    del latex_list[1]
    del latex_list[2]
    del latex_list[-2]
    latex = '\n'.join(latex_list)
    with open(f'tables/{filename}.tex', 'w') as f_output:
        f_output.write(latex)

def fmt_latex(x, fmt='.2f'):
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    if mant == 1:
        s = f"10^{{{exp}}}"
    else:
        s = f"{{mant:{fmt}}} \cdot 10^{{{{{exp}}}}}".format(mant=mant)
    return s
