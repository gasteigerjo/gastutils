import numpy as np
import matplotlib as mpl
from IPython.display import IFrame
mpl.use('pgf')


def figsize(scale, ratio_yx=None):
    fig_width_pt = 397.48499  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
    if ratio_yx is None:
        ratio_yx = golden_mean
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio_yx  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def sns_facetsize(tot_width=0.95, ratio_yx_facet=1.6, nrows=1, ncols=1):
    ratio_yx = ratio_yx_facet * nrows / ncols
    size = figsize(tot_width, ratio_yx)
    height_facet = size[1] / nrows
    ratio_xy_facet = 1 / ratio_yx_facet
    return height_facet, ratio_xy_facet


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        r"\usepackage{amsmath}",
        r"\usepackage{accents}",  # for hattilde
        r"\newlength{\dtildeheight}"
        r"\newcommand{\hattilde}[1]{"
        r"\settoheight{\dtildeheight}{\ensuremath{\tilde{#1}}}"
        r"\addtolength{\dtildeheight}{-0.1ex}"
        r"\hat{\vphantom{\rule{1pt}{\dtildeheight}}"
        r"\smash{\tilde{#1}}}}",
        r"\newcommand*{\mat}[1]{\boldsymbol{#1}}",
    ]
}
mpl.rcParams.update(pgf_with_latex)

# Set line widths
default_params = {
    'grid.linewidth': 0.5,
    'grid.color': '.8',
    'axes.linewidth': 0.75,
    'axes.edgecolor': '.7',
}

import matplotlib.pyplot as plt
import seaborn as sns


def set_style(style='whitegrid'):
    if style:
        sns.set(style=style, color_codes=True)
    else:
        sns.set(color_codes=True)
    mpl.rcParams.update(pgf_with_latex)
    mpl.rcParams.update(default_params)


# Customized newfig and savefig functions
def newfig(
        width, ratio_yx=None, style='whitegrid', subplots=True,
        nrows=1, ncols=1):
    # plt.clf()
    set_style(style=style)
    if subplots:
        return plt.subplots(
                nrows, ncols,
                figsize=figsize(width, ratio_yx=ratio_yx))
    else:
        return plt.subplots(figsize=figsize(width, ratio_yx=ratio_yx))


def savefig(
        filename, fig=None, tight={'pad': 0.5},
        dpi=600, formats=['pdf', 'pgf'], **kwargs):
    if fig is None:
        if tight:
            plt.tight_layout(**tight)
        for fmt in formats:
            plt.savefig('{}.{}'.format(filename, fmt), dpi=dpi, **kwargs)
    else:
        if tight:
            fig.tight_layout(**tight)
        for fmt in formats:
            fig.savefig('{}.{}'.format(filename, fmt), dpi=dpi, **kwargs)
    return IFrame(f"{filename}.{formats[0]}", width=700, height=500)


if __name__ == "__main__":
    # Simple plot
    fig, ax = newfig(0.6)

    def ema(y, a):
        s = []
        s.append(y[0])
        for t in range(1, len(y)):
            s.append(a * y[t] + (1 - a) * s[t - 1])
        return np.array(s)

    y = [0] * 200
    y.extend([20] * (1000 - len(y)))
    s = ema(y, 0.01)

    ax.plot(s)
    ax.set_xlabel('X Label')
    ax.set_ylabel('EMA')

    # savefig('ema')
