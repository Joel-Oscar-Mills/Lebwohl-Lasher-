import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

print(plt.style.available)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('text.latex', preamble=R'\usepackage{amsmath} \usepackage{bbold}')
plt.rcParams.update(tex_fonts)
width = 252

path2 = "./numba_runtime_vs_NMAX_2.txt"
path4 = "./numba_runtime_vs_NMAX_4.txt"
path8 = "./numba_runtime_vs_NMAX_8.txt"
path16 = "./numba_runtime_vs_NMAX_16.txt"
path32 = "./numba_runtime_vs_NMAX_32.txt"

list_NMAX2 = np.array([ceil(2*(1000/2)**(i/20)) for i in range(21)])
runtimes2 = np.loadtxt(path8)

list_NMAX4 = np.array([ceil(4*(1000/4)**(i/20)) for i in range(21)])
runtimes4 = np.loadtxt(path8)

#list_NMAX8 = np.array([ceil(8*(10000/8)**(i/20)) for i in range(21)])
list_NMAX8 = np.array([ceil(8*(1000/8)**(i/20)) for i in range(21)])
runtimes8 = np.loadtxt(path8)

#list_NMAX16 = np.array([ceil(16*(10000/16)**(i/20)) for i in range(21)])
list_NMAX16 = np.array([ceil(16*(1000/16)**(i/20)) for i in range(21)])
runtimes16 = np.loadtxt(path16)

#list_NMAX32 = np.array([ceil(32*(10000/32)**(i/20)) for i in range(21)])
list_NMAX32 = np.array([ceil(32*(1000/32)**(i/20)) for i in range(21)])
runtimes32 = np.loadtxt(path32)

fig, ax = plt.subplots(1, 1, figsize=set_size(width))
ax.set(xlabel=R'NMAX', ylabel=R"Runtime (s)")

ax.grid(which='minor',axis='both',linewidth=0.3)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.plot(list_NMAX2, runtimes2, color="g",linewidth=0.5,label=R"STEPS = 5, NP = 2")
#ax.plot(list_NMAX4, runtimes4, color="b",linewidth=0.5,label=R"STEPS = 5, NP = 4")
#ax.plot(list_NMAX8, runtimes8, color="m",linewidth=0.5,label=R"STEPS = 5, NP = 8")
ax.plot(list_NMAX16, runtimes16, color="k",linewidth=0.5,label=R"STEPS = 5, NP = 16")
#ax.plot(list_NMAX32, runtimes32, color="r",linewidth=0.5,label=R"STEPS = 5, NP = 32")
ax.legend()
#plt.savefig("./NMAX_vs_runtime.pdf", format='pdf', bbox_inches="tight")
plt.show()