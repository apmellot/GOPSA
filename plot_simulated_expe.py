# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import r2_score


DEBUG = False

# activate latex text rendering
plt.rc('text', usetex=True)

if DEBUG:
    all_results = pd.read_csv(
        './results/simulated_expe_debug.csv',
        index_col=0
    )
else:
    all_results = pd.read_csv(
        './results/simulated_expe.csv',
        index_col=0
    )

all_results.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

all_results = all_results.replace(
    [
        'shift_X',
        'shift_Y',
        'shift_XY',
    ],
    [
        'Shift in ' + r'$X$',
        'Shift in ' + r'$y$',
        'Shift in ' + r'($X$, $y$)',
    ]
)

all_results = all_results.replace(
    [
        'dummy',
        'baseline_no_recenter',
        'baseline_green',
        'baseline_recenter',
        'baseline_rescale',
        'baseline_fit_intercept',
        'geodesic_optim'
    ],
    [
        r'\texttt{DO Dummy}',
        r'\texttt{No DA}',
        r'\texttt{GREEN}',
        r'\texttt{Re-center}',
        r'\texttt{Re-scale}',
        r'\texttt{DO Intercept}',
        r'\texttt{GOPSA}',
    ]
)
order = [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
]

# group by (method, scenario, parameter, random_state)
# and compute R2 from y_true and y_pred
all_results = all_results.groupby(
    ['method', 'scenario', 'parameter', 'random_state']
).apply(
    lambda x: pd.Series({
        'r2': r2_score(x['y_true'], x['y_pred'])
    })
).reset_index()

sns.set_theme(style="ticks", font_scale=2)
sns.set_palette('colorblind')
palette = np.array(sns.color_palette("colorblind"))[:7]
palette = list(palette[[3, 1, 5, 2, 6, 0, 4]])
g = sns.FacetGrid(
    all_results,
    col="scenario",
    col_wrap=3,
    legend_out=True,
    hue='method',
    hue_kws=dict(marker=['s', 'o', '<', 'X', 'v', '>', 'P'],
                 ls=['--', '-.', ':',
                     (0, (5, 10)),
                     (0, (3, 5, 1, 5, 1, 5)),
                     (0, (1, 10)),
                     '-']),
    height=4, aspect=1,
    margin_titles=True,
    sharex=False,
    hue_order=order,
    palette=palette
)

g.map_dataframe(sns.lineplot, 'parameter', 'r2', markersize=10)
g.set(ylim=(-0.15, 1.1))
g.set_axis_labels("Parameter value", r"$R^2$")

# g.set(xscale='log')
g.add_legend(title='Methods')
g.set_titles(col_template='{col_name}')
letters = ['A', 'B', 'C', 'D']
for i, ax in enumerate(g.axes.flat):
    ax.set_xlabel(r'$\xi$')

    # Change x_ticks with "No shift", "Max. shift"
    shift_values = all_results[
        all_results['scenario'] == all_results['scenario'].unique()[i]
    ]['parameter'].unique()
    min_shift = min(shift_values)
    max_shift = max(shift_values)
    ax.set_xticks([min_shift, max_shift])
    ax.set_xticklabels(['No shift', 'Max. shift'])

    # Add letter
    ax.annotate(
        text=letters[i], xy=(-0.1, 1.13),
        xycoords=('axes fraction'), fontsize=25,
        weight='bold'
    )

sns.despine(trim=True)
g.tight_layout()

if DEBUG:
    g.savefig(
        FIGURES_FOLDER / "simulated_expe_debug.pdf",
        bbox_inches='tight'
    )
else:
    g.savefig(
        FIGURES_FOLDER / "simulated_expe.pdf",
        bbox_inches='tight'
    )
