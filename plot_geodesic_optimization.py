# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import scipy.stats
import os
import re

# %%
# Load data
df_all = []
# get all filnames beginning with results/geodesic_optimization_results_
filenames = os.listdir('results')
filenames = [f for f in filenames if re.match(
    r'geodesic_optimization_results_\d+', f
    )]
filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))
for filename in filenames:
    df = pd.read_csv(f'results/{filename}', index_col=0)
    df_all.append(df)
df_all = pd.concat(df_all)
df_all.reset_index(inplace=True, drop=True)


def const_line(*args, **kwargs):
    x = [0, 90]
    y = x
    plt.plot(y, x, 'k--')


palette = np.array(sns.color_palette("colorblind"))[:7]
palette = list(palette[[3, 1, 5, 2, 6, 0, 4]])

df_all = df_all.drop(
    df_all[df_all['sites_source_index'].isin([6])].index
)

# %% Make a df for scores for site_target combined
df_score = []

for i in df_all['sites_source_index'].unique():
    df = df_all[df_all['sites_source_index'] == i]
    for method in df['method'].unique():
        df_method = df[df['method'] == method]
        for split in df_method['split_index'].unique():
            df_split = df_method[df_method['split_index'] == split]
            r2 = sklearn.metrics.r2_score(df_split['y_true'],
                                          df_split['y_pred'])
            mae = sklearn.metrics.mean_absolute_error(df_split['y_true'],
                                                      df_split['y_pred'])
            spearman_rank = scipy.stats.spearmanr(
                df_split['y_true'], df_split['y_pred'],
                nan_policy='omit').statistic
            df_score.append({'site_source': i,
                             'method': method,
                             'split_index': split,
                             'r2': r2,
                             'mae': mae,
                             'spearman': 0 if np.isnan(
                                 spearman_rank) else spearman_rank
                             })
df_score = pd.DataFrame(df_score)

# %% Change font to latex
plt.rcParams.update({
    "text.usetex": True
})

df_score = df_score.replace([
    'dummy',
    'baseline_no_recenter',
    'baseline_green',
    'baseline_recenter',
    'baseline_rescale',
    'baseline_fit_intercept',
    'geodesic_optim'
], [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
])


# %% Make plot of the 3 scores for all combinatins of sites_source
sns.set_theme(font_scale=2, style='ticks')

order = [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, score in enumerate(['spearman', 'r2', 'mae']):
    sns.boxplot(data=df_score, x=score, y='method', order=order,
                hue='method', showfliers=False, palette=palette, ax=axes[i])
    for patch in axes[i].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))
    sns.stripplot(data=df_score, x=score, y='method', order=order,
                  hue='method', palette=palette, alpha=.5, ax=axes[i])

axes[0].set_ylabel('')
axes[0].set_xlabel(r"Spearman's $\rho$ $\uparrow$")
axes[1].set_xlabel(r'$R^2$ score $\uparrow$')
axes[2].set_xlabel(r'MAE (years) $\downarrow$')
axes[0].set_xticks(
    [0.5, 0.75, 1]
)
axes[1].set_xticks(
    [-0.2, 0.2, 0.6, 0.8]
)
axes[2].set_xticks(
    [7, 9, 11, 13, 15]
)
sns.despine(fig=fig, trim=True)
plt.tight_layout()
plt.show()
fig.savefig('figures/results_figure.pdf')

# %%
order = [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
]
sns.set_theme(font_scale=2, style='ticks')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, score in enumerate(['spearman', 'r2', 'mae']):
    sns.boxplot(data=df_score, x=score, y='method', order=order,
                hue='method', showfliers=False, palette=palette, ax=axes[i])
    for patch in axes[i].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))
    sns.stripplot(data=df_score, x=score, y='method', order=order,
                  hue='method', palette=palette, alpha=.5, ax=axes[i])

axes[0].set_ylabel('')
axes[0].set_xlabel(r"Spearman's $\rho$ $\uparrow$")
axes[1].set_xlabel(r'$R^2$ score $\uparrow$')
axes[2].set_xlabel(r'MAE (years) $\downarrow$')
axes[0].set_xticks(
    [0.3, 0.6, 0.9]
)
axes[1].set_xticks(
    [-2, -1, 0, 1]
)
axes[2].set_xticks(
    [7, 10, 13, 16, 19, 22]
)
sns.despine(fig=fig, trim=True)
plt.tight_layout()
plt.show()
fig.savefig('figures/results_figure_full.pdf')

# %% Make plot of the 3 scores for seperate combinations of sites_source
sns.set_theme(font_scale=1.3, style='ticks')

sites_source = ['Ba,Cho,G,S', 'Be,Chb,S', 'Ba,Co,G',
                'Cu03,M,R,S', 'Ba,Be,Cho,Co,Cu90,G,R']

score = 'spearman'

g = sns.FacetGrid(df_score, col='site_source', col_wrap=3,
                  sharex=False)
g.map(sns.boxplot, score, "method", order=order,
      palette=palette, showfliers=False)
g.map(sns.stripplot, score, "method", order=order,
      palette=palette, alpha=.5)
if score == 'spearman':
    g.set(xlabel=r"Spearman's $\rho$ $\uparrow$")
elif score == 'mae':
    g.set(xlabel=r'MAE (years) $\downarrow$')
elif score == 'r2':
    g.set(xlabel=r'$R^2$ score $\uparrow$')
g.set(ylabel='')
axes = g.axes.flatten()
for i in range(len(axes)):
    axes[i].set_title(sites_source[i], fontsize=12)
    for patch in axes[i].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))
    if score == 'spearman':
        axes[i].set_xticks(
            [0.4, 0.6, 0.8]
        )
    if score == 'mae':
        axes[i].set_xticks(
            [7, 10, 13, 16]
        )
if score == 'r2':
    axes[1].set_xticks(
        [-2, 0, 2]
    )
    axes[2].set_xticks(
        [0, 0.25, 0.5, 0.75]
    )
    axes[-1].set_xticks(
        [0, 0.25, 0.5, 0.75]
    )
if score == 'mae':
    axes[1].set_xticks(
        [8, 12, 16, 20, 24]
    )
fig = plt.gcf()
sns.despine(fig=fig, trim=True)
plt.tight_layout()
plt.savefig(f'figures/{score}.pdf')

# %% Plot normalized results

df_score = []

for i in df_all['sites_source_index'].unique():
    df = df_all[df_all['sites_source_index'] == i]
    for method in df['method'].unique():
        df_method = df[df['method'] == method]
        for split in df_method['split_index'].unique():
            df_split = df_method[df_method['split_index'] == split]
            r2 = sklearn.metrics.r2_score(df_split['y_true'],
                                          df_split['y_pred'])
            mae = sklearn.metrics.mean_absolute_error(df_split['y_true'],
                                                      df_split['y_pred'])
            spearman_rank = scipy.stats.spearmanr(
                df_split['y_true'], df_split['y_pred'],
                nan_policy='omit').statistic
            df_score.append({'site_source': i,
                             'method': method,
                             'split_index': split,
                             'r2': r2,
                             'mae': mae,
                             'spearman': 0 if np.isnan(
                                 spearman_rank) else spearman_rank
                             })
df_score = pd.DataFrame(df_score)
# %%
df_score_norm = []
for site in df_score['site_source'].unique():
    df_site = df_score[df_score['site_source'] == site]
    for score in ['spearman', 'r2', 'mae']:
        site_score = df_site[score].values
        site_score_norm = (
            site_score - np.min(site_score)
        ) / (np.max(site_score) - np.min(site_score))
        df_site[score] = site_score_norm
    df_score_norm.append(df_site)
df_score_norm = pd.concat(df_score_norm)

plt.rcParams.update({
    "text.usetex": True
})

df_score_norm = df_score_norm.replace([
    'dummy',
    'baseline_no_recenter',
    'baseline_green',
    'baseline_recenter',
    'baseline_rescale',
    'baseline_fit_intercept',
    'geodesic_optim'
], [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
])


df_score = df_score.replace([
    'dummy',
    'baseline_no_recenter',
    'baseline_green',
    'baseline_recenter',
    'baseline_rescale',
    'baseline_fit_intercept',
    'geodesic_optim'
], [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
])

# %% Same figure with re-center
order = [
    r'\texttt{DO Dummy}',
    r'\texttt{No DA}',
    r'\texttt{GREEN}',
    r'\texttt{Re-center}',
    r'\texttt{Re-scale}',
    r'\texttt{DO Intercept}',
    r'\texttt{GOPSA}'
]
sites_source = ['Ba,Cho,G,S', 'Be,Chb,S', 'Ba,Co,G',
                'Cu03,M,R,S', 'Ba,Be,Cho,\nCo,Cu90,G,R']

sns.set_theme(font_scale=2.2, style='ticks')

fig, axes = plt.subplot_mosaic(
    [['spearman', 'r2', 'mae'],
     ['spearman', 'r2', 'mae'],
     ['spearman', 'r2', 'mae'],
     ['spearman', 'r2', 'mae'],
     ['diff_spearman_0', 'diff_r2_0', 'diff_mae_0'],
     ['diff_spearman_1', 'diff_r2_1', 'diff_mae_1'],
     ['diff_spearman_2', 'diff_r2_2', 'diff_mae_2'],
     ['diff_spearman_3', 'diff_r2_3', 'diff_mae_3'],
     ['diff_spearman_4', 'diff_r2_4', 'diff_mae_4']],

    sharey=False,
    layout='constrained',
    figsize=(17, 9)
    )

for score in ['spearman', 'r2', 'mae']:
    sns.boxplot(data=df_score_norm, x=score, y='method', order=order,
                hue='method', showfliers=False, palette=palette,
                ax=axes[score])
    for patch in axes[score].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))
    sns.stripplot(data=df_score_norm, x=score, y='method', order=order,
                  hue='method', palette=palette, ax=axes[score])
    axes[score].set_xlim([-0.05, 1.05])
    axes[score].set_xticks(
        [0, 0.25, 0.5, 0.75, 1],
        labels=['0', '0.25', '0.5', '0.75', '1']
    )

# Read p-values
p_values = pd.read_csv('results/p_values.csv', index_col=0)

for i, site in enumerate(df_score_norm['site_source'].unique()):
    df_site = df_score_norm[df_score_norm['site_source'] == site]
    df1 = df_site[
        df_site['method'] == r'\texttt{DO Intercept}'
    ].reset_index()
    df2 = df_site[
        df_site['method'] == r'\texttt{GOPSA}'
    ].reset_index()
    assert np.all(df1['split_index'] == df2['split_index'])
    df_diff = {}
    for score in ['spearman', 'r2', 'mae']:
        df_diff[f'diff_{score}'] = df2[score] - df1[score]
    df_diff = pd.DataFrame(df_diff)
    for diff in ['diff_spearman', 'diff_r2', 'diff_mae']:
        sns.boxplot(data=df_diff, x=diff, ax=axes[f'{diff}_{i}'],
                    showfliers=False, color='gray')
        for patch in axes[f'{diff}_{i}'].patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .5))
        sns.stripplot(data=df_diff, x=diff, ax=axes[f'{diff}_{i}'],
                      color='gray')
        axes[f'{diff}_{i}'].set_xlabel('')
        axes[f'{diff}_{i}'].axvline(x=0, color='k', linestyle='--')
        axes[f'{diff}_{i}'].set_xlim([-0.2, 0.2])
        if i == 4:
            axes[f'{diff}_{i}'].set_xticks(
                [-0.2, -0.1, 0, 0.1, 0.2],
                labels=['0.2', '-0.1', '0', '0.1', '0.2']
            )
        else:
            axes[f'{diff}_{i}'].set_xticks(
                [-0.2, -0.1, 0, 0.1, 0.2],
                labels=[]
            )

    axes[f'diff_spearman_{i}'].sharex(axes[f'diff_r2_{i}'])
    axes[f'diff_r2_{i}'].set_yticklabels([])
    axes[f'diff_r2_{i}'].set_ylabel('')
    axes[f'diff_mae_{i}'].set_yticklabels([])
    axes[f'diff_mae_{i}'].set_ylabel('')
    axes[f'diff_spearman_{i}'].set_yticks(
        [0],
        labels=[f'{sites_source[i]}']
    )
    p_vs = p_values[p_values['source'] == i+1]
    p_vs = p_vs.iloc[-1]
    for score in ['spearman', 'r2']:
        p_v = p_vs[score]
        axes[f'diff_{score}_{i}'].annotate(
            xy=(0, 0.15), text=r'$p={:.3f}$'.format(p_v),
            xycoords='axes fraction', fontsize=24)
    p_v = p_vs['mae']
    axes[f'diff_mae_{i}'].annotate(
        xy=(1, 0.15), text=r'$p={:.3f}$'.format(p_v),
        xycoords='axes fraction', fontsize=24,
        horizontalalignment='right')

axes['diff_r2_0'].set_title(
        r'Difference: \texttt{GOPSA} - \texttt{DO Intercept}'
    )

for axes_labels in ['r2', 'mae']:
    axes[axes_labels].set_yticklabels([])
    axes[axes_labels].set_ylabel('')

axes['spearman'].set_ylabel('')
axes['spearman'].set_xlabel(r"Normalized Spearman's $\rho$ $\uparrow$")
axes['r2'].set_xlabel(r'Normalized $R^2$ score $\uparrow$')
axes['mae'].set_xlabel(r'Normalized MAE $\downarrow$')

axes['diff_spearman_4'].set_xlabel(r"Normalized Spearman's $\rho$ $\uparrow$")
axes['diff_r2_4'].set_xlabel(r'Normalized $R^2$ score $\uparrow$')
axes['diff_mae_4'].set_xlabel(r'Normalized MAE $\downarrow$')

axes['diff_spearman_0'].annotate(
    xy=(-0.5, 0.9), text='B', xycoords='axes fraction',
    weight='bold', fontsize=35)
axes['spearman'].annotate(xy=(-0.5, 0.9), text='A', xycoords='axes fraction',
                          weight='bold', fontsize=35)

sns.despine(fig=fig, trim=True)
plt.show()
fig.savefig('figures/results_figure_relative_recenter.pdf')
# %%
# Values for table
sites_source = ['Ba,Cho,G,S', 'Be,Chb,S', 'Ba,Co,G',
                'Cu03,M,R,S', 'Ba,Be,Cho,Co,Cu90,G,R']

df_table = []
score = 'r2'
for method in df_score['method'].unique():
    df_method = df_score[df_score['method'] == method]
    mean_ = []
    std_ = []
    for i in df_method['site_source'].unique():
        df_site = df_method[df_method['site_source'] == i]
        mean = np.mean(
            df_site[score].values
        )
        mean_.append(mean)
        std = np.std(
            df_site[score].values
        )
        std_.append(std)
        df_table.append({'site_source': i,  # sites_source[i-1],
                         'method': method,
                         'mean': mean,
                         'std': std})
    score_mean = np.mean(mean_)
    score_std = np.mean(std_)
    df_table.append({'site_source': 100,  # 'Mean',
                     'method': method,
                     'mean': score_mean,
                     'std': score_std})
df_table = pd.DataFrame(df_table)
df_table = df_table.sort_values(by=['site_source'])


# Custom function to format the mean and std values
def format_mean_std(df):
    if score == 'mae':
        formatted_values = []
        for site_source in np.unique(df['site_source'].values):
            df_site = df[df['site_source'] == site_source]
            min_mean = df_site['mean'].values.min()
            for mean, std in zip(df_site['mean'].values,
                                 df_site['std'].values):
                if mean == min_mean:
                    formatted_values.append(
                        "\\textbf{{ {:.2f} $\pm$ {:.2f} }}".format(mean, std)
                    )
                else:
                    formatted_values.append(
                        "{:.2f} $\pm$ {:.2f}".format(mean, std)
                    )
    else:
        formatted_values = []
        for site_source in np.unique(df['site_source'].values):
            df_site = df[df['site_source'] == site_source]
            max_mean = df_site['mean'].values.max()
            for mean, std in zip(df_site['mean'], df_site['std']):
                if mean == max_mean:
                    formatted_values.append(
                        "\\textbf{{ {:.2f} $\pm$ {:.2f} }}".format(mean, std)
                    )
                else:
                    formatted_values.append(
                        "{:.2f} $\pm$ {:.2f}".format(mean, std)
                    )
    return formatted_values


# Apply the custom formatting function to create a new column
df_table['mean_std'] = format_mean_std(df_table)

# Pivot the DataFrame to have methods as columns
pivot_df = df_table.pivot(index='site_source',
                          columns='method',
                          values='mean_std')
pivot_df = pivot_df[order]
pivot_df.index = sites_source + ['Mean']
# Convert to LaTeX table
latex_table = pivot_df.to_latex(escape=False)
print(latex_table)