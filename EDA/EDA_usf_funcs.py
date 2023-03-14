import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings

warnings.filterwarnings('ignore')


# for kaggle seasons:
def compare_distrubs(train, test, origin=None):
    features = test.columns
    n_bins = 50
    histplot_hyperparams = {
        'kde': True,
        'alpha': 0.4,
        'stat': 'percent',
        'bins': n_bins
    }

    columns = features
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    ax = ax.flatten()

    for i, column in enumerate(columns):
        plot_axes = [ax[i]]
        sns.kdeplot(
            train[column], label='Train',
            ax=ax[i], color='red'
        )

        sns.kdeplot(
            test[column], label='Test',
            ax=ax[i], color='blue'
        )

        if origin != None:
            sns.kdeplot(
                origin[column], label='Original',
                ax=ax[i], color='green'
            )

        ax[i].set_title(f'{column} Distribution')
        ax[i].set_xlabel(None)

        plot_axes = [ax[i]]
        handles = []
        labels = []
        for plot_ax in plot_axes:
            handles += plot_ax.get_legend_handles_labels()[0]
            labels += plot_ax.get_legend_handles_labels()[1]
            plot_ax.legend().remove()

    for i in range(i + 1, len(ax)):
        ax[i].axis('off')

    fig.suptitle(f'Dataset Feature Distributions\n\n\n', ha='center', fontweight='bold', fontsize=25)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=3)
    plt.tight_layout()


def plot_count(df, col_list: list, title_name: str = 'Train') -> None:
    palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
               '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

    f, ax = plt.subplots(len(col_list), 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0)

    s1 = df[col_list].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1 / N

    outer_colors = [palette[0], palette[0], '#ff781f', '#ff9752', '#ff9752']
    inner_colors = [palette[1], palette[1], '#ffa66b']

    ax[0].pie(
        outer_sizes, colors=outer_colors,
        labels=s1.index.tolist(),
        startangle=90, frame=True, radius=1.3,
        explode=([0.05] * (N - 1) + [.3]),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size': 13,
        'weight': 'bold',
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1] * (N - 1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0, 0), .68, color='black',
                               fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = [0, 1]
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=palette[:2], orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i + 0.1, str(v), color='black',
                   fontweight='bold', fontsize=12)

    #     plt.title(col_list)
    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col_list, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()


def shapes_and_miss_vals_info(train, test, origin):
    from colorama import Style, Fore

    blu = Style.BRIGHT + Fore.BLUE
    red = Style.BRIGHT + Fore.RED
    print(f'{blu}[INFO] Shapes:'
          f'{blu}\n[+] origin ->  {red}{origin.shape}'
          f'{blu}\n[+] train  -> {red}{train.shape}'
          f'{blu}\n[+] test   ->  {red}{test.shape}\n')

    print(f'{blu}[INFO] Any missing values:'
          f'{blu}\n[+] origin -> {red}{origin.isna().any().any()}'
          f'{blu}\n[+] train  -> {red}{train.isna().any().any()}'
          f'{blu}\n[+] test   -> {red}{test.isna().any().any()}')
