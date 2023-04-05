import matplotlib.pyplot as plt
import seaborn as sns


def box_plot(df, x_col, y_col, output_file_path, baseline=None):
    plt.clf()
    sns.set_theme()

    ax = sns.boxplot(data=df, x=x_col, y=y_col)

    if baseline is not None:
        ax.axvline(baseline, color="gray", linestyle="--")
    ax.set_ylim(0, 1)
    plt.rcParams['xtick.labelsize'] = 8
    plt.tight_layout()

    #plt.xticks(rotation=20)
    plt.savefig(output_file_path)


def curve_plot(df, x_col, y_col, color_group_col, style_group_col, output_file_path):
    plt.clf()
    sns.set_theme()

    ax = sns.lineplot(data=df, x=x_col, y=y_col, hue=color_group_col, style=style_group_col)

    ax.set_ylim(0, 1)
    plt.rcParams['xtick.labelsize'] = 8
    plt.tight_layout()

    #plt.xticks(rotation=20)
    plt.savefig(output_file_path)


def class_distribution_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()

    ax = sns.barplot(data=df, x="label", y="label_count", hue="group")
    plt.xticks(rotation=20)
    plt.savefig(output_file_path)


def feature_prevalence_distribution_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()
    ax = sns.displot(data=df)
    ax.set_xlabels("Prevalence of feature (%)")
    ax.set_ylabels("Number of features")

    plt.rcParams['xtick.labelsize'] = 8
    plt.tight_layout()
    plt.savefig(output_file_path)


def top_k_features_box_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()
    ax = sns.boxplot(data=df, x="value", y="variable")
    ax.set_xlabel("Mean absolute coefficient across iterations")
    ax.set_ylabel("3-mer")
    plt.rcParams['ytick.labelsize'] = 8
    plt.tight_layout()

    plt.savefig(output_file_path)


def feature_imp_by_prevalence_scatter_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()
    ax = sns.scatterplot(data=df, x="prevalence", y="imp", s=4)
    ax.set_xlabel("Prevalence (%)")
    ax.set_ylabel("Mean absolute coefficient across iterations")
    # plt.rcParams['ytick.labelsize'] = 8


    plt.tight_layout()

    plt.savefig(output_file_path)


def validation_scores_multiline_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()
    sns.lineplot(data=df, x="variable", y="value", hue="split")
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(output_file_path)