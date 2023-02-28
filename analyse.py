import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def read_dfs():
    df_info = pd.read_csv("info.csv", decimal=',', sep=';')
    df_info.set_index(df_info.columns[0], inplace=True)
    df_info = df_info.T
    df = pd.read_csv("data.csv")
    df['time'] = df.index
    return df, df_info


def create_melt(df, df_info):
    cols = df.columns
    df_melt = pd.melt(df, id_vars='time', value_vars=cols)
    sample = []
    number = []
    date = []
    for i in df_melt['variable']:
        sample.append(df_info['sample'][i])
        number.append(df_info['number'][i])
        date.append(df_info['date'][i])
    df_melt['sample'] = sample
    df_melt['date'] = date
    df_melt['number'] = number
    return df_melt


def plot_measurements(df_melt):
    fig = px.scatter(df_melt, x="time", y="value", color="sample")
    fig.show()


def prep_pca(df, df_info):
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    X = [df[i].to_list() for i in df.columns]
    y = df.index.to_list()
    target_names = [df_info.loc[i]['sample'] for i in df.index]
    print(len(set(target_names)))
    return X, y, target_names


def process_pca(X, y, target_names):
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["navy", "turquoise", "darkorange",
              "red", "blue", "green", "yellow"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")

    plt.show()


df, df_info = read_dfs()
print(df_info)
exit()
df_melt = create_melt(df, df_info)
# plot_measurements(df_melt)
