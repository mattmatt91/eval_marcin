import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_dfs():
    df_info = pd.read_csv("info.csv", decimal=',', sep=';')
    df_info.set_index(df_info.columns[0], inplace=True)
    df_info = df_info.T
    df_info.index.rename('name', inplace=True)

    df = pd.read_csv("data.csv")

    # df['time'] = df.index

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

    time_stamps = df['time']
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    X = np.transpose(np.array([df[i].to_list() for i in df.columns]))
    names = df.index
    y = [df_info['sample'][i] for i in df.index]
    target_names = [df_info.loc[i]['sample'] for i in df.index]
    dict_targets = {}
    for name, i in zip(set(target_names), range(len(set(target_names)))):
        dict_targets[name] = i
    return X, y, dict_targets, names, time_stamps


def process_pca(X, y, target_names, names, time_stamps):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    y = [target_names[i] for i in y]
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()


    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)
    loadings = pd.DataFrame(pca.components_.T, columns=[
                            'PC1', 'PC2', 'PC3'], index=time_stamps)
    loadings = loadings.abs()
    loadings_sum = loadings.sum(axis=1)
    loadings['sum'] = loadings_sum
    loadings.to_csv('loadings.csv', decimal=',', sep=";")
    fig = px.line(loadings_sum)
    fig.write_html('loadings.html')
    plot_pca(X_pca, df_info, names)


def plot_pca(X_pca, df_info, names):
    df = prepare_plot_pca(X_pca, df_info, names)
    fig = px.scatter_3d(df, x="df, ", y="y", z='z',
                        color="sample",  hover_data=df.columns)
    fig.write_html('plot_pca.html')
    fig.show()


def prepare_plot_pca(X_pca, df_info, names):
    data = []
    for this_x,  name in zip(X_pca, names):
        data.append(
            {'name': name, 'df, ': this_x[0], 'y': this_x[1], 'z': this_x[2]} | df_info.loc[name].to_dict())
    df = pd.DataFrame(data)
    return df


df, df_info = read_dfs()

# df_melt = create_melt(df, df_info)
# print(df_melt)
# plot_measurements(df_melt)
X, y, target_names, names, time_stamps = prep_pca(df, df_info)
process_pca(X, y, target_names, names, time_stamps)
