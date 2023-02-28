import pandas as pd
import os


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def read_measurement(path):
    path_measurement = os.path.join(directory, path, 'tic_front.csv')
    if os.path.isfile(path_measurement):
        try:
            info = extract_info(path)
            print(info)
            df = pd.read_csv(path_measurement, delimiter=',',
                             decimal='.', header=2)
            df.columns = ['time', info['sample']]
            df[df.columns[0]] = df[df.columns[0]].round(decimals=2)
            time = []
            values = []
            for i in pd.unique(df[df.columns[0]]):
                time.append(i)
                values.append(df[df[df.columns[0]] == i][df.columns[1]].mean())
            df_new = pd.DataFrame({'time': time, path: values})
            df_new.set_index('time', inplace=True)
            return True, df_new, info
        except:
            print('could not read ' + measurement + 's info file')
        return True, None, None
    else:
        print('could not read ' + measurement)
        return True, None, None


def extract_info(path):
    info = {}
    date = path.split(' ')[0]
    info['date'] = date
    number = path.split(' ')[1].split('_')[0]
    info['number'] = number
    sample = path.split(' ')[1].split('_')[1]
    info['sample'] = sample
    print(info)
    return info


def merge_dfs(dfs):
    df_result = pd.concat(dfs, axis=1)
    df_result.sort_index()
    return df_result


directory = 'F:\\results'
measurements = get_immediate_subdirectories(directory)
dfs = []
infos = {}
for measurement in measurements:

    flag, df, info = read_measurement(measurement)
    if flag:
        dfs.append(df)
        infos[measurement] = info
df_info = pd.DataFrame(infos)
df_info.to_csv('info.csv', header=True, index=True)
df = merge_dfs(dfs)
df = df[~df.isnull().any(axis=1)]
df.to_csv('data.csv', header=True, index=True)
