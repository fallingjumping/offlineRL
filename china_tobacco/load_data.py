import numpy as np
import pandas as pd
import os


def historical_dataset():
    file_path = './7月工艺过程数据/'
    files = os.listdir(file_path)
    f_len = len(files)
    sta_ = np.empty(shape=(1, 9), dtype=np.float32)
    next_sta_ = np.empty(shape=(1, 9), dtype=np.float32)
    act_ = np.empty(shape=(1, 1), dtype=np.float32)
    rew_ = np.empty(shape=(1, 1), dtype=np.float32)
    terminals_ = np.empty(shape=(1,), dtype=bool)
    for i in range(f_len):
        data = pd.read_csv(os.path.join(
            file_path, files[i]), sep='\t', encoding='utf-16')
        data.drop(columns=['时间', '薄板烘丝牌号', '薄板烘丝批次'], inplace=True)
        sta = np.array(data.loc[:, ['薄板烘丝_定量喂料物料累计量', '薄板烘丝_定量喂料物料流量',
                                    '薄板烘丝_烘丝机二区筒体温度', '薄板烘丝_热风温度', '薄板烘丝_烘丝机出口温度', '薄板烘丝_烘丝机出口水分',
                                    '薄板烘丝_增温增湿蒸汽流量', '薄板烘丝_增温增湿出口物料温度', '薄板烘丝_增温增湿入口水分']])
        act = np.expand_dims(np.array(data.iloc[:, 3]), axis=1)
        next_sta = np.array(data.loc[1:, ['薄板烘丝_定量喂料物料累计量', '薄板烘丝_定量喂料物料流量',
                                          '薄板烘丝_烘丝机二区筒体温度', '薄板烘丝_热风温度', '薄板烘丝_烘丝机出口温度', '薄板烘丝_烘丝机出口水分',
                                          '薄板烘丝_增温增湿蒸汽流量', '薄板烘丝_增温增湿出口物料温度', '薄板烘丝_增温增湿入口水分']])
        (row, dim) = sta.shape
        next_sta = np.append(next_sta, np.empty(shape=(1, dim)), axis=0)
        goal_moisture = 135
        rewards = np.array(
            (-100) * abs(data.loc[:, ['薄板烘丝_烘丝机出口水分']] - goal_moisture))
        terminals = np.array([False] * (row - 1))
        terminals = np.append(terminals, np.array([True]), axis=0)

        sta_ = np.append(sta_, sta, axis=0)
        next_sta_ = np.append(next_sta_, next_sta, axis=0)
        act_ = np.append(act_, act, axis=0)
        rew_ = np.append(rew_, rewards, axis=0)
        terminals_ = np.append(terminals_, terminals, axis=0)
    return {
        'states': sta_,
        'actions': act_,
        'next_states': next_sta_,
        'rewards': rew_,
        'terminals': terminals_
    }


# a = historical_dataset()
# print(a['actions'].shape)
