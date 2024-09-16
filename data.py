import glob
import numpy as np
import os
import pandas as pd
import torch


def load_data(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    dfs = []
    for filename in files:
        df = pd.read_csv(filename)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


class JSTDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df,
        target_column,
        zcm_columns,
        pcm_columns,
        speech_columns,
        zcm_transform = None,
        pcm_transform = None,
        speech_transform = None
    ):
        self.df = df
        self.target_column = target_column
        self.zcm_columns = zcm_columns 
        self.pcm_columns = pcm_columns
        self.speech_columns = speech_columns

        self.zcm_transform = zcm_transform
        self.pcm_transform = pcm_transform
        self.speech_transform = speech_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data from DAMS columns and convert to numpy array
        labels = self.df.loc[idx, self.target_column].values.astype(np.float32)

        # Get data from feature columns and convert to numpy array
        zcm_features = self.df.loc[idx, self.zcm_columns].values.astype(np.float32)
        pcm_features = self.df.loc[idx, self.pcm_columns].values.astype(np.float32)
        speech_features = self.df.loc[idx, self.speech_columns].values.astype(np.float32)

        timestamp = self.df.loc[idx, "time_stamp"].values[0]

        if self.zcm_transform is not None:
            zcm_features = self.zcm_transform(zcm_features)
        if self.pcm_transform is not None:
            pcm_features = self.pcm_transform(pcm_features)
        if self.speech_transform is not None:
            speech_features = self.speech_transform(speech_features)

        return {
            "zcm": zcm_features,
            "pcm": pcm_features,
            "speech": speech_features,
            "labels": labels,
            "subject": self.df.loc[idx, "ID"],
            "timestamp": timestamp
        }

if __name__ == '__main__':
    # df = load_data("/nas/staff/data_work/Meishu/JST_2023/724data/train")
    df = pd.read_csv("/nas/staff/data_work/Meishu/JST_2023/724data/train/split_normalized_ag_0122.csv")
    print(df.head())
    dataset = JSTDataset(df, ["DAMS1", "DAMS2"])
    print(dataset.zcm_columns)
    print(dataset.pcm_columns)
    print(len(dataset.speech_columns))
    x = dataset[0]
    print(x['zcm'].shape)