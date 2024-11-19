import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# weather 数据集
class Dataset_weather(Dataset):

    def __init__(self,path,seqs=5,preds=1,scale=True) -> None:
        super().__init__()
        self.data_path = path
        self.scale = scale
        self.seqs = seqs # 每个样本中的图片序列长度
        self.preds = preds # 每个样本预测图片的数量
        self.__read_data__() # 这里读取后的数据形状都是(self.channels,self.new_L)即横置的

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(self.data_path,index_col=0) # shape:(52696, 21)
        data_array = df_raw.to_numpy()
        # 截断数据使其为C的倍数
        L, C = data_array.shape
        self.channels = C
        new_L = (L // C ) * C 
        self.new_L = new_L
        trun_data_array = data_array[:new_L,:]
        assert len(trun_data_array) % C == 0
        # 数据缩放
        if self.scale:
            self.scaler.fit(trun_data_array) # 此时数据为竖置
            data = self.scaler.transform(trun_data_array)
        else:
            data = trun_data_array
        
        dataT = data.T
        # print(dataT.shape) (C,new_L)        
        self.data_x = dataT
        # self.data_y = dataT
       
    def __getitem__(self, index):
        C,L = self.channels,self.new_L
        
        seq_x = self.data_x[:,index*C:(index+self.seqs)*C]
        y_true = self.data_x[:,(index+self.seqs)*C:(index+self.seqs+1)*C]

        return seq_x,y_true

    def __len__(self):
        return int((self.new_L / self.channels) - self.seqs)
    
    def inverse_transform(self, data):
        data = data.T # 横置转竖置
        inverse_trans_data = self.scaler.inverse_transform(data)
        # 竖置转横置
        return inverse_trans_data.T

# SMD数据集_pkl格式    
class Dataset_SMD_pkl(Dataset):
    def __init__(self,path,seqs=5,preds=1,scale=False) -> None:
        super().__init__()
        self.data_path = path
        self.scale = scale
        self.seqs = seqs # 每个样本中的图片序列长度
        self.preds = preds # 每个样本预测图片的数量
        self.__read_data__() # 这里读取后的数据形状都是(self.channels,self.new_L)即横置的

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        # 读取pkl格式数据
        with open(self.data_path,'rb') as file:
            data_array = pickle.load(file) # smd_1-1_test.shape:(28479,38)
        # 截断数据使其为C的倍数
        L, C = data_array.shape
        self.channels = C
        new_L = (L // C ) * C 
        self.new_L = new_L
        trun_data_array = data_array[:new_L,:]
        assert len(trun_data_array) % C == 0
        # 数据缩放
        if self.scale:
            self.scaler.fit(trun_data_array) # 此时数据为竖置
            data = self.scaler.transform(trun_data_array)
        else:
            data = trun_data_array
        
        dataT = data.T
        print(f'截断后数据形状:{dataT.shape}')  # (C,new_L)        
        self.data_x = dataT
        # self.data_y = dataT
       
    def __getitem__(self, index):
        C,L = self.channels,self.new_L
        
        seq_x = self.data_x[:,index*C:(index+self.seqs)*C]
        y_true = self.data_x[:,(index+self.seqs)*C:(index+self.seqs+1)*C]

        return seq_x,y_true

    def __len__(self):
        return int((self.new_L / self.channels) - self.seqs)
    
    def inverse_transform(self, data):
        data = data.T # 横置转竖置
        inverse_trans_data = self.scaler.inverse_transform(data)
        # 竖置转横置
        return inverse_trans_data.T
    

class Dataset_MSL(Dataset):
    def __init__(self,path,seqs=5,preds=1,scale=True) -> None:
        super().__init__()
        self.data_path = path
        self.scale = scale
        self.seqs = seqs # 每个样本中的图片序列长度
        self.preds = preds # 每个样本预测图片的数量
        self.__read_data__() # 这里读取后的数据形状都是(self.channels,self.new_L)即横置的

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        data_array = np.load(self.data_path) # shape:(73729, 55) 测试集
        # 截断数据使其为C的倍数
        L, C = data_array.shape
        self.channels = C
        new_L = (L // C ) * C # 73700
        self.new_L = new_L
        trun_data_array = data_array[:new_L,:]
        assert len(trun_data_array) % C == 0
        # 数据缩放
        if self.scale:
            self.scaler.fit(trun_data_array) # 此时数据为竖置
            data = self.scaler.transform(trun_data_array)
        else:
            data = trun_data_array
        
        dataT = data.T
        print(dataT.shape)  # (C,new_L) (55, 73700)
        self.data_x = dataT
        # self.data_y = dataT
       
    def __getitem__(self, index):
        C,L = self.channels,self.new_L
        
        seq_x = self.data_x[:,index*C:(index+self.seqs)*C]
        y_true = self.data_x[:,(index+self.seqs)*C:(index+self.seqs+1)*C]

        return seq_x,y_true

    def __len__(self):
        return int((self.new_L / self.channels) - self.seqs)
    
    def inverse_transform(self, data):
        data = data.T # 横置转竖置
        inverse_trans_data = self.scaler.inverse_transform(data)
        # 竖置转横置
        return inverse_trans_data.T

# SWaT数据集处理已经缩放好的npy数据
class Dataset_SWaT_npy(Dataset):
    def __init__(self,path,seqs=5,preds=1,scale=False) -> None:
        super().__init__()
        self.data_path = path
        self.scale = scale
        self.seqs = seqs # 每个样本中的图片序列长度
        self.preds = preds # 每个样本预测图片的数量
        self.__read_data__() # 这里读取后的数据形状都是(self.channels,self.new_L)即横置的

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        # 读取npy格式数据
        data_array = np.load(self.data_path) # shape(449919,51)
        # 截断数据使其为C的倍数
        L, C = data_array.shape
        self.channels = C
        new_L = (L // C ) * C 
        self.new_L = new_L
        trun_data_array = data_array[:new_L,:]
        assert len(trun_data_array) % C == 0
        # 数据缩放
        if self.scale:
            self.scaler.fit(trun_data_array) # 此时数据为竖置
            data = self.scaler.transform(trun_data_array)
        else:
            data = trun_data_array
        
        dataT = data.T
        print(f'截断后数据形状:{dataT.shape}')  # (C,new_L)        
        self.data_x = dataT
        # self.data_y = dataT
       
    def __getitem__(self, index):
        C,L = self.channels,self.new_L
        
        seq_x = self.data_x[:,index*C:(index+self.seqs)*C]
        y_true = self.data_x[:,(index+self.seqs)*C:(index+self.seqs+1)*C]

        return seq_x,y_true

    def __len__(self):
        return int((self.new_L / self.channels) - self.seqs)
    
    def inverse_transform(self, data):
        data = data.T # 横置转竖置
        inverse_trans_data = self.scaler.inverse_transform(data)
        # 竖置转横置
        return inverse_trans_data.T
    
# SWaT数据集处理已经缩放好的npy数据
class Dataset_PSM_npy(Dataset):
    def __init__(self,path,seqs=5,preds=1,scale=False) -> None:
        super().__init__()
        self.data_path = path
        self.scale = scale
        self.seqs = seqs # 每个样本中的图片序列长度
        self.preds = preds # 每个样本预测图片的数量
        self.__read_data__() # 这里读取后的数据形状都是(self.channels,self.new_L)即横置的

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        # 读取npy格式数据
        data_array = np.load(self.data_path) # shape (87841,25)
        # 截断数据使其为C的倍数
        L, C = data_array.shape
        self.channels = C
        new_L = (L // C ) * C 
        self.new_L = new_L
        trun_data_array = data_array[:new_L,:]
        assert len(trun_data_array) % C == 0
        # 数据缩放
        if self.scale:
            self.scaler.fit(trun_data_array) # 此时数据为竖置
            data = self.scaler.transform(trun_data_array)
        else:
            data = trun_data_array
        
        dataT = data.T
        print(f'截断后数据形状:{dataT.shape}')  # (C,new_L)        
        self.data_x = dataT
        # self.data_y = dataT
       
    def __getitem__(self, index):
        C,L = self.channels,self.new_L
        
        seq_x = self.data_x[:,index*C:(index+self.seqs)*C]
        y_true = self.data_x[:,(index+self.seqs)*C:(index+self.seqs+1)*C]

        return seq_x,y_true

    def __len__(self):
        return int((self.new_L / self.channels) - self.seqs)
    
    def inverse_transform(self, data):
        data = data.T # 横置转竖置
        inverse_trans_data = self.scaler.inverse_transform(data)
        # 竖置转横置
        return inverse_trans_data.T
    

    
