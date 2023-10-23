import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from models import LSTM,PointTransformer
from utils.tool import StandardScaler, adjust_learning_rate, MinMaxScaler
from utils.data_loader import MyDataset
from utils.time_features import time_features
import time
from datetime import datetime
import warnings


class LSTMModel():
    def __init__(self, input_len=14, output_len=7, input_size=14, hid_size=8, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.input_size = input_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.cols = ['power',
                     'ele_price',
                     'ser_price',
                     'after_ser_price',
                     'f1',
                     'f2',
                     'f3',
                     'parking_free',
                     'flag',
                     'h3',
                     'ac_equipment_kw',
                     'dc_equipment_kw'
                     ]
        self._min, self._max = None, None
        self.model = self._build_model()

    def _build_model(self):
        model = LSTM(n_lags=self.input_len, input_size=self.input_size, hidden_layer_size=self.hid_size,
                     num_layers=self.num_layers, output_size=self.output_len, dropout=self.dropout)
        # model = PointTransformer(n_lags=self.input_len, input_size=self.input_size, hidden_layer_size=self.hid_size,
        #              num_layers=self.num_layers, output_size=self.output_len, dropout=self.dropout)

        return model

    def load(self, path):
        check = torch.load(path)
        self.model.load_state_dict(check['state_dict'])
        self.check = check

    def fit(self, df, lr=0.001, epochs=100):
        self.model = self.model
        self.model.train()
        train_sets = []
        val_sets = []
        for i in range(500):
            train_sets.append(MyDataset(df, flag='train', cols=self.cols, id_encode=i, input_len=self.input_len,
                                        output_len=self.output_len))
            # val_sets.append(MyDataset(df, flag='val', cols=self.cols, id_encode=i, input_len=self.input_len,
            #                           output_len=self.output_len))
        train_set = ConcatDataset(train_sets)
        # val_set = ConcatDataset(val_sets)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=1)  # 此处的batchsize不等于1会怎么样？  必须等于1 后面计算验证集损失时，要一一对应
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        writer = SummaryWriter('./runs/train_lstm_{}_{}_h{}_nl{}_lr{}_dp{}_ep{}'.format(self.input_len,
                                                                                        self.output_len, self.hid_size,
                                                                                        self.num_layers, lr,
                                                                                        self.dropout, epochs))
        for epoch in range(epochs):
            self.model.train()
            iter = 0
            train_loss = []
            adjust_learning_rate(optimizer, epoch, lr=lr, lradj=1)
            epoch_time = time.time()
            for x, y in train_loader:
                # print(x.shape)    # torch.Size([64, 14, 15])
                optimizer.zero_grad()
                x = x.float()
                y = y.float()
                outputs = self.model(x)
                loss = criterion(outputs.contiguous(), y.contiguous())
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                # if (iter+1)%100==0:
                # print('epoch {0} iters {1}/{2} | loss: {3:.7f}'.format(epoch+1,iter+1,len(train_loader),loss.item()))
                iter += 1
            print('epoch {0} cost time {1}'.format(epoch + 1, time.time() - epoch_time))
            train_l = np.average(train_loss)
            print('--------start to validate-----------')
            # val_l, val_rmse = self.val(val_sets, val_loader, criterion)
            # print("Epoch: {} | Train Loss: {:.7f} valid Loss: {:.7f} Valid RMSE: {:.2f}".format(
            #     epoch + 1, train_l, val_l, val_rmse))

            # writer.add_scalar('valid_loss', val_l, global_step=epoch)
            writer.add_scalar('train_loss', train_l, global_step=epoch)
        save_check_1 = {f'_min_{i}': train_sets[i].scaler._min for i in range(500)}
        save_check_2 = {f'_max_{i}': train_sets[i].scaler._max for i in range(500)}
        save_check = {'state_dict': self.model.state_dict()}
        save_check.update(save_check_1)
        save_check.update(save_check_2)  ##？？？？
        torch.save(save_check, './check/LSTMnet_{}_{}_h{}_lv{}_lr{}_dp{}_ep{}.pkl'.format(self.input_len,
                                                                                          self.output_len,
                                                                                          self.hid_size,
                                                                                          self.num_layers, lr,
                                                                                          self.dropout, epoch + 1))

    def val(self, val_sets, val_loader, criterion):
        self.model.eval()
        losses = []
        mses = []
        with torch.no_grad():
            for i, x, y in val_loader:
                x = x.float()
                y = y.float()
                outputs = self.model(x)
                val_l = criterion(outputs.contiguous(), y.contiguous()).item()
                if val_l <= 1:
                    losses.append(val_l)
                else:
                    print('*' * 20)
                    print(i)
                    print('*' * 20)
                    # import pdb;pdb.set_trace()     outputs.contiguous(),y.contiguous()
                pred = val_sets[i].scaler.inverse_transform(outputs.contiguous())
                # print(i,pred)
                gt = val_sets[i].scaler.inverse_transform(y.contiguous())
                # print(i,gt)
                # print(criterion(pred,gt))
                mses.append(criterion(pred, gt).item())
        # import pdb;pdb.set_trace()
        val_loss = np.average(losses)
        # print(np.average(mses))
        rmse = np.sqrt(np.average(mses))
        return val_loss, rmse
    def val(self, val_sets, val_loader):
        self.model.eval()
        losses = []
        mses = []
        with torch.no_grad():
            for i, x, y in val_loader:
                x = x.float()
                y = y.float()
                outputs = self.model(x)


                pred = val_sets[i].scaler.inverse_transform(outputs.contiguous())
                # print(i,pred)

                # print(i,gt)
                # print(criterion(pred,gt))

        # import pdb;pdb.set_trace()
        val_loss = np.average(losses)
        # print(np.average(mses))
        rmse = np.sqrt(np.average(mses))
        return val_loss, rmse
    def val2(self, val_sets, val_loader):
        self.model.eval()

        with torch.no_grad():
            for i, x, y in val_loader:
                x = x.float()
                y = y.float()
                outputs = self.model(x)
                pred = val_sets.scaler.inverse_transform(outputs.contiguous())

        return pred
    def predict(self, df):
        '''
        Predict next 7 days' power.
        ---
        input
        :param df: dataframe of 32 historical days, shape like 32xC
        ---
        output
        7x1 forecast
        '''
        self.model.eval()
        id = df.loc[0, 'id_encode']
        _min = self.check[f'_min_{id}']
        _max = self.check[f'_max_{id}']
        scaler = MinMaxScaler(_min, _max)
        X_1 = df[self.cols].values
        # 归一化
        X_1 = scaler.transform(torch.tensor(X_1, dtype=torch.float32))

        df['ds'] = pd.to_datetime(df['ds'], format='%Y%m%d')
        df_stamp = pd.DataFrame({'ts': df['ds']})
        X_2 = torch.from_numpy(time_features(df_stamp, timeenc=1, freq='D'))

        X = torch.cat([X_1, X_2], axis=1).type(torch.float32).unsqueeze(0)
        # print(X)   # torch.Size([1, 21, 15])
        with torch.no_grad():
            outputs = self.model(X)
        # 反归一化
        pred = scaler.inverse_transform(outputs.contiguous())
        # print(pred)
        return pred


def predict_all(model):
    '''
    Funtion to generate all test data forecasting results
    ---
    return:
    result_df, wrong stations
    '''
    df = pd.read_csv('train_test_df.csv')
    df = df.fillna(0)
    power = np.array([0.] * 500 * 7)
    power2 = np.array([0.] * 500 * 7)
    result_df = df[df['tag'] == 'test'][['id_encode', 'ds']].reset_index(drop=True)
    probs = []

    for i in range(500):
        try:
            df_train_sta0 = df[(df['tag'] == 'train') & (df['id_encode'] == i)][-28:-7].reset_index(
                drop=True)  # 为什么选择历史数据是32呢？有啥讲究？    没讲究，随便选的   -28:-7

            pred = model.predict(df_train_sta0).detach().cpu().numpy()  # 预测结果会出现零 <class 'numpy.ndarray'>
            # print(pred)
            val_sets = MyDataset(df, flag='val', cols=['power',
                     'ele_price',
                     'ser_price',
                     'after_ser_price',
                     'f1',
                     'f2',
                     'f3',
                     'parking_free',
                     'flag',
                     'h3',
                     'ac_equipment_kw',
                     'dc_equipment_kw'
                     ], id_encode=i, input_len=21,output_len=7)
            val_loader = DataLoader(val_sets, batch_size=1)
            pred2 = model.val2(val_sets,val_loader)
            pred2[pred2 < 0] = 0
            pred[pred < 0] = 0
            power[i * 7:(i + 1) * 7] = pred
            power2[i * 7:(i + 1) * 7] = pred2
                # print(i)
        except:
            probs.append(i)
    result_df['power'] = power
    result_df['power2'] = power2
    return result_df, probs


if __name__ == '__main__':
    model = LSTMModel(input_len=21, output_len=7, input_size=15, hid_size=128, num_layers=2, dropout=0.1)
    df = pd.read_csv('train_test_df.csv', parse_dates=['ds'])
    df = df.fillna(0)

    # train
    # model.fit(df, lr=0.001, epochs=15)

    # test
    model.load('check/LSTMnet_21_7_h128_lv2_lr0.001_dp0.1_ep15.pkl')
    result_df,probs = predict_all(model)
    print(probs)
    result_df.to_csv('./result/result_lstm_new.csv',index=False)