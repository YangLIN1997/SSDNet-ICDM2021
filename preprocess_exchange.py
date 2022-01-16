from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import matplotlib

matplotlib.use('Agg')
import os
import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--L', default=30, type=int,help='Number of input period')
parser.add_argument('--H', default=20, type=int,help='Number of forecasting period')


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd



def prep_data(data, data_mean, data_scale, task='search_', name='train', data2=None):
    input_size = window_size - stride_size
    time_len = data.shape[0]
    total_windows = n_id * (time_len - input_size) // stride_size
    print("windows pre: ", total_windows, "   No of days:", total_windows / n_id)
    x_input = np.zeros((total_windows, window_size, num_covariates), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    for i in range(total_windows // n_id):
        window_start = stride_size * i
        window_end = window_start + window_size
        x_input[i * n_id:(i + 1) * n_id, 0, 0] = (x_input[i * n_id:(i + 1) * n_id, 0, 0] - data_mean) / data_scale
        x_input[i * n_id:(i + 1) * n_id, 1:, 0] = data[window_start:window_end - 1, :, 0].swapaxes(0, 1).reshape(-1,
                                                                                                                 window_size - 1)
        x_input[i * n_id:(i + 1) * n_id, :, 1:] = data[window_start:window_end, :, 1:].swapaxes(0, 1).reshape(-1,
                                                                                                              window_size,
                                                                                                              num_covariates - 1)
        label[i * n_id:(i + 1) * n_id, :] = data[window_start:window_end, :, 0].swapaxes(0, 1).reshape(-1, window_size)
    # zeros_index = np.zeros(x_input.shape[0])
    # for i in range(window_size // stride_size):
    #     var = np.var(x_input[:, i * stride_size + 1:(i + 1) * stride_size, 0] * data_scale[
    #         x_input[:, 0, -1].astype(np.int)].reshape(-1, 1) + data_mean[x_input[:, 0, -1].astype(np.int)].reshape(-1,
    #                                                                                                                1),
    #                  axis=1)
    #     zeros_index += (var < 1e-3)
    # zeros_index = np.where((zeros_index > 0))[0]
    # x_input = np.delete(x_input, zeros_index, axis=0)
    # label = np.delete(label, zeros_index, axis=0)
    print('x_input', x_input.shape)
    # print('x_input', x_input[10,:,0])
    # print('label', label[10,:])

    if data2 is not None:
        time_len = data2.shape[0]
        total_windows = n_id * (time_len - input_size) // stride_size
        print("windows pre2: ", total_windows, "   No of days:", total_windows / n_id)
        x_input2 = np.zeros((total_windows, window_size, num_covariates), dtype='float32')
        label2 = np.zeros((total_windows, window_size), dtype='float32')
        for i in range(total_windows // n_id):
            window_start = stride_size * i
            window_end = window_start + window_size
            x_input2[i * n_id:(i + 1) * n_id, 0, 0] = (x_input2[i * n_id:(i + 1) * n_id, 0, 0] - data_mean) / data_scale
            x_input2[i * n_id:(i + 1) * n_id, 1:, 0] = data2[window_start:window_end - 1, :, 0].swapaxes(0, 1).reshape(
                -1, window_size - 1)
            x_input2[i * n_id:(i + 1) * n_id, :, 1:] = data2[window_start:window_end, :, 1:].swapaxes(0, 1).reshape(-1,
                                                                                                                    window_size,
                                                                                                                    num_covariates - 1)
            label2[i * n_id:(i + 1) * n_id, :] = data2[window_start:window_end, :, 0].swapaxes(0, 1).reshape(-1,
                                                                                                             window_size)
        zeros_index = np.zeros(x_input2.shape[0])
        for i in range(window_size // stride_size):
            var = np.var(x_input2[:, i * stride_size + 1:(i + 1) * stride_size, 0] * data_scale[
                x_input2[:, 0, -1].astype(np.int)].reshape(-1, 1) + data_mean[
                             x_input2[:, 0, -1].astype(np.int)].reshape(-1, 1), axis=1)
            zeros_index += (var < 1e-3)
        zeros_index = np.where((zeros_index > 0))[0]
        x_input2 = np.delete(x_input2, zeros_index, axis=0)
        label2 = np.delete(label2, zeros_index, axis=0)

        x_input = np.concatenate((x_input, x_input2), axis=0)
        label = np.concatenate((label, label2), axis=0)

    prefix = os.path.join(save_path, name + '_')
    np.save(prefix + 'data_' + task + save_name, x_input)
    print(prefix + 'data_' + task + save_name, x_input.shape)

    np.save(prefix + 'mean_' + task + save_name, data_mean[x_input[:, 0, -1].astype(np.int)])
    np.save(prefix + 'scale_' + task + save_name, data_scale[x_input[:, 0, -1].astype(np.int)])
    np.save(prefix + 'label_' + task + save_name, label)


def prepare(task='search_'):
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates, n_id)
    train_data = covariates[:data_frame[:train_end].shape[0]].copy()
    valid_data = covariates[data_frame[:valid_start].shape[0] - 1:data_frame[:valid_end].shape[0]].copy()
    test_data = covariates[data_frame[:test_start].shape[0] - 1:data_frame[:test_end].shape[0]].copy()
    valid_data[:, :, 0] = data_frame[valid_start:valid_end].copy()
    test_data[:, :, 0] = data_frame[test_start:test_end].copy()
    train_data[:, :, 0] = data_frame[train_start:train_end].copy()
    print(test_data.shape)
    # Standardlize data
    data_scale = np.zeros(n_id)
    data_mean = np.zeros(n_id)

    for i in range(n_id):
        st_scaler = StandardScaler()
        st_scaler.fit(train_data[data_start[i]:, i, 0].reshape(-1,1))
        train_data[:, i, 0] = st_scaler.transform(train_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        valid_data[:, i, 0] = st_scaler.transform(valid_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        test_data[:, i, 0] = st_scaler.transform(test_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        data_scale[i] = st_scaler.scale_[0]
        data_mean[i] = st_scaler.mean_[0]
    # Prepare data
    # print(data_mean)
    prep_data(train_data, data_mean, data_scale, task, name='train', data2=None)
    prep_data(valid_data, data_mean, data_scale, task, name='valid', data2=None)
    prep_data(test_data, data_mean, data_scale, task, name='test', data2=None)


def gen_covariates(times, num_covariates, n_id):
    covariates = np.zeros((times.shape[0], n_id, num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, :, 1] = input_time.weekday()
        covariates[i, :, 2] = input_time.month
    for i in range(n_id):
        covariates[:, i, -1] = i
        cov_age = np.zeros((times.shape[0],))
        cov_age[data_start[i]:] = stats.zscore(np.arange(times.shape[0] - data_start[i]))
        covariates[:, i, 3] = cov_age
    for i in range(1,num_covariates-1):
        covariates[:,:,i] = stats.zscore(covariates[:,:,i])
    return covariates


if __name__ == '__main__':

    global save_path
    name = 'exchange_rate.txt'
    save_name = 'exchange_rate'
    num_covariates = 5  # z;time feature;id

    args = parser.parse_args()    
    day_steps = args.H
    window_size = args.L+args.H
    stride_size = args.H
    pred_days = args.L
    given_days = args.H


    save_path = os.path.join('data', save_name)
    csv_path = os.path.join(save_path, name)
    print("csv_path: ", csv_path)
    data_frame = pd.read_csv(csv_path, sep=",",header=None, decimal='.')


    date = pd.date_range("1990", freq="D",periods=10622)
    data_frame = pd.DataFrame(data_frame.values, date[date.dayofweek < 5])
    print('From: ', data_frame.index[0], 'to: ', data_frame.index[-1])
    # data_frame.fillna(0, inplace=True)
    # visualize(data_frame, 365, day_num=20, save_name=save_name)
    n_id = data_frame.shape[1]
    n_day = data_frame.shape[0] / stride_size
    print('total days:', n_day)
    print('total samples:', data_frame.shape[0])
    print('total series:', data_frame.shape[1])

    total_time = data_frame.shape[0]  # 32304
    data_start = (data_frame.values != 0).argmax(axis=0)  # find first nonzero value in each time series
    data_start = (data_start // stride_size) * stride_size

    # For gridsearch
    # train_start = '2011-01-01 00:00:00'
    # train_end = '2014-08-17 23:00:00'
    # valid_start = '2014-08-11 00:00:00'  # need additional 7 days as given info
    # valid_end = '2014-08-24 23:00:00'
    # test_start = '2014-08-18 00:00:00' #need additional 7 days as given info
    # test_end = '2014-08-31 23:00:00'
    # train_start2 = '2014-09-01 00:00:00'
    # train_end2 = '2014-12-31 23:00:00'
    # prepare(task='search_')

    # For inference
    train_start = '1990-01-01 00:00:00'
    # train_end = '2014-08-31 23:00:00'
    train_end = '2013-02-04 00:00:00'
    valid_start = '2013-02-04 00:00:00'  # need additional 7 days as given info
    valid_end = '2015-01-19 00:00:00'
    test_start = '2015-01-19 00:00:00'  # need additional 7 days as given info
    test_end = '2016-12-31 00:00:00'
    prepare(task='')
