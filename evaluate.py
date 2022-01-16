import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import utils
import model.Models as Models_Sanyo
from dataloader import *
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model.Models as net_Sanyo

logger = logging.getLogger('SSDNet.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'

seed = 0

# use GPU if available
cuda_exist = torch.cuda.is_available()
# Set random seeds for reproducible experiments if necessary
if seed >= 0:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True, plot=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
        # plot_batch = np.random.randint(len(test_loader)-1)
        plot_batch = 0
        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)

        for i, (test_batch, scale, mean, labels) in enumerate((test_loader)):
            test_batch = test_batch.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[0]
            scale = scale.to(torch.float32).to(params.device)
            mean = mean.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device)  # scaled
            trend = torch.zeros(batch_size, labels.shape[1], device=params.device)  # scaled
            seasonality = torch.zeros(batch_size, labels.shape[1], device=params.device)  # scaled
            Sigma = torch.zeros(batch_size, labels.shape[1], device=params.device)  # scaled

            if test_batch.shape[1] == params.n_position:
                mu,alpha,sigma,dec_enc_attn = model.test(test_batch[:, 0:params.test_predict_start, :].clone(),
                                test_batch[:, params.test_predict_start:, :].clone())
            else:
                sigma = torch.zeros(batch_size, labels[:, params.test_predict_start:].shape[1], device=params.device)
                mu = torch.zeros(batch_size, labels[:, params.test_predict_start:].shape[1], device=params.device)
                alpha = torch.zeros(batch_size, labels[:, params.test_predict_start:].shape[1], 24,device=params.device)
                dec_enc_attn = torch.zeros(batch_size, params.n_head,labels[:, params.test_predict_start:].shape[1], labels[:, :params.test_predict_start].shape[1],device=params.device)
                for i in range(7):
                    mu[:,i*params.predict_steps:(i+1)*params.predict_steps],alpha[:,i*(params.predict_steps):(i+1)*(params.predict_steps)],\
                    sigma[:,i*params.predict_steps:(i+1)*params.predict_steps],dec_enc_attn[:,:,i*params.predict_steps:(i+1)*params.predict_steps] \
                                    = model.test(test_batch[:, i*params.predict_steps:params.test_predict_start+i*params.predict_steps, :].clone(), \
                                    test_batch[:, params.test_predict_start+i*params.predict_steps:params.test_predict_start+(i+1)*params.predict_steps, :].clone())
                    if i<6:
                        test_batch[:, (i+1)*params.predict_steps:(i+2)*params.predict_steps, 0]=mu[:,i*params.predict_steps:(i+1)*params.predict_steps]

            if params.n_id==0:
                scale_o = scale[0,0]
                mean_o = mean[0,0]
            else:
                scale_o = scale.reshape(-1,1)
                mean_o = mean.reshape(-1,1)

            sample_mu = scale_o * mu + mean_o
            trend[:,params.test_predict_start:] = scale_o * alpha[:,:,0] + mean_o
            seasonality[:,params.test_predict_start:] = scale_o * alpha[:,:,1]
            Sigma[:,params.test_predict_start:] = scale_o * sigma
            labels = scale_o * labels + mean_o
            labels[labels<0]=0
            gaussian = torch.distributions.normal.Normal(sample_mu, Sigma[:, params.test_predict_start:] )
            sample_j = gaussian.icdf(torch.tensor(0.9))

            # numerator, denominator=net_Sanyo.accuracy_ROU(0.1, gaussian.icdf(torch.tensor(0.1)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.1:',numerator/denominator)
            # print('p0.1:',torch.mean((gaussian.icdf(torch.tensor(0.1))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.2, gaussian.icdf(torch.tensor(0.2)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.2:',numerator/denominator)
            # print('p0.2:',torch.mean((gaussian.icdf(torch.tensor(0.2))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.3, gaussian.icdf(torch.tensor(0.3)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.3:',numerator/denominator)
            # print('p0.3:',torch.mean((gaussian.icdf(torch.tensor(0.3))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.4, gaussian.icdf(torch.tensor(0.4)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.4:',numerator/denominator)
            # print('p0.4:',torch.mean((gaussian.icdf(torch.tensor(0.4))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.6, gaussian.icdf(torch.tensor(0.6)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.6:',numerator/denominator)
            # print('p0.6:',torch.mean((gaussian.icdf(torch.tensor(0.6))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.7, gaussian.icdf(torch.tensor(0.7)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.7:',numerator/denominator)
            # print('p0.7:',torch.mean((gaussian.icdf(torch.tensor(0.7))> labels[:, params.test_predict_start:]).type(torch.float)))
            # numerator, denominator=net_Sanyo.accuracy_ROU(0.8, gaussian.icdf(torch.tensor(0.8)), labels[:, params.test_predict_start:], relative=params.relative_metrics)
            # print('p0.8:',numerator/denominator)
            # print('p0.8:',torch.mean((gaussian.icdf(torch.tensor(0.8))> labels[:, params.test_predict_start:]).type(torch.float)))

            raw_metrics = utils.update_metrics(raw_metrics, input_mu, sample_mu, Sigma[:, params.test_predict_start:], sample_j, labels, params.test_predict_start,
                                               relative=params.relative_metrics, sample=sample)
            if plot == True and i == plot_batch:
                sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start,
                                                   relative=params.relative_metrics, sample=sample)
                # select 10 from samples with highest error and 10 from the rest
                top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
                chosen = set(top_10_nd_sample.tolist())
                all_samples = set(range(batch_size))
                not_chosen = np.asarray(list(all_samples - chosen))
                top_10_nd_sample = (-sample_metrics['ND']).argsort()[:10]  # hard coded to be 10
                bot_10_nd_sample = (sample_metrics['ND']).argsort()[:10]
                if batch_size < 100:  # make sure there are enough unique samples to choose top 10 from
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
                else:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
                if batch_size < 12:  # make sure there are enough unique samples to choose bottom 90 from
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
                else:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
                top_5_nd_sample = (-sample_metrics['ND']).argsort()[:5]  # hard coded to be 10
                bot_5_nd_sample = (sample_metrics['ND']).argsort()[:5]
                combined_sample = np.concatenate((top_5_nd_sample, bot_5_nd_sample))

                label_plot = labels[combined_sample].data.cpu().numpy()
                predict_mu = sample_mu[combined_sample].data.cpu().numpy()
                plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
                plot_Sigma = Sigma[combined_sample].data.cpu().numpy()
                plot_trend = trend[combined_sample].data.cpu().numpy()
                plot_seasonality = seasonality[combined_sample].data.cpu().numpy()
                plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
                plot_eight_windows(params.plot_dir, plot_mu, np.sqrt(plot_Sigma), label_plot, params.test_window, params.test_predict_start,
                                   plot_num, plot_metrics,plot_trend,plot_seasonality, sampling=sample)

            summary_metric = utils.final_metrics(raw_metrics, sample=sample)
            # if plot == True:
            metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
            if plot == True:
                logger.info('- test metrics: ' + metrics_string)
            else:
                logger.info('- valid metrics: ' + metrics_string)

            with open(os.path.join(params.model_dir, 'results.npy'), 'wb') as f:
                np.save(f, labels.data.cpu().numpy())
                np.save(f, input_mu.data.cpu().numpy())
                np.save(f, sample_mu.data.cpu().numpy())
                np.save(f, Sigma.data.cpu().numpy())
                np.save(f, trend.data.cpu().numpy())
                np.save(f, seasonality.data.cpu().numpy())
                np.save(f, dec_enc_attn.data.cpu().numpy())
                np.save(f, test_batch.data.cpu().numpy())

    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       trend=0,seasonality=0,
                       sampling=False
                       ):
    # window_size = 24*14
    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 11
    ncols = 1
    ax = f.subplots(nrows, ncols)
    for k in range(nrows):
        if k == 5:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 5 and bottom 5', fontsize=10)
            continue
        m = k if k < 5 else k - 1
        ax[k].plot(x[predict_start:], predict_values[m,predict_start:], color='r', label='y_{hat}')
        ax[k].plot(x[predict_start:], trend[m,predict_start:], color='y', label='trend')
        ax[k].plot(x[predict_start:], seasonality[m,predict_start:], color='g', label='seasonality')
        # ax[k].plot(x, irregular[m], color='c', label='irregular')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='b', label='y')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')
        ax[k].legend()
        # metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})
        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
                           f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'
        ax[k].set_title(plot_metrics_str, fontsize=10)
    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = Models_Sanyo.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
