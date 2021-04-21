import torch
import numpy as np
import higher
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable

from lstm_learner import LSTM
from copy import deepcopy
from plot_tools import plot_predict

class MAML_Higher_Learner(nn.Module):
    """
    MAML Meta Learner
    """
    def __init__(self, args, base_model_config, device):
        """
        args: update_lr, meta_lr, n_way, k_spt, k_qry, 
              task_num, update_step(default: 1, inner loop), update_step_test

        base_model_config(LSTM): input_size, hidden_size, num_layers, output_size
        device: 'cuda' or 'cpu'
        """
        super(MAML_Higher_Learner, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.clip_val = args.clip_val
        self.base_model_config = base_model_config

        self.device = device

        self.net = LSTM(
            input_size=base_model_config['input_size'], 
            hidden_size=base_model_config['hidden_size'], 
            num_layers=base_model_config['num_layers'], 
            output_size=base_model_config['output_size'],
            device=self.device
            )
        self.meta_opt = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def train(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [batch, setsz, seq_num, seq_len]
        :param y_spt:   [batch, setsz, seq_num]
        :param x_qry:   [batch, querysz, seq_num, seq_len]
        :param y_qry:   [batch, querysz, seq_num]
        :return:
        """
        task_num, _, _, _ = x_spt.size()
        qry_losses = []

        inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.update_lr)
        # loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.L1Loss()

        self.meta_opt.zero_grad()
        # 这里的task_num就是一个batch的task数量
        for i in range(task_num):
            # higher implementation
            with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # 1. run the i-th task and compute loss for k = 0 ~ self.update_steps
                for _ in range(self.update_step):
                    y_pred_i = fnet(x_spt[i])
                    spt_loss = loss_fn(y_pred_i, y_spt[i])
                    diffopt.step(spt_loss)

                # query_set meta backfoward
                y_pred_i_q = fnet(x_qry[i])
                qry_loss = loss_fn(y_pred_i_q, y_qry[i])
                qry_losses.append(qry_loss.detach())

                # update model's meta-parameters to optimize the query
                qry_loss.backward()

        self.meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        return qry_losses.item()

    def fineTunning(self, x_spt, y_spt, x_qry, y_qry, task_i, predict=False, pred_dir=None):
        """
        :param x_spt:   [setsz, seq_num, seq_len]
        :param y_spt:   [setsz, seq_num]
        :param x_qry:   [querysz, seq_num, seq_len]
        :param y_qry:   [querysz, seq_num]
        :return:
        """
        assert len(x_spt.shape) == 3

        ft_net = deepcopy(self.net)
        loss_fn = torch.nn.L1Loss()
        optimizer_ft = torch.optim.SGD(ft_net.parameters(), lr=self.update_lr)
        test_loss = 0

        # non-higher implementation
        for _ in range(self.update_step_test):
            y_pred_spt = ft_net(x_spt)
            spt_loss = loss_fn(y_pred_spt, y_spt)

            optimizer_ft.zero_grad()
            spt_loss.backward()
            # clipping to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(ft_net.parameters(), self.clip_val)
            optimizer_ft.step()
            
        # query loss
        y_pred_qry = ft_net(x_qry)
        qry_loss = loss_fn(y_pred_qry, y_qry)
        test_loss = qry_loss.detach().item()
        
        # prediction if pred is set to be True
        if predict == True:
            ts_pred, ts_ori = self.predictOneStep(ft_net, x_qry, y_qry)
            task_pred_dir = os.path.join(pred_dir, 'meta_test_task_{}'.format(task_i))
            if os.path.exists(task_pred_dir) is False:
                os.makedirs(task_pred_dir)

            for i in range(ts_pred.shape[0]):
                fig_name = os.path.join(task_pred_dir, 'query_{}.png'.format(i + 1))
                plot_predict(y_pred=ts_pred[i], y_true=ts_ori[i], fig_name=fig_name)
        
        return test_loss
    
    def saveParams(self, save_path):
        torch.save(self.state_dict(), save_path)

    def predictOneStep(self, fnet, x, y):
        """
        :param x:           [setsz, seq_num, seq_len]
        :param y:           [setsz, seq_num]
        :return ts_pred:    [setsz, ts_len]
        :return ts_ori:     [setsz, ts_len]
        """
        assert len(x.shape) == 3 and len(y.shape) == 2
        setsz, _, _ = x.size()

        ts_pred = []
        ts_ori = []
        for i in range(setsz):
            ts_pred_i = fnet(x[i].unsqueeze(0))
            ts_pred_i_cpu = ts_pred_i.data.cpu().numpy()
            ts_pred_i_cpu = np.squeeze(ts_pred_i_cpu)
            ts_ori_i_cpu = y[i].data.cpu().numpy()
            ts_pred.append(ts_pred_i_cpu)
            ts_ori.append(ts_ori_i_cpu)
        
        return np.array(ts_pred), np.array(ts_ori)
            
def main():
    pass

if __name__ == '__main__':
    main()