import numpy as np
import torch
import os
import argparse
import logging

from collections import OrderedDict
from fsl_ts_dataloader import poolRead, getBatchTask
from maml_higher_learner import MAML_Higher_Learner
from plot_tools import plot_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s line:%(lineno)d - %(message)s")

def main(args):
    lstm_config = {
        'input_size': 1, 
        'hidden_size': 64, 
        'num_layers': 1, 
        'output_size': 1, 
    }

    train_flag = args.train
    eval_flag = args.eval
    test_flag = args.test
    pred_flag = args.pred

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    # create logs dir
    if os.path.exists(args.logs_dir) is False:
        os.makedirs(args.logs_dir)

    # RNN can't use cudnn to compute second-order gradients
    with torch.backends.cudnn.flags(enabled=False):
        maml = MAML_Higher_Learner(args, lstm_config, device).to(device)
        meta_train, meta_test = poolRead()

        if train_flag == True:
            # /------meta train------/
            logging.info(" ------ meta training start ------ ")
            train_losses = []
            train_history_loss = []
            eval_losses = []
            for step in range(args.epoch):
                x_spt, y_spt, x_qry, y_qry = getBatchTask(meta_train, batch_num=args.task_num)
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                step_loss = maml.train(x_spt, y_spt, x_qry, y_qry)
                train_history_loss.append(step_loss)

                if step % args.echo_step == 0:
                    train_history_mean_loss = np.array(train_history_loss).mean()
                    train_losses.append(train_history_mean_loss)
                    logging.info('step: {}, train_loss = {}'.format(step, train_history_mean_loss))
                    train_history_loss = []

                # /------meta evaluate------/
                # 这里更正确的做法应该再划分一个meta_eval来做验证，这里因为不做网络微调所以直接取了meta_test
                if step % args.eval_step == 0:
                    eval_history_loss = []
                    logging.info(" ------ meta evaluating start ------ ")
                    # debug setting: using meta train to make sure meta_train loss is converge
                    x_ev_spt, y_ev_spt, x_ev_qry, y_ev_qry = getBatchTask(meta_test, batch_num=args.task_num_eval)
                    x_ev_spt, y_ev_spt, x_ev_qry, y_ev_qry = torch.from_numpy(x_ev_spt).to(device), \
                                                             torch.from_numpy(y_ev_spt).to(device), \
                                                             torch.from_numpy(x_ev_qry).to(device), \
                                                             torch.from_numpy(y_ev_qry).to(device)
                    task_num_eval = x_ev_spt.shape[0]
                    for i in range(task_num_eval):
                        eval_loss = maml.fineTunning(x_ev_spt[i], y_ev_spt[i], x_ev_qry[i], y_ev_qry[i], task_i=i)
                        eval_history_loss.append(eval_loss)
                    
                    mean_eval_losses = np.array(eval_history_loss).mean()
                    eval_losses.append(mean_eval_losses)
                    logging.info("mean eval loss on {} meta eval tasks = {}".format(task_num_eval, mean_eval_losses))
                    
            # plot train loss
            train_plot_dir = os.path.join(
                args.figure_dir, 'train_loss_epoch-{}_metalr-{}_updatelr-{}'.format(
                args.epoch, args.meta_lr, args.update_lr
            ))
            plot_loss(np.array(train_losses), train_plot_dir)
            # plot eval loss
            eval_plot_dir = os.path.join(
                args.figure_dir, 'eval_loss_epoch-{}_metalr-{}_updatelr-{}'.format(
                args.epoch, args.meta_lr, args.update_lr
            ))
            plot_loss(np.array(eval_losses), eval_plot_dir)

            # save init_params
            if os.path.exists(args.model_params_dir) is False:
                    os.makedirs(args.model_params_dir)
            save_path = os.path.join(
                args.model_params_dir, 'maml_init_params_epoch-{}_metalr-{}_updatelr-{}.pt'.format(
                args.epoch, args.meta_lr, args.update_lr
            ))
            maml.saveParams(save_path)

        if test_flag == True:
            # /------meta test------/
            logging.info(" ------ meta testing start ------ ")
            if args.check_point is not None:
                maml_params_file = os.path.join(args.model_params_dir, args.check_point)
                maml.load_state_dict(torch.load(maml_params_file))
            
            # debug setting: using meta train to make sure meta_train loss is converge
            x_test_spt, y_test_spt, x_test_qry, y_test_qry = getBatchTask(meta_test, batch_num=args.task_num_test)
            x_test_spt, y_test_spt, x_test_qry, y_test_qry = torch.from_numpy(x_test_spt).to(device), \
                                                             torch.from_numpy(y_test_spt).to(device), \
                                                             torch.from_numpy(x_test_qry).to(device), \
                                                             torch.from_numpy(y_test_qry).to(device)
            task_num_test = x_test_spt.shape[0]
            test_losses = []
            for i in range(task_num_test):
                pred_dir = os.path.join(args.figure_dir, 'meta_test_query_predict_epoch-{}_metalr-{}_updatelr-{}'.format(
                    args.epoch, args.meta_lr, args.update_lr
                ))
                if args.check_point is not None:
                    pred_dir = os.path.join(pred_dir, '_with_checkpoint')
                else:
                    pred_dir = os.path.join(pred_dir, '_without_checkpoint')

                test_loss = maml.fineTunning(x_test_spt[i], y_test_spt[i], x_test_qry[i], y_test_qry[i], 
                                             task_i=i + 1, predict=pred_flag, pred_dir=pred_dir)
                logging.info("test loss on meta test task({}) = {}".format(i + 1, test_loss))
                test_losses.append(test_loss)
            
            mean_test_losses = np.array(test_losses).mean()
            logging.info("mean test loss on {} meta test tasks = {}".format(task_num_test, mean_test_losses))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--no_cuda', action="store_true", help="set this parameter not to use gpu")
    argparser.add_argument('--train', action="store_true", help="set this parameter to train meta learner")
    argparser.add_argument('--eval', action="store_true", help="set this parameter to evaluate meta learner")
    argparser.add_argument('--test', action="store_true", help="set this parameter to test meta learner")
    argparser.add_argument('--pred', action="store_true", help="set this parameter to predict query set from meta test")
    argparser.add_argument('--echo_step', type=int, help='steps to echo a train loss', default=10)
    argparser.add_argument('--eval_step', type=int, help='steps to echo a eval loss', default=100)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--n_way', type=int, help='n way', default=3)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=2)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=8)
    argparser.add_argument('--task_num', type=int, help='meta_train batch size, namely task_num', default=16)
    argparser.add_argument('--task_num_eval', type=int, help='meta_eval batch size, namely task_num_eval', default=16)
    argparser.add_argument('--task_num_test', type=int, help='meta_test batch size, namely task_num_test', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--clip_val', type=float, help='clipping value to avoid gradient explosion', default=1.0)

    argparser.add_argument('--model_params_dir', type=str, 
                            help='path to store model parameters', default=os.path.join('fsl_model_params', 'maml'))
    argparser.add_argument('--figure_dir', type=str, 
                            help='path to store maml figures', default=os.path.join('fsl_figures', 'maml'))
    argparser.add_argument('--check_point', type=str, 
                            help='the check point model params of maml learner in model_params_dir', default=None)
    argparser.add_argument('--logs_dir', type=str, 
                            help='path to echo meta training/testing logs', default=os.path.join('fsl_logs', 'maml'))

    args = argparser.parse_args()
    logging.info(args)

    main(args)