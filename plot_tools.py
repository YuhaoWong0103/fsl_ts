import os
import matplotlib.pyplot as plt

def plot_loss(losses, plot_dir):
    if os.path.exists(plot_dir) is False:
        os.makedirs(plot_dir)
    
    fig_name = os.path.join(plot_dir, 'loss.png')
    plt.plot(losses, label="loss")
    plt.legend()
    plt.savefig(fig_name)
    plt.close()

def plot_predict(y_pred, y_true, fig_name):
    plt.plot(y_true, label='y_true')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.savefig(fig_name)
    plt.close()