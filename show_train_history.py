import matplotlib.pylab as plt
import pandas as pd


def load_data_train(model):
    data_path = 'trained_model/result_{}.csv'.format(model)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    print(model, data_path)
    reader = pd.read_csv(data_path)
    epoch_num = len(reader.iloc[:, 0])
    for j in range(epoch_num):
        train_loss.append(reader.iloc[:, 1][j])
        val_loss.append(reader.iloc[:, 2][j])
        train_acc.append(reader.iloc[:, 3][j])
        val_acc.append(reader.iloc[:, 4][j])
    return train_loss, val_loss, train_acc, val_acc


def ax_plot(ax, train_loss, val_loss, train_acc, val_acc, name):
    x = range(len(train_loss))
    ax1, = plt.plot(x, train_acc, color='crimson', linewidth=lw)
    ax2, = plt.plot(x, val_acc, color='steelblue', linewidth=lw)
    ax3, = plt.plot(x, train_loss, color='darkorange', linewidth=lw)
    ax4, = plt.plot(x, val_loss, color='green', linewidth=lw)
    plt.xlabel('Epoch', fontname="Arial", fontsize=fs)
    plt.ylabel("Loss or accuracy", fontname="Arial", fontsize=fs)
    if name == 'unet':
        plt.xlim(0, 200)
    else:
        plt.xlim(0, 300)
    plt.ylim(0, 1)
    plt.legend([ax1, ax2, ax3, ax4],
               ['training accuracy', 'validation accuracy',
                'training loss', 'validation loss'],
               frameon=False, loc='best')
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontname="Arial", fontsize=fs)
    plt.xticks(fontname="Arial", fontsize=fs)
    plt.yticks(fontname="Arial", fontsize=fs)
    ax.tick_params(which='major', width=lw/2)
    ax.tick_params(which='minor', width=lw/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(lw/2)
    ax.spines['left'].set_linewidth(lw/2)
    plt.tight_layout()


if __name__ == '__main__':
    lw = 0.75
    fs = 8
    train_loss1, val_loss1, train_acc1, val_acc1 = load_data_train('unet')
    train_loss2, val_loss2, train_acc2, val_acc2 = load_data_train('LeNet')
    plt.figure(figsize=(6, 2.5))

    ax1 = plt.subplot(1, 2, 1)
    ax_plot(ax1, train_loss1, val_loss1, train_acc1, val_acc1, 'unet')
    # ax1.set_title('  U-Net', fontname="Arial", fontsize=fs+2,
    #               fontweight='bold', loc='center')

    ax2 = plt.subplot(1, 2, 2)
    ax_plot(ax2, train_loss2, val_loss2, train_acc2, val_acc2, 'LeNet')
    # ax2.set_title('  LeNet', fontname="Arial", fontsize=fs+2,
    #               fontweight='bold', loc='center')
    plt.savefig('figure/train.png')
    plt.savefig('figure/train.pdf')
    plt.show()