import matplotlib.pylab as plt
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pandas import DataFrame


def load_data(R, part):
    path = 'testing_set/{}R'.format(R)
    measure_path = 'testing_set/{}R_measure'.format(R)
    filelist = os.listdir(measure_path)
    length, width, label, elg = [], [], [], []
    image_name = []
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(path + '/' + item))
        tot_num = len(sub_filelist)
        tot_num = round(tot_num / part)
        # print(tot_num)
        file_name = measure_path + '/' + item
        for i in range(tot_num):
            data_name = file_name + '/' + str(i) + '_process4.csv'
            # print(data_name)
            if os.path.exists(data_name):
                reader = pd.read_csv(data_name)
                for j in range(len(reader.iloc[:, 3])):
                    width.append(min(reader.iloc[:, 3][j],
                                     reader.iloc[:, 4][j]))
                    length.append(max(reader.iloc[:, 3][j],
                                      reader.iloc[:, 4][j]))
                    elg.append(1 / reader.iloc[:, 5][j])
                    label.append(reader.iloc[:, 7][j])
            image_name.append(data_name)
    return width, length, label, elg, image_name


def load_data_hand(R):
    data_path = 'testing_set/figure/{}R.csv'.format(R)
    length, width, elg = [], [], []
    reader = pd.read_csv(data_path)
    for j in range(len(reader.iloc[:, 0])):
        width.append(min(reader.iloc[:, 1][j],
                         reader.iloc[:, 2][j]))
        length.append(max(reader.iloc[:, 1][j],
                          reader.iloc[:, 2][j]))
        elg.append(reader.iloc[:, 3][j])
    return width, length, elg


def get_y_limit(n):
    if max(n) > 119:
        y_limit = int(max(n) - max(n) % 30 + 30)
        y_major = 30
    elif max(n) > 69:
        y_limit = int(max(n) - max(n) % 20 + 20)
        y_major = 20
    elif max(n) > 29:
        y_limit = int(max(n) - max(n) % 10 + 10)
        y_major = 10
    else:
        y_limit = int(max(n) - max(n) % 5 + 5)
        y_major = 5
    return y_limit, y_major


def ax_plot(ax, data, bins, ran, x_low, x_up, x_name, fc, x_major, x_minor):
    n, bins, patches = plt.hist(data, bins=bins, range=ran, histtype='bar', linewidth=lw,
                                facecolor=fc, edgecolor='k', fill=True, )
    plt.xlim(x_low, x_up)
    y_up, y_major = get_y_limit(n)
    plt.ylim(0, y_up)
    plt.xlabel(x_name, fontname="Arial", fontsize=fs)
    plt.ylabel("Counts", fontname="Arial", fontsize=fs)
    plt.xticks(fontname="Arial", fontsize=fs)
    plt.yticks(fontname="Arial", fontsize=fs)
    ax.tick_params(which='major', width=lw)
    ax.tick_params(which='minor', width=lw)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    xmajorLocator = MultipleLocator(x_major)
    ax.xaxis.set_major_locator(xmajorLocator)
    xminorLocator = MultipleLocator(x_minor)
    ax.xaxis.set_minor_locator(xminorLocator)
    ymajorLocator = MultipleLocator(y_major)
    ax.yaxis.set_major_locator(ymajorLocator)
    plt.tight_layout()


def ax_box(ax, data, y_name, y_low, y_up):
    labels = 'Manual', "CNN", "CNN/2", "CNN/4"
    plt.boxplot(data, labels=labels,
                boxprops={'color': 'cholate'})
    plt.ylabel(y_name, fontname="Arial", fontsize=fs)
    plt.ylim(y_low, y_up)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontname="Arial", fontsize=fs)
    plt.yticks(fontname="Arial", fontsize=fs)
    plt.tight_layout()


def prepare_violin(listm, list1, list2, list4):
    df: DataFrame = pd.DataFrame(columns=["Manual"])
    for i in range(len(listm)):
        df.loc[i, 'Manual'] = listm[i]
    for i in range(len(list1)):
        df.loc[i, 'CNN'] = list1[i]
    for i in range(len(list2)):
        df.loc[i, 'CNN/2'] = list2[i]
    for i in range(len(list4)):
        df.loc[i, 'CNN/4'] = list4[i]
    return df


def ax_violin(ax, data, color, y_name, y_low, y_up):
    my_pal = {"Manual": color[0], "CNN": color[1],
              "CNN/2": color[2], "CNN/4": color[3]}
    sns.violinplot(data=data, palette=my_pal, width=0.5, linewidth=1)
    # sns.violinplot(data=data)
    plt.ylabel(y_name, fontname="Arial", fontsize=fs)
    plt.ylim(y_low, y_up)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontname="Arial", fontsize=fs)
    plt.yticks(fontname="Arial", fontsize=fs)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    plt.tight_layout()


def get_label_data(lab_list, len_list, elg_list):
    length1, elg1 = [], []
    length2, elg2 = [], []
    length3, elg3 = [], []
    for i in range(len(lab_list)):
        if lab_list[i] == 'co':
            length1.append(len_list[i])
            elg1.append(elg_list[i])
        elif lab_list[i] == 'prism':
            length2.append(len_list[i])
            elg2.append(elg_list[i])
        else:
            length3.append(len_list[i])
            elg3.append(elg_list[i])
    return length1, elg1, length2, elg2, length3, elg3


def ax_label(lab_list, len_list, elg_list):
    length1, elg1, length2, elg2, length3, elg3 = get_label_data(lab_list, len_list, elg_list)
    ax1 = plt.scatter(elg1, length1, c='None', edgecolors='peru', marker='d', clip_on=False)
    ax2 = plt.scatter(elg2, length2, c='None', edgecolors='forestgreen', marker='s', clip_on=False)
    ax3 = plt.scatter(elg3, length3, c='None', edgecolors='dodgerblue', marker='v', clip_on=False)
    plt.xlim(0.2, 1)
    plt.ylim(0, 300)
    plt.xlabel("Axial ratio (width/length)", fontname="Arial", fontsize=fs)
    plt.ylabel("Length (nm)", fontname="Arial", fontsize=fs)
    plt.legend((ax1, ax2, ax3), ('cuboctahedron', 'prism', 'bullet'), frameon=False,
               loc=10, ncol=3, bbox_to_anchor=(0.52, 1.06),
               handletextpad=0., fontsize=fs, prop="Arial")
    plt.xticks(fontname="Arial", fontsize=fs)
    plt.yticks(fontname="Arial", fontsize=fs)
    plt.tight_layout()


def ax_pie(lab_list, len_list, elg_list):
    length1, _, length2, _, length3, _ = get_label_data(lab_list, len_list, elg_list)
    label_data = [len(length1) / len(len_list),
                  len(length2) / len(len_list),
                  len(length3) / len(len_list)]
    labels = 'cuboctahedron', 'prism', 'bullet'
    colors = 'peru', 'forestgreen', 'dodgerblue'
    plt.pie(label_data, labels=labels, colors=colors, autopct='%1.1f%%',
            textprops={'fontname': "Arial", 'fontsize': fs})
    plt.tight_layout()


def plot_data(R):
    wid_list1, len_list1, lab_list1, elg_list1, image_name1 = load_data(R, 1)
    wid_list2, len_list2, _, elg_list2, image_name2 = load_data(R, 2)
    wid_list4, len_list4, _, elg_list4, image_name4 = load_data(R, 4)
    wid_list, len_list, elg_list = load_data_hand(R)
    print('******{}R******'.format(R))
    print('particle#', len(wid_list), len(wid_list1), len(wid_list2), len(wid_list4))
    print('image#', len(image_name1), len(image_name2), len(image_name4))
    data_len = [len_list, len_list1, len_list2, len_list4]
    data_elg = [elg_list, elg_list1, elg_list2, elg_list4]
    data_len_violin = prepare_violin(len_list, len_list1, len_list2, len_list4)
    data_elg_violin = prepare_violin(elg_list, elg_list1, elg_list2, elg_list4)
    color = ['chocolate', 'steelblue', 'seagreen', 'grey']
    plt.figure(figsize=(fig_size, fig_size))

    ax1 = plt.subplot(4, 4, 1)
    ax_plot(ax1, len_list, 40, [0, 200], 0, 200,
            'Length (nm)', color[0], 50, 5)
    ax1.set_title('(a)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax2 = plt.subplot(4, 4, 2)
    ax_plot(ax2, len_list1, 40, [0, 200], 0, 200,
            'Length (nm)', color[1], 50, 5)
    ax2.set_title('(b)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax3 = plt.subplot(4, 4, 3)
    ax_plot(ax3, len_list2, 40, [0, 200], 0, 200,
            'Length (nm)', color[2], 50, 5)
    ax3.set_title('(c)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax4 = plt.subplot(4, 4, 4)
    ax_plot(ax4, len_list4, 40, [0, 200], 0, 200,
            'Length (nm)', color[3], 50, 5)
    ax4.set_title('(d)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax5 = plt.subplot(4, 4, 5)
    ax_plot(ax5, elg_list, 32, [0.2, 1], 0.2, 1,
            'Axial ratio (width/length)', color[0], 0.25, 0.025)
    ax5.set_title('(e)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax6 = plt.subplot(4, 4, 6)
    ax_plot(ax6, elg_list1, 32, [0.2, 1], 0.2, 1,
            'Axial ratio (width/length)', color[1], 0.25, 0.025)
    ax6.set_title('(f)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax7 = plt.subplot(4, 4, 7)
    ax_plot(ax7, elg_list2, 32, [0.2, 1], 0.2, 1,
            'Axial ratio (width/length)', color[2], 0.25, 0.025)
    ax7.set_title('(g)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax8 = plt.subplot(4, 4, 8)
    ax_plot(ax8, elg_list4, 32, [0.2, 1], 0.2, 1,
            'Axial ratio (width/length)', color[3], 0.25, 0.025)
    ax8.set_title('(h)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    # *************************box
    # ax9 = plt.subplot(4, 2, 5)
    # ax_box(ax9, data_len, 'Length (nm)', 0, 300)
    # ax9.set_title('I', fontname="Arial", fontsize=fs + 2,
    #               fontweight='regular', loc='left')
    #
    # ax10 = plt.subplot(4, 2, 6)
    # ax_box(ax10, data_elg, 'Axial ratio (width/length)', 0.2, 1)
    # ax10.set_title('J', fontname="Arial", fontsize=fs + 2,
    #                fontweight='regular', loc='left')

    # *************************violin
    ax9 = plt.subplot(4, 2, 5)
    ax_violin(ax9, data_len_violin, color, 'Length (nm)', 0, 300)
    ax9.set_title('(i)', fontname="Arial", fontsize=fs + 2,
                  fontweight='regular', loc='left')

    ax10 = plt.subplot(4, 2, 6)
    ax_violin(ax10, data_elg_violin, color, 'Axial ratio (width/length)', 0.2, 1)
    ax10.set_title('(j)', fontname="Arial", fontsize=fs + 2,
                   fontweight='regular', loc='left')

    gs = gridspec.GridSpec(4, 4)
    ax11 = plt.subplot(gs[3, :3])
    ax_label(lab_list1, len_list1, elg_list1)
    ax11.set_title('(k)', fontname="Arial", fontsize=fs + 2,
                   fontweight='regular', loc='left')

    ax12 = plt.subplot(gs[3, 3])
    ax_pie(lab_list1, len_list1, elg_list1)
    ax12.set_title('(l)', fontname="Arial", fontsize=fs + 2,
                   fontweight='regular', loc='left')

    plt.savefig('figure/{}R.png'.format(R))
    plt.savefig('figure/{}R.pdf'.format(R))
    # plt.show()


if __name__ == '__main__':
    lw = 0.4
    fs = 12
    fig_size = 10
    plot_data(16)
