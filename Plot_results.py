import os
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from matplotlib.lines import Line2D
from matplotlib.patches import *
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics

def star_bar(ax, x, height, color, n_stars=25):

    if height <= 0:
        return

    step = height / n_stars
    y = 0

    for _ in range(n_stars + 1):
        ax.plot(
            x, y,
            marker='P',
            markersize=12,
            color=color,
            markeredgecolor='black',
            markeredgewidth=0.6
        )
        y += step

def Plots_Results():
    eval = np.load('Evaluate_plot.npy', allow_pickle=True)
    Terms = [' MAE', 'MPE', 'MSE', 'RMSE', 'MAPE', 'Accuracy']
    Graph_Terms = [0, 2, 3, 4 ]
    Algorithm = ['ROA-ARA-DRNN', 'FNO-ARA-DRNN', 'QSO-ARA-DRNN', 'APO-ARA-DRNN', 'PFAPO-ARA-DRNN']
    Classifier = ['KNN', 'SVR', 'DNN', 'RA-DRNN  ', 'PFAPO-ARA-DRNN']
#']
    Kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for m in range(eval.shape[2]):
                    Graph[k, m] = eval[i, k, m, Graph_Terms[j] ]

            nos = len(Kfold)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Step Per Epochs')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Kfold))
            width = 0.18
            colos = ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']
            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 0] - cap_height, 0), width=width, color='darkmagenta',
                               label=Algorithm[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 1] - cap_height, 0), width=width, color='darkorange',
                               label=Algorithm[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 2] - cap_height, 0), width=width, color='darkslateblue',
                               label=Algorithm[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 3] - cap_height, 0), width=width, color='#80B918',
                               label=Algorithm[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width, color='black',
                               label=Algorithm[4]))
            plt.xticks(X + 0.40, ['1', '2', '3', '4', '5'], fontname="Arial",
                       fontsize=16,
                       fontweight='bold', color='#14213d')
            plt.xlabel('KFold', fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for
                           color
                           in colos]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 16})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph)
                    triangle = [
                        (x, h),  # left top of bar
                        (x + w, h),  # right top
                        (x + w / 2, h + w * max / 3.9)  # taller triangle point
                    ]
                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)

            ax.set_ylim(0, max * 1.3)
            path=f"./Results/Dataset_KFold_{i+1}{Terms[Graph_Terms[j]]}_Alg.png"
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            colors = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Step Per Epochs')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Kfold))
            width = 0.18
            colos = ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']
            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:,5] - cap_height, 0), width=width, color=colors[0],
                               label=Classifier[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 6] - cap_height, 0), width=width, color=colors[1],
                               label=Classifier[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 7] - cap_height, 0), width=width, color=colors[2],
                               label=Classifier[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 8] - cap_height, 0), width=width, color=colors[3],
                               label=Classifier[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 9] - cap_height, 0), width=width, color=colors[4],
                               label=Classifier[4]))
            plt.xticks(X + 0.40, ['1', '2', '3', '4', '5'], fontname="Arial",
                       fontsize=16,
                       fontweight='bold', color='#14213d')
            plt.xlabel('KFold', fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for
                           color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 16})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph)
                    triangle = [
                        (x, h),  # left top of bar
                        (x + w, h),  # right top
                        (x + w / 2, h + w * max / 3.9)  # taller triangle point
                    ]
                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)

            ax.set_ylim(0, max * 1.3)
            path=f"./Results/Dataset_KFold_{i+1}{Terms[Graph_Terms[j]]}_Cls.png"
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def Table():
    eval = np.load('Evaluate_plot.npy', allow_pickle=True)
    Algorithm = ['TERMS/Activation function','ROA-ARA-DRNN', 'FNO-ARA-DRNN', 'QSO-ARA-DRNN', 'APO-ARA-DRNN', 'PFAPO-ARA-DRNN']
    Classifier = ['TERMS/Activation function', 'KNN', 'SVR', 'DNN', 'RA-DRNN  ', 'PFAPO-ARA-DRNN']
    Terms = [' MAE', 'MPE', 'MSE', 'RMSE', 'MAPE', 'Accuracy']

    Graph_Terms = np.array([0, 1, 2, 3, 4, 5]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4, 5]
    table_terms = [Terms[i] for i in Table_Terms]
    Epoch = ['Sigmoid', 'Relu', 'Softmax', 'Leaky ReLU', 'Tanh']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Epoch)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Epoch)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('--------------------------------------- Dataset - ', i + 1,  table_terms[k],
                  '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')

    Algorithm = ['TERMS','ROA-ARA-DRNN', 'FNO-ARA-DRNN', 'QSO-ARA-DRNN', 'APO-ARA-DRNN', 'PFAPO-ARA-DRNN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(3):
        Fitness = np.load('Fitness_' + str(i + 1) + '.npy', allow_pickle=True)
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[1])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        plt.plot(length, Fitness[0, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Fitness[1, :], color='#7b5de1', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Fitness[2, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Fitness[3, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Fitness[4, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.legend(loc=1, fontsize=16)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()
# RA-DRNN
# PFAPO-ARA-DRNN

def Plot_Results_Proposed():
    eval = np.load('Evaluate_plot.npy', allow_pickle=True)
    Terms = [' MAE', 'MPE', 'MSE', 'RMSE', 'MAPE', 'Accuracy']
    Graph_Terms = [0, 2, 3, 4, ]
    Algorithm = ['APO-ARA-DRNN', 'PFAPO-ARA-DRNN']
    Classifier = ['RA-DRNN', 'PFAPO-ARA-DRNN']

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j]]

            # -------------------------- Proposed Algorithm --------------------------
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset- ' + str(i+1) + Terms[Graph_Terms[j]] + 'Algorithm Comparison of Learning Percentage')
            ax.set_axisbelow(True)
            X = np.arange(Graph.shape[0])
            bar1 = ax.bar(X + 0.00, Graph[:, 3], color='#218380', edgecolor='w', width=0.35,
                          label=Algorithm[0])
            bar1_labels = ax.bar_label(container=bar1, size=9, label_type='edge',
                                       labels=[f' {x:.2f}% ' for x in Graph[:, 3]],
                                       fontweight='bold', color='k', padding=2)
            for label in bar1_labels:
                label.set_x(label.get_position()[0] - 0.10)

            bar2 = ax.bar(X + 0.37, Graph[:, 4], color='k', edgecolor='w', width=0.35, label=Algorithm[1])
            bar2_labels = ax.bar_label(container=bar2, size=9, label_type='edge',
                                       labels=[f' {x:.2f}% ' for x in Graph[:, 4]],
                                       fontweight='bold', color='k', padding=2)
            for label in bar2_labels:
                label.set_x(label.get_position()[0] + 1.02)
            # 8, 16, 32, 48, 64
            plt.xticks(X + 0.22, ['35', '45', '55', '65', '75'], fontname="Arial", fontsize=16, fontweight='bold',
                       color='#003566')
            plt.xlabel('Learning Percentage', fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#003566')
            for s in ['top', 'right']:
                plt.gca().spines[s].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#218380', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                       frameon=False, ncol=len(Algorithm), prop={'weight': 'bold', 'size': 16})
            plt.grid(alpha=0.2)
            path = "./Results/Dataset_%s_%s_Prop_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            fig.canvas.manager.set_window_title('Learning Percentage vs ' + Terms[Graph_Terms[j]])
            plt.show(block=False)
            plt.pause(1)
            plt.close()




def Plots_Results_solar():
    eval = np.load('Integrated_DG_System.npy', allow_pickle=True)
    Terms = ['Solar Energy Utilization Efficiency', 'Grid Stability', 'Scalability', 'Emissions', 'State of Charge ',' Depth of Discharge' , 'Charge–Discharge Efficiency', 'Voltage Regulation Index', 'Operating Cost', 'Cost of Battery Degradation', 'Net Present Value' ]
    Graph_Terms = [0, 2, 3, 4,5,6,7,8,9 ]
    Algorithm = ['ROA', 'FNO', 'QSO', 'APO', 'PFAPO']
    Classifier = ['IDCNN-LSTM', 'DNN', 'ResNet', 'RAEn  ', 'CIF-ALA-SCA-ARAEn']
    Kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for m in range(eval.shape[2]):
                    Graph[k, m] = eval[i, k, m, Graph_Terms[j]]

            nos = len(Kfold)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Step Per Epochs')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Kfold))
            width = 0.18
            colos = ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']
            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 0] - cap_height, 0), width=width, color='darkmagenta',
                               label=Algorithm[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 1] - cap_height, 0), width=width, color='darkorange',
                               label=Algorithm[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 2] - cap_height, 0), width=width, color='darkslateblue',
                               label=Algorithm[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 3] - cap_height, 0), width=width, color='#80B918',
                               label=Algorithm[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width, color='black',
                               label=Algorithm[4]))
            plt.xticks(X + 0.40, ['10', '20', '30', '40', '50'], fontname="Arial",
                       fontsize=16,
                       fontweight='bold', color='#14213d')
            plt.xlabel(' No of Batteries', fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for
                           color
                           in colos]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 16})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph)

                    triangle = np.array([
                        (x, h),
                        (x + w, h),
                        (x + w / 2, h + w * max / 3.9)
                    ])

                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)
                    ax.plot(x + w / 2, h + w * max / 3.9,
                            marker='*',
                            color='red',
                            markersize=14,
                            zorder=10)

            ax.set_ylim(0, max * 1.1)
            path=f"./Results/Dataset_KFold_{Terms[Graph_Terms[j]]}_Alg.png"
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Step Per Epochs')
            term_max = np.max(Graph)  # Calculate once per metric
            cap_height = term_max * 0.05
            X = np.arange(len(Kfold))
            width = 0.18
            colos = ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']
            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 0] - cap_height, 0), width=width, color='darkmagenta',
                               label=Algorithm[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 1] - cap_height, 0), width=width, color='darkorange',
                               label=Algorithm[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 2] - cap_height, 0), width=width, color='darkslateblue',
                               label=Algorithm[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 3] - cap_height, 0), width=width, color='#80B918',
                               label=Algorithm[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0), width=width, color='black',
                               label=Algorithm[4]))
            plt.xticks(X + 0.40, ['10', '20', '30', '40', '50'], fontname="Arial",
                       fontsize=16,
                       fontweight='bold', color='#14213d')
            plt.xlabel(' No of Batteries', fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=16, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#14213d')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for
                           color
                           in colos]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 16})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    max = np.max(Graph)

                    triangle = np.array([
                        (x, h),
                        (x + w, h),
                        (x + w / 2, h + w * max / 3.9)
                    ])

                    poly = Polygon(triangle, closed=True, color=color)
                    ax.add_patch(poly)
                    ax.plot(x + w / 2, h + w * max / 3.9,
                            marker='*',
                            color='red',
                            markersize=14,
                            zorder=10)

            ax.set_ylim(0, max * 1.1)
            path = f"./Results/Dataset_KFold_{Terms[Graph_Terms[j]]}_Alg.png"
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()



if __name__ == "__main__":
    # Plots_Results_solar()
    # Plot_Results_Proposed()
    # plotConvResults()
    # Table()
    Plots_Results()
