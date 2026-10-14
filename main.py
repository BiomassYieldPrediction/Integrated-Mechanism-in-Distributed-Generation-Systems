import pandas as pd
from numpy import matlib
from APO import APO
from FNO import FNO
from Global_Vars import Global_Vars
from Model_ARDNN import Model_ARA_DRNN, Model_ARA_DRNN_energy
from Model_Dnn import Model_DNN, Model_DNN_energy
from Model_KNN import Model_KNN, Model_KNN_energy
from Model_svr import Model_SVR, Model_SVR_energy
from OOA import OOA
from ROA import ROA
from obj_function import objfun, obj_battery
from Plot_results import *
from Proposed import Proposed
from QSO import QSO

No_of_dataset=3

# Read dataset 1
an = 0
if an == 1:
    df = pd.read_csv('./Dataset/1 data/Plant_1_Generation_Data.csv',encoding="latin-1")
    Data = df.iloc[:, :-1].values
    Target = df.iloc[:, -1].values
    np.save('Data_1.npy', Data)
    np.save('Target_1.npy',Target)

# Read dataset 2
an = 0
if an == 1:
    df = pd.read_csv('./Dataset/archive/Location1.csv',encoding="latin-1")
    Data = df.iloc[:, :-1].values
    Target = df.iloc[:, -1].values
    np.save('Data_2.npy', Data)
    np.save('Target_2.npy',Target)

# Read dataset 3
an = 0
if an == 1:
    Datasets = './Dataset/3/PowerGeneration.csv'
    df = pd.read_csv(Datasets)
    Datas = np.asarray(df)
    Target = df['Actual (MW)']
    Tar =  Target.to_numpy()
    np.save('Data_3.npy',Datas)
    np.save('Target_3.npy', Tar)



# optimization for Power Generation Forecasting
an = 0
if an == 1:
    for n in range(No_of_dataset):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5, 0.01, 5], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 50], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("ROA...")
        [bestfit1, fitness1, bestsol1, time] = ROA(initsol, fname, xmin, xmax, Max_iter)  # SOA

        print("FNO...")
        [bestfit2, fitness2, bestsol2, time1] =  FNO(initsol, fname, xmin, xmax, Max_iter)  # NBOA

        print("QSO...")
        [bestfit3, fitness3, bestsol3, time2] =  QSO(initsol, fname, xmin, xmax, Max_iter)  # RBMO

        print("APO...")
        [bestfit4, fitness4, bestsol4, time3] = APO(initsol, fname, xmin, xmax, Max_iter)  # AOBOA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
        BestFit=[bestfit1,bestfit2,bestfit3,bestfit4,bestfit5]
        Time = [time, time1, time2, time3, time4]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)
        np.save('Fitness_' + str(n + 1) + '.npy', Fitness)
        np.save('BestFit_' + str(n + 1) + '.npy', Fitness)
        np.save('Time_' + str(n + 1) + '.npy', Time)

# Power Generation Forecasting
an = 0
if an == 1:
    Evaluate = []
    for n in range(No_of_dataset):
        Feat = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Learning_per = [35, 45, 55, 65, 75]
        for act in range(len(Learning_per)):
            learnperc = round(Feat.shape[0] *Learning_per[act])  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 7))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = BestSol[j, :]
                Eval[j, :], pred = Model_ARA_DRNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)

            Eval[5, :], pred1 = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                         )
            Eval[6, :], pred2 = Model_KNN(Train_Data, Train_Target, Test_Data,
                                            Test_Target)
            Eval[7, :], pred3 = Model_SVR(Train_Data, Train_Target, Test_Data, Test_Target,
                                        )
            Eval[8, :], pred4 = Model_ARA_DRNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                           )
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Evaluate_prop.npy', Evaluate)


# optimization for battery storage
an =0
if an == 1:
    for n in range(No_of_dataset):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([100, 0, 32], Npop, 1)
        xmax = matlib.repmat([1000, 1, 256], Npop, 1)
        fname = obj_battery
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("ROA...")
        [bestfit1, fitness1, bestsol1, time] = ROA(initsol, fname, xmin, xmax, Max_iter)  # SOA

        print("FNO...")
        [bestfit2, fitness2, bestsol2, time1] =  FNO(initsol, fname, xmin, xmax, Max_iter)  # NBOA

        print(" QSO...")
        [bestfit3, fitness3, bestsol3, time2] =  QSO(initsol, fname, xmin, xmax, Max_iter)  # RBMO

        print("APO...")
        [bestfit4, fitness4, bestsol4, time3] = APO(initsol, fname, xmin, xmax, Max_iter)  # AOBOA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
        BestFit=[bestfit1,bestfit2,bestfit3,bestfit4,bestfit5]
        Time = [time, time1, time2, time3, time4]
        np.save('BestSol_batt_' + str(n + 1) + '.npy', BestSol)
        np.save('Fitness_batt_' + str(n + 1) + '.npy', Fitness)
        np.save('BestFit_batt_' + str(n + 1) + '.npy', Fitness)
        np.save('Time_batt_' + str(n + 1) + '.npy', Time)



# Battery storage management
an = 0
if an == 1:
    Evaluate = []
    for n in range(No_of_dataset):
        Feat = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        learnperc = round(Feat.shape[0] *0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 12))
        for j in range(BestSol.shape[0]):
            sol = BestSol[j, :]
            Eval[j, :], pred =Model_ARA_DRNN_energy(Train_Data, Train_Target, Test_Data, Test_Target, sol)

        Eval[5, :], pred1 = Model_DNN_energy(Train_Data, Train_Target, Test_Data, Test_Target,
                                     )
        Eval[6, :], pred2 = Model_KNN_energy(Train_Data, Train_Target, Test_Data,
                                        Test_Target)
        Eval[7, :], pred3 = Model_SVR_energy(Train_Data, Train_Target, Test_Data, Test_Target,
                                    )
        Eval[8, :], pred4 = Model_ARA_DRNN_energy(Train_Data, Train_Target, Test_Data, Test_Target,
                                       )
        Eval[9, :] = Eval[4, :]
        Evaluate.append(Eval)
    np.save('Evaluate_all_LearningRate.npy', Evaluate)



Plots_Results_solar()
Table()
Plot_Results_Proposed()
plotConvResults()