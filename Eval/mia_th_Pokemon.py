import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve




path_test1 = [
r"path/to/../Pokemon_emb_..._train_t_[140].txt"
][0]

path_test2 = [
r"path/to/../Pokemon_emb_..._test_t_[140].txt"
][0]


alphas = np.linspace(0, 1.0, 21)





MetricName = '---'
Dataname = '---'


def get_ori_data(path_train, path_test):
    global Dataname
    if '/' in path_train:
        Dataname = path_train.split('/')[-2]
    elif '\\' in path_train:
        Dataname = path_train.split('\\')[-2]

    print('dataname:', Dataname)
    # exit()

    with open(path_train, 'r', encoding='utf8') as f:
        train_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    with open(path_test, 'r', encoding='utf8') as f:
        test_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    train = np.array(train_list[:])
    test = np.array(test_list[:])


    
    return train, test,  # max_v, min_v


def get_l_clidavg_last3(train, test):


    ## MoFit ##
    train_out = [[e[-1], e[-2]] for e in train]
    test_out = [[e[-1], e[-2]] for e in test]


    return train_out, test_out



def deal_data_weight_avg(train, test, alpha):
    global MetricName
    MetricName = 'weight_avg'
    assert len(train[0]) == 2

    ## Default ##
    train = [(1 - alpha) * e[0] + alpha * e[1] for e in train]
    test = [(1 - alpha) * e[0] + alpha * e[1] for e in test]

 
    return np.array(train), np.array(test)





def get_th(train, test, n_points=2000):
    # print('get th...')
    train_list = train
    test_list = test
    max_e =  max(np.concatenate((train_list, test_list)))  #  max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))

    # n_points = 2000
    best_asr = 0
    best_threshold = 0


    FPR_list = []
    TPR_list = []
    from sklearn import metrics

    for threshold in list(np.arange(min_e, max_e, (max_e - min_e) / n_points)):
        
        # print(threshold, type(threshold))
        TP = (train_list <= threshold).sum()
        TN = (test_list > threshold).sum()
        FP = (test_list <= threshold).sum()
        FN = (train_list > threshold).sum()
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        ASR = (TP + TN) / (TP + TN + FP + FN)
        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > best_asr:
            best_asr = ASR
            best_threshold = threshold
            
        # cnt += 1
    # print(f"count: {cnt}")

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    
    thresholds = np.arange(min_e, max_e, (max_e - min_e) / n_points)


    tpr_at_1fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.01), 0.0)
    tpr_at_01fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.001), 0.0)



    return best_threshold, best_asr, auc, tpr_at_1fpr, tpr_at_01fpr, max_e, min_e, None



def get_th_max_tpr_at_1fpr(train, test, n_points=2000):
    train_list = train
    test_list = test
    max_e = max(np.concatenate((train_list, test_list)))
    min_e = min(np.concatenate((train_list, test_list)))

    best_tpr = 0
    best_threshold = None

    FPR_list = []
    TPR_list = []

    for threshold in np.linspace(min_e, max_e, n_points):
        TP = (train_list <= threshold).sum()
        TN = (test_list > threshold).sum()
        FP = (test_list <= threshold).sum()
        FN = (train_list > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        FPR_list.append(FPR.item())
        TPR_list.append(TPR.item())

        if FPR <= 0.01 and TPR > best_tpr:
            best_tpr = TPR
            best_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)

    from sklearn import metrics
    auc = metrics.auc(FPR_list, TPR_list)

    tpr_at_1fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.01), 0.0)
    tpr_at_01fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.001), 0.0)

    return best_threshold, best_tpr, auc, tpr_at_1fpr, tpr_at_01fpr, max_e, min_e

def get_cls_withTh(train, test, th):
    train_list = train
    test_list = test
    max_e = max(np.concatenate((train_list, test_list)))  # max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))
    # n_points = 2000

    print("\ntrain_list[:3], test_list[:3]", train_list[:3], test_list[:3])
    print("\nmax_e, min_e:", max_e, min_e)

    TP = (train_list <= th).sum()
    TN = (test_list > th).sum()
    FP = (test_list <= th).sum()
    FN = (train_list > th).sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ASR = (TP + TN) / (TP + TN + FP + FN)

    print('\n', 'TEST: ', 'ASR:', ASR, 'by the given threshold:', th)

    return ASR  # best_threshold, best_asr, auc, FPR_list, TPR_list, max_e, min_e



def get_1_fpr(train_data_target, test_data_target):
    print('**** get get_1_fpr: ******')
    labels = [0]*len(train_data_target)+[1]*len(test_data_target)
    datas = np.concatenate((train_data_target, test_data_target), axis=0)


    best_threshold = None
    best_accuracy = 0.0

    min_threshold = min(datas)
    max_threshold = max(datas)
    threshold_step = (max_threshold - min_threshold) / 2000

    for threshold in list(np.arange(min_threshold, max_threshold, threshold_step)):
        predicted_values = [1 if value > threshold else 0 for value in datas]

        accuracy = accuracy_score(labels, predicted_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print( '|   best_accuracy, best_threshold, th% :', best_accuracy, best_threshold,
          (best_threshold - min_threshold) / (max_threshold - min_threshold))

    auc = roc_auc_score(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    print( "|    AUC Score:", auc)

    fpr, tpr, _ = roc_curve(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in datas])
    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    print( "|   tpr_at_1_percent_fpr:", tpr_at_1_percent_fpr)




if __name__ == '__main__':
    from sklearn.preprocessing import RobustScaler

    print('begin .. ')
    print('------- get data ------------')

    train_data_2, test_data_2 = get_ori_data(path_test1, path_test2)


    best_alpha = -1
    best_auc_shadow = 0
    test_asr_target = 0
    test_auc_target = 0
    best_train_data_target = None
    best_test_data_target = None

    train_data_target, test_data_target = get_l_clidavg_last3(train_data_2, test_data_2)
   
    Scale = RobustScaler()
    Scale.fit(np.concatenate((train_data_target, test_data_target)))

    train_data_target, test_data_target = Scale.transform(train_data_target), Scale.transform(test_data_target)

    results = []
    
    
    best_train_data_target_alpha_npy = None
    best_test_data_target_alpha_npy = None
    best_alpha_value = None
    
    for alpha in alphas:
       
        train_data_target_alpha, test_data_target_alpha = deal_data_weight_avg(train_data_target, test_data_target, alpha)

        

        best_threshold, best_asr, auc, tpr1, tpr01, max_e, min_e, thres_tpr1 = get_th(train_data_target_alpha, test_data_target_alpha)

      
        results.append((
            alpha,
            best_asr * 100,
            best_threshold,
            auc * 100,
            round(tpr1 * 100, 4),
            round(tpr01 * 100, 4)
        ))
        

        train_data_target_alpha_npy = np.array(train_data_target_alpha)
        test_data_target_alpha_npy = np.array(test_data_target_alpha)

        if best_asr * 100 >= max([r[1] for r in results], default=0):
            best_train_data_target_alpha_npy = train_data_target_alpha_npy
            best_test_data_target_alpha_npy = test_data_target_alpha_npy
            best_alpha_value = alpha
       
    import pandas as pd
    df = pd.DataFrame(results, columns=["Alpha", "Best ASR", "Best Threshold", "AUC", "TPR@1%FPR", "TPR@0.1%FPR"])
    print(df.to_string(index=False))

    max_asr = df["Best ASR"].max()
    df_max_asr = df[df["Best ASR"] == max_asr]

    best_row = df_max_asr.loc[df_max_asr["AUC"].idxmax()]

    print("\n📌 Best Overall Result (Highest ASR, then Highest AUC):")
    print(best_row.to_frame().T.to_string(index=False))
    
