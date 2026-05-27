import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



## COCO ##
vlm_path1 = [
r"path/to/MoFit/Results/COCO/COCO_blip_500images_train_t_[140].txt",
][0]
vlm_path2 = [
r"path/to/MoFit/Results/COCO/COCO_blip_500images_test_t_[140].txt",
][0]



path_test1 = [
r"path/to/MoFit/Results/COCO/COCO_emb_500images_train_t_[140].txt"
][0]

path_test2 = [
r"path/to/MoFit/Results/COCO/COCO_emb_500images_test_t_[140].txt"
][0]




#################################################################################



MetricName = '---'
Dataname = '---'


def get_ori_data(path_train, path_test, vlm_path1, vlm_path2):
    global Dataname
    if '/' in path_train:
        Dataname = path_train.split('/')[-2]
    elif '\\' in path_train:
        Dataname = path_train.split('\\')[-2]

    print('dataname:', Dataname)


    with open(path_train, 'r', encoding='utf8') as f:
        train_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    with open(path_test, 'r', encoding='utf8') as f:
        test_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]
        

    train = np.array(train_list[:])
    test = np.array(test_list[:])
    
    
    
    with open(vlm_path1, 'r', encoding='utf8') as f:
        vlm_train_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]

    with open(vlm_path2, 'r', encoding='utf8') as f:
        vlm_test_list = [[float(e) for e in line.split('\t')] for line in f.readlines()[1:]]
        
    vlm_train = np.array(vlm_train_list[:])
    vlm_test = np.array(vlm_test_list[:])
    
    

    return train, test, vlm_train, vlm_test # max_v, min_v


def get_l_clidavg_last3(train, test, vlm_train, vlm_test):
    

    ## MoFit ##
    train_out = [[e[-2]] for e in train]
    test_out = [[e[-2]] for e in test]
    

    ## VLM default
    vlm_train_out = [[-e[-2]] for e in vlm_train]
    vlm_test_out = [[-e[-2]] for e in vlm_test]

    


    return train_out, test_out, vlm_train_out, vlm_test_out



def deal_data_weight_avg(train, test, vlm_train, vlm_test, alpha):
    global MetricName
    MetricName = 'weight_avg'

    ## Default ##
    train = [(1 - alpha) * vlm + alpha * e for e, vlm in zip(train, vlm_train)]
    test = [(1 - alpha) * vlm + alpha * e for e, vlm in zip(test, vlm_test)]



    return np.array(train), np.array(test)





def get_th(train, test, n_points=2000):
    train_list = train
    test_list = test
    max_e =  max(np.concatenate((train_list, test_list)))  #  max(train_list + test_list)
    min_e = min(np.concatenate((train_list, test_list)))

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

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    
    tpr_at_1fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.01), 0.0)
    tpr_at_01fpr = next((t for f, t in zip(FPR_list, TPR_list) if f >= 0.001), 0.0)




    return best_threshold, best_asr, auc, tpr_at_1fpr, tpr_at_01fpr, max_e, min_e


def get_cls_withTh(train, test, th):
    train_list = train
    test_list = test
    max_e = max(np.concatenate((train_list, test_list)))  
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


    train_data_2, test_data_2, vlm_train_data, vlm_test_data = get_ori_data(path_test1, path_test2, vlm_path1, vlm_path2)


    best_alpha = -1
    best_auc_shadow = 0
    test_asr_target = 0
    test_auc_target = 0
    best_train_data_target = None
    best_test_data_target = None

    train_data_target, test_data_target, vlm_train, vlm_test = get_l_clidavg_last3(train_data_2, test_data_2, vlm_train_data, vlm_test_data)

    train_data_target = np.array(train_data_target)     # shape: (N, 1)
    test_data_target = np.array(test_data_target)
    vlm_train = np.array(vlm_train)                # shape: (N, 1)
    vlm_test = np.array(vlm_test)
    
    
    
    emb_data = np.concatenate((train_data_target, test_data_target), axis=0)
    Scale1 = RobustScaler()
    Scale1.fit(emb_data)
    
    vlm_data = np.concatenate((vlm_train, vlm_test), axis=0)
    Scale2 = RobustScaler()
    Scale2.fit(vlm_data)
    

   
    train_data_target, test_data_target, vlm_train_stack, vlm_test_stack = Scale1.transform(train_data_target), Scale1.transform(test_data_target),  Scale2.transform(vlm_train),  Scale2.transform(vlm_test)
    

    vlm_train = vlm_train_stack
    vlm_test = vlm_test_stack
    
    results = []

    # alphas = np.linspace(0.1, 10.1, 201)
    alphas = np.linspace(0, 1.0, 21)
    

    best_train_data_target_alpha_npy = None
    best_test_data_target_alpha_npy = None
    best_alpha_value = None
    
    for alpha in alphas:
        
        train_data_target_alpha, test_data_target_alpha = deal_data_weight_avg(train_data_target, test_data_target, vlm_train, vlm_test, alpha)
        

        best_threshold, best_asr, auc, tpr1, tpr01, max_e, min_e = get_th(train_data_target_alpha, test_data_target_alpha)

        results.append((alpha, best_asr*100, best_threshold, auc*100, tpr1*100, tpr01*100))
        
        ## Save .npy ##
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
    
