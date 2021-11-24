
from datasets.datasets import *
from sklearn.metrics import roc_auc_score
import matplotlib
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

matplotlib.use('Agg')

from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        type=str, default="../weight_folder/last.model")
    parser.add_argument('--save_name', help='save visualized results',
                        type=str, default="visual")
    parser.add_argument("-f", type=str, default=1)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()

    
def get_total_scores(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return labels, scores

def percormance_multi_class(TARGET, PROB):

    cnf_matrix = confusion_matrix(TARGET, PROB)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    TPR = "Abnormal case is positive \nSensitivity, hit rate, recall, or true positive rate: " + np.array2string(TPR[1])
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    TNR = "Specificity or true negative rate: " + np.array2string(TNR[1])
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    PPV = "Precision or positive predictive value: " + np.array2string(PPV[1])
    # Negative predictive value
    NPV = TN/(TN+FN)
    NPV = "Negative predictive value: " + np.array2string(NPV[1])
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FPR = "Fall out or false positive rate: " + np.array2string(FPR[1])
    # False negative rate
    FNR = FN/(TP+FN)
    FNR = "False negative rate: " + np.array2string(FNR[1])
    # False discovery rate
    FDR = FP/(TP+FP)
    FDR = "False discovery rate: " + np.array2string(FDR[1])
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC = "Accuracy: " + np.array2string(ACC[1])

    REPORTS = TPR  + "\n" + TNR + "\n" + PPV + "\n" + NPV + "\n" + FPR + "\n" + FNR +"\n" + FDR +"\n" + ACC
    
    return REPORTS

def positive_performance(TARGET, PROB):
    cnf_matrix = confusion_matrix(TARGET, PROB)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    TPR = TPR[1]
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    PPV = PPV[1]
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC = ACC[1]
    TNR = TN/(TN+FP)
    TNR = TNR[1]
    return TPR, PPV, ACC, TNR
    

def get_metric():
    preds = np.copy(scores)
    index_normal = np.where(preds>best_thresh)[0]
    index_abnormal = np.where(preds<=best_thresh)[0]
    preds[index_normal]=0
    preds[index_abnormal]=1
    targets = np.copy(labels)
    index_normal = np.where(targets==1)[0]
    index_abnormal = np.where(targets==0)[0]
    targets[index_normal]=0
    targets[index_abnormal]=1
    cr = classification_report(targets, preds, digits=4)
    cm = np.array2string(confusion_matrix(targets, preds))
    reports = percormance_multi_class(targets, preds)
    with open(os.path.join(job_dir, f'report_{label}_q{int(percentile*100)}.txt'), "w") as f: 
        f.write('Title\n\nClassification Report\n\n{}\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, reports, cm))
        
P = parse_args()


print(P)
labels_list = [1, 3, 2]
patients_list = ['Pneumonia', 'TB', 'COVID-19']

tpr_dict={"Pneumonia":[],"TB":[],"COVID-19":[]}
ppv_dict={"Pneumonia":[],"TB":[],"COVID-19":[]}
acc_dict={"Pneumonia":[],"TB":[],"COVID-19":[]}
AUC_dict={"Pneumonia":[],"TB":[],"COVID-19":[]}
TNR_dict={"Pneumonia":[],"TB":[],"COVID-19":[]}
percentile=0.2
job_dir = '../visual'
for label, patient in zip(labels_list, patients_list):

    normal_dir = os.path.join(job_dir, "0", "npy", "*").replace("\\","/")
    abnormal_dir = os.path.join(job_dir, str(label), "npy", "*").replace("\\","/")
    normal_npy = glob.glob(normal_dir)
    abnormal_npy = glob.glob(abnormal_dir)

    scores_id = []
    for npy in normal_npy:
        cam = np.load(npy)
        nonzero_cam = cam[np.nonzero(cam)].ravel()
        percentile_cam = np.quantile(nonzero_cam, percentile)
        scores_id.append(percentile_cam)

    scores_ood = []
    for npy in abnormal_npy:
        cam = np.load(npy)
        nonzero_cam = cam[np.nonzero(cam)].ravel()
        percentile_cam = np.quantile(nonzero_cam, percentile)
        scores_ood.append(percentile_cam)

    labels, scores = get_total_scores(scores_id, scores_ood)
    AUC = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    best_thresh = thresholds[ix]

    preds = np.copy(scores)
    index_normal = np.where(preds>best_thresh)[0]
    index_abnormal = np.where(preds<=best_thresh)[0]
    preds[index_normal]=0
    preds[index_abnormal]=1
    targets = np.copy(labels)
    index_normal = np.where(targets==1)[0]
    index_abnormal = np.where(targets==0)[0]
    targets[index_normal]=0
    targets[index_abnormal]=1
    TPR, PPV, ACC, TNR = positive_performance(targets, preds)
    tpr_dict[patient].append(TPR)
    ppv_dict[patient].append(PPV)
    acc_dict[patient].append(ACC)
    AUC_dict[patient].append(AUC)
    TNR_dict[patient].append(TNR)
    print('='*30)
    print(patient+'\nsensitivity : ',TPR,'\n','specificity : ',TNR,'\n','precision : ', PPV,'\n','AUC : ', AUC,'\n', )

print('='*30)