
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
    

        
P = parse_args()


print(P)
labels_list = [1, 3, 2]
patients_list = ['Pneumonia', 'TB', 'COVID-19']

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
  
    print('='*30)
    print(patient)
    print('sensitivity : %.3f'%TPR)
    print('specificity : %.3f'%TNR)
    print('precision : %.3f'%PPV)
    print('AUC : %.3f'%AUC)

print('='*30)