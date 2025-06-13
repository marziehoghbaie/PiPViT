import pandas as pd
from bstrap import boostrapping_CI
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
def bootstrap_CI(path):
    """
    bootstrap CI for the metrics with CI of 95%
    http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
    """
    df = pd.read_csv(path) # y_probs,y_pred,y_true
    df['y_probs'] = df['y_probs'].apply(lambda x: list(map(float, x.strip('[]').split())))


    f1 = boostrapping_CI(f1_score, df, nbr_runs=1000, **{'average':'weighted', 'metric_type':'f1_score'})
    precision = boostrapping_CI(precision_score, df, nbr_runs=1000, **{'average':'weighted', 'metric_type':'precision_score'})
    recall = boostrapping_CI(recall_score, df, nbr_runs=1000, **{'average':'weighted', 'metric_type':'recall_score'})
    bacc = boostrapping_CI(balanced_accuracy_score, df, nbr_runs=1000, **{'metric_type':'balanced_accuracy_score'})

    if len(df['y_true'].unique()) == 2:
        df['y_probs'] = df['y_probs'].apply(lambda x: x[1])
        df['y_onehot_test'] = df['y_true']
    else:
        label_binarizer = LabelBinarizer().fit(list(df['y_true'].values))
        y_onehot_test = label_binarizer.transform(list(df['y_true'].values))
        df['y_onehot_test'] = list(y_onehot_test)

    roc_auc = boostrapping_CI(roc_auc_score, df, nbr_runs=1000, **{'metric_type':'roc_auc_score'})
    metrics = {'bacc': bacc, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall, 'f1': f1
               }
    for m in metrics.keys():
        avg_metric = metrics[m]['avg_metric']
        metric_ci_lb = metrics[m]['metric_ci_lb']
        metric_ci_ub = metrics[m]['metric_ci_ub']
        print(f'{m}: {avg_metric:.3f} ({metric_ci_lb:.3f}-{metric_ci_ub:.3f})')

if __name__ == '__main__':
    csv_path = ('/PiPViTV2/Results/test_384_Ablation_scratch_NOImagenet/avg/'
                'vit_base_patch32_384_normed_relu_384_OCT5K/20250127-122418/results.csv')
    bootstrap_CI(csv_path)