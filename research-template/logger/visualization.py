import os
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def roc_curve(df_metrics, save_dir, config):
    if os.path.isdir(save_dir) is None:
        raise ValueError('Invalid input(save_dir)')

    # roc curve
    sorted_index = np.argsort(df_metrics['false_positive_rate'])
    fpr_list_sorted = np.array(df_metrics['false_positive_rate'])[sorted_index]
    tpr_list_sorted = np.array(df_metrics['recall'])[sorted_index]
    roc_auc = integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
    plt.figure()
    lw = 2

    if config['distance_thresholds']['logscale']:
        plt.xscale('log')
        
    plt.plot(
        df_metrics['false_positive_rate'],
        df_metrics['recall'],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(save_dir, dpi=300)


def precision_recall_curve(df_metrics, save_dir, config):
    if os.path.isdir(save_dir) is None:
        raise ValueError('Invalid input(save_dir)')

    if config['distance_thresholds']['logscale']:
        plt.xscale('log')
    
    # precision-recall curve
    lw = 2
    plt.plot(
        df_metrics['recall'],
        df_metrics['precision'],
        color="red",
        lw=lw,
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.savefig(save_dir, dpi=300)
    