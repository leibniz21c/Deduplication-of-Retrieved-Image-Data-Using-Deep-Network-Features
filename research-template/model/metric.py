def _get_confusion_mat(output_clusters, target_clusters):
    if target_clusters.shape != output_clusters.shape:
        raise IndexError("target_clusters and output_clusters must be same dimension.")
    comb = target_clusters.shape[0]*(target_clusters.shape[0] - 1)/2
    summed = target_clusters + output_clusters
    subtracted = target_clusters - output_clusters
    
    true_positive = (summed == 2).nnz
    false_positive = (subtracted == -1).nnz
    false_negative = (subtracted == 1).nnz
    true_negative = comb - true_positive - false_negative - false_positive
    
    return {'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative}

def precision(output_clusters, target_clusters):
    """Metric 1. Precision
    
    """
    confusion_mat = _get_confusion_mat(output_clusters, target_clusters)
    precision = confusion_mat['true_positive']/(confusion_mat['true_positive'] + confusion_mat['false_positive'])
    return precision

def recall(output_clusters, target_clusters):
    """Metric 2. Recall
    
    """
    confusion_mat = _get_confusion_mat(output_clusters, target_clusters)
    recall = confusion_mat['true_positive']/(confusion_mat['true_positive'] + confusion_mat['false_negative'])
    return recall

def f1_score(output_clusters, target_clusters):
    """Metric 3. F1-score
    
    """
    confusion_mat = _get_confusion_mat(output_clusters, target_clusters)
    precision = confusion_mat['true_positive']/(confusion_mat['true_positive'] + confusion_mat['false_positive'])
    recall = confusion_mat['true_positive']/(confusion_mat['true_positive'] + confusion_mat['false_negative'])
    f1_score = 2*(precision*recall)/(precision + recall)
    return f1_score