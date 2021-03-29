from collections import defaultdict

def accuracy_score(pred: list, target: list) -> float:
    """
    Metrics: Accuracy score
    args:
        pred: List predictions of model acronym disambiguation
        target: Expansion of acronym word to be predicted
    """
    assert len(pred) == len(target), "Lenght of prediction and target should be equal"
    correct = 0
    for i in range(len(target)):
        if pred[i] == target[i]: correct += 1
    return correct/len(pred)

def precision_recall_f1(pred:list, target:list):
    """
    Metrics: Precision, Recall and F1:
    args:
        pred: List predictions of model acronym disambiguation
        target: Expansion of acronym word to be predicted
    """

    assert len(pred) == len(target), "Lenght of prediction and target should be equal"
    
    expansion = set()

    correct_per_expansion = defaultdict()
    total_per_expansion = defaultdict()
    pred_per_expansion = defaultdict()

    for i in range(len(target)):
        expansion.add(target[i])
        total_per_expansion[target[i]] += 1
        pred_per_expansion[pred[i]] += 1
        if pred[i] == target[i]:
            correct_per_expansion[target[i]] += 1
    
    precision = defaultdict() # TP/(TP + FP)
    recall = defaultdict()  # TP/(TP + FN)

    for exp in expansion:
        precision[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        recall[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(total_per_expansion.values())
    micro_f1 = 2*micro_prec*micro_recall/(micro_prec+micro_recall) if micro_prec+micro_recall != 0 else 0

    macro_prec = sum(precision.values()) / len(precision)
    macro_recall = sum(recall.values()) / len(recall)
    macro_f1 = 2*macro_prec*macro_recall / (macro_prec+macro_recall) if macro_prec+macro_recall != 0 else 0

    return micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1

def metrics(pred, target):
    correct = 0
    total_pred = 0
    for i in range(len(pred)):
        if target[i][0] in pred[i]: correct += 1
        total_pred += len(pred[i])
    recall = correct/len(target)
    precision = correct/total_pred
    f1_score = 2*recall*precision/(recall+precision)
    return recall, precision, f1_score

def accuracy(pred, target):
    correct = 0
    assert len(pred) == len(target), "Number of predictions have to be equal target"
    for i in range(len(pred)):
        if pred[i][0] == target[i][0]: correct += 1
    return correct/len(pred)