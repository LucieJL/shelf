import numpy as np

def precision_score(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError('Lenght of targets and predictions should be the same')

    TP = 0
    FP = 0
    for tar, pred in zip(targets, predictions):
        if pred == 1:
            if tar == 1:
                TP += 1
            else:
                FP += 1

    if TP + FP == 0:
        return 0.0

    return TP / (TP + FP)

def recall_score(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError('Lenght of targets and predictions should be the same')

    TP = 0
    FN = 0

    for tar, pred in zip(targets, predictions):
        if tar == 1:
            if pred == 1:
                TP += 1
            else:
                FN += 1

    if TP + FN == 0:
        return 0.0

    return TP / (TP + FN)

def f_beta_score(targets, predictions, beta=1):
    if len(targets) != len(predictions):
        raise ValueError('Lenght of targets and predictions should be the same')

    prec = precision_score(targets, predictions)
    rec = recall_score(targets, predictions)

    if prec + rec == 0:
        return 0.0

    return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)

def tcr_score(targets, predictions, best_threshes): # targets and predition = [ABZ, DCALL, BP]
    tp_abz = 0
    tp_dcall = 0
    tp_bp = 0

    total_abz = 0
    total_dcall = 0
    total_bp = 0

    for tar, pred in zip(targets, predictions):
        if pred[0] > best_threshes['abz']:
            total_abz += 1
            if tar[0] == 1:
                tp_abz += 1

        if pred[1] > best_threshes['dcall']:
            total_dcall += 1
            if tar[1] == 1:
                tp_dcall += 1
        if pred[2] > best_threshes['bp']:
            total_bp += 1
            if tar[2] == 1:
                tp_bp += 1
    tcr_abz = tp_abz / total_abz if total_abz != 0 else 0
    tcr_dcall = tp_dcall / total_dcall if total_dcall !=0 else 0
    tcr_bp = tp_bp / total_bp if total_bp !=0 else 0

    return np.mean([tcr_abz, tcr_dcall, tcr_bp])

def nmr_score(targets, predictions, best_threshes):
    noisy_abz = 0
    noisy_dcall = 0
    noisy_bp = 0

    total_noise = 0

    abz_thresh = best_threshes['abz']
    dcall_thresh = best_threshes['dcall']
    bp_thresh = best_threshes['bp']

    for tar, pred in zip(targets, predictions):
        if pred[0] < abz_thresh and pred[1] < dcall_thresh and pred[2] < bp_thresh: # total noise predictions
            total_noise += 1

        if sum(tar) == 0: # true noise case
            if pred[0] > abz_thresh:
                noisy_abz +=1
            if pred[1] > dcall_thresh:
                noisy_dcall += 1
            if pred[2] > bp_thresh:
                noisy_bp += 1

    if total_noise == 0:
        return 0

    return (noisy_abz + noisy_dcall + noisy_bp) / total_noise

def cmr_score(targets, predictions, best_threshes):
    total_abz = 0
    total_dcall = 0
    total_bp = 0

    dcall_for_abz = 0
    bp_for_abz = 0

    abz_for_dcall = 0
    bp_for_dcall = 0

    abz_for_bp = 0
    dcall_for_bp = 0

    abz_thresh = best_threshes['abz']
    dcall_thresh = best_threshes['dcall']
    bp_thresh = best_threshes['bp']


    for tar, pred in zip(targets, predictions):
        if pred[0] > abz_thresh:
            total_abz += 1
        if pred[1] > dcall_thresh:
            total_dcall += 1
        if pred[2] > bp_thresh:
            total_bp += 1

        if tar[0] == 1 and pred[0] < abz_thresh: # missed ABZ
            if pred[1] > dcall_thresh : # confused with D-call
                dcall_for_abz += 1
            if pred[2] > bp_thresh: # confused with Bp tonal
                bp_for_abz += 1

        if tar[1] == 1 and pred[1] < dcall_thresh:
            if pred[0] > abz_thresh:
                abz_for_dcall += 1
            if pred[2] > bp_thresh:
                bp_for_dcall += 1

        if tar[2] == 1 and pred[2] < bp_thresh:
            if pred[0] > abz_thresh:
                abz_for_bp += 1
            if pred[1] > dcall_thresh:
                dcall_for_bp += 1

    abz1 = dcall_for_abz / total_abz if total_abz != 0 else 0
    abz2 = bp_for_abz / total_abz if total_abz != 0 else 0
    dcall1 = abz_for_dcall / total_dcall if total_dcall != 0 else 0
    dcall2 = bp_for_dcall / total_dcall if total_dcall != 0 else 0
    bp1 = abz_for_bp / total_bp if total_bp != 0 else 0
    bp2 = dcall_for_bp / total_bp if total_bp != 0 else 0

    return np.mean([abz1, abz2, dcall1, dcall2, bp1, bp2])


def fitness_metric(tcr, nmr, cmr):
    return np.mean([tcr, (1-nmr), (1-nmr), (1-cmr)])

