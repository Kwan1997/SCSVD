import numpy as np
from collections import defaultdict
from sklearn.metrics import auc
from tqdm import trange


def getMAE(predictions):
    return np.mean([float(abs(true_r - est)) for (_, _, true_r, est, _) in predictions])


def getRMSE(predictions):
    return np.sqrt(np.mean([float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]))


def getColdMAE(predictions, ur):
    # First map the predictions to each user.
    MAE = 0
    cnt = 0
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        # cold users
        if len(ur[uid]) <= 5:
            MAE += np.sum(np.array([np.abs(t - e) for e, t in user_ratings]))
            cnt += len(user_ratings)
    return MAE / cnt


def getColdRMSE(predictions, ur):
    # First map the predictions to each user.
    RMSE = 0
    cnt = 0
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        # cold users
        if len(ur[uid]) <= 5:
            RMSE += np.sum(np.array([(t - e) ** 2 for e, t in user_ratings]))
            cnt += len(user_ratings)
    return np.sqrt(RMSE / cnt)


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = len(user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum((true_r >= threshold)
                              for (_, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    p = sum(prec for prec in precisions.values()) / len(precisions)
    r = sum(rec for rec in recalls.values()) / len(recalls)
    f = 2 * p * r / (p + r)
    return p, r, f


def getROC(predictions, threshold=3.5):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    user_item = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        user_item[uid].append(iid)
    maxk = max(map(len, user_item.values()))
    fpr = np.array([0])
    tpr = np.array([0])
    for k in trange(1, maxk + 1, 1):
        failures = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            # Number of irrelevant items
            n_irrel = sum((true_r < threshold) for (_, true_r) in user_ratings)
            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum((true_r >= threshold)
                                  for (_, true_r) in user_ratings[:k])
            # Number of irrelevant and recommended items in top k
            n_irrel_and_rec_k = sum((true_r < threshold)
                                    for (_, true_r) in user_ratings[:k])
            # Failure@K: Proportion of recommended items that are irrelevant
            failures[uid] = n_irrel_and_rec_k / n_irrel if n_irrel != 0 else 0
            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        f = sum(prec for prec in failures.values()) / len(failures)
        r = sum(rec for rec in recalls.values()) / len(recalls)
        fpr = np.append(fpr, f)
        tpr = np.append(tpr, r)
    fpr = np.append(fpr, 1)
    tpr = np.append(tpr, 1)
    return fpr, tpr


def getAUC(fpr, tpr):
    return auc(fpr, tpr)


def getNDCG(predictions, k=10, threshold=3.5):
    idcg = 0
    dcg = 0
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    user_posLength = defaultdict()
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        user_posLength[uid] = sum(
            true_r >= threshold for est, true_r in user_ratings)
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        rec = user_ratings[:k]
        for ind in range(len(rec)):
            index = ind + 1
            idcg += 1 / (np.log2(index+1)
                         ) if index <= user_posLength[uid] else 0
            dcg += 1 / (np.log2(index+1)) if rec[ind][1] >= threshold else 0
    print(dcg, idcg)
    return dcg/idcg if idcg != 0 else 1


def getNDCG2(predictions, k=10, threshold=3.5):
    ndcg = dict()
    user_est_true = defaultdict(list)
    user_posLength = defaultdict()
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        user_posLength[uid] = sum(
            true_r >= threshold for est, true_r in user_ratings)
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        rec = user_ratings[:k]
        idcg = 0
        dcg = 0
        for ind in range(len(rec)):
            index = ind + 1
            idcg += 1 / (np.log2(index + 1)
                         ) if index <= user_posLength[uid] else 0
            dcg += 1 / (np.log2(index + 1)) if rec[ind][1] >= threshold else 0
        ndcg[uid] = dcg / idcg if idcg != 0 else 1
    return sum(val for val in ndcg.values()) / len(ndcg)


def getMRR(predictions, k=10, threshold=3.5):
    rr_list = []
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        rec = user_ratings[:k]
        rr = 0
        for ind in range(len(rec)):
            if rec[ind][1] >= threshold:
                index = ind + 1
                rr = rr + (1 / index)
        rr_list.append(rr)
    return np.mean(rr_list)


def getMRR2(predictions, k=10, threshold=3.5):
    rr_list = []
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        rec = user_ratings[:k]
        rr = 0
        for ind in range(len(rec)):
            if rec[ind][1] >= threshold:
                index = ind + 1
                rr = 1 / index
                break
        rr_list.append(rr)
    return np.mean(rr_list)

# def getMRR2test(predictions, k=10, threshold=3.5):
#     rr_list = []
#     # First map the predictions to each user.
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))
#         if true_r <= threshold:
#             print(true_r)
#     for uid, user_ratings in user_est_true.items():
#         # Sort user ratings by estimated value
#         user_ratings.sort(key=lambda x: x[0], reverse=True)
#         rec = user_ratings[:k]
#         rr = 0
#         for ind in range(len(rec)):
#             if rec[ind][1] >= threshold:
#                 index = ind + 1
#                 rr = 1 / index
#                 break
#         rr_list.append(rr)
#     return np.mean(rr_list)
