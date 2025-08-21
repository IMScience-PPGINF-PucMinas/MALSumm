# -*- coding: utf-8 -*-
import numpy as np

def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """

    user_summary = np.atleast_2d(user_summary)

    #print(f"============Predicted summary length: {len(predicted_summary)}, User summary shape: {user_summary.shape}")
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        sum_s = sum(S)
        sum_g = sum(G)

        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum_s if sum_s > 0 else 0
        recall = sum(overlapped) / sum_g if sum_g > 0 else 0
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)
