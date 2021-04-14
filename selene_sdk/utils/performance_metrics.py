from collections import defaultdict, namedtuple
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import rankdata

logger = logging.getLogger("selene")

Metric = namedtuple("Metric", ["fn", "data"])


def visualize_roc_curves(prediction,
                         target,
                         output_dir,
                         report_gt_feature_n_positives=50,
                         style="seaborn-colorblind",
                         fig_title="Feature ROC curves",
                         dpi=500):
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
            plt.plot(fpr, tpr, 'r-', color="black", alpha=0.3, lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "roc_curves.svg"),
                format="svg",
                dpi=dpi)


def visualize_precision_recall_curves(
        prediction,
        target,
        output_dir,
        report_gt_feature_n_positives=50,
        style="seaborn-colorblind",
        fig_title="Feature precision-recall curves",
        dpi=500):
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision, 'r-',
                color="black", alpha=0.3, lw=1, where="post")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.svg"),
                format="svg",
                dpi=dpi)


def compute_score(prediction, target, metric_fn, report_gt_feature_n_positives=10):
    feature_scores = np.ones(target.shape[1]) * np.nan
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if len(np.unique(feature_targets)) > 0 and \
                np.count_nonzero(feature_targets) > report_gt_feature_n_positives:
            try:
                feature_scores[index] = metric_fn(
                    feature_targets, feature_preds)
            except ValueError:  # do I need to make this more generic?
                continue
    valid_feature_scores = [s for s in feature_scores if not np.isnan(s)]  # Allow 0 or negative values.
    if not valid_feature_scores:
        return None, feature_scores
    average_score = np.average(valid_feature_scores)
    return average_score, feature_scores


def get_feature_specific_scores(data, get_feature_from_index_fn):
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_index_fn(index)
        if not np.isnan(score):
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


def auc_u_test(labels, predictions):
    len_pos = int(np.sum(labels))
    len_neg = len(labels) - len_pos
    rank_sum = np.sum(rankdata(predictions)[labels == 1])
    u_value = rank_sum - (len_pos * (len_pos + 1)) / 2
    auc = u_value / (len_pos * len_neg)
    return auc


class PerformanceMetrics(object):
    def __init__(self,
                 get_feature_from_index_fn,
                 report_gt_feature_n_positives=10,
                 metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score)):
        self.skip_threshold = report_gt_feature_n_positives
        self.get_feature_from_index = get_feature_from_index_fn
        self.metrics = dict()
        for k, v in metrics.items():
            self.metrics[k] = Metric(fn=v, data=[])

    def add_metric(self, name, metric_fn):
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, prediction, target):
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                prediction, target, metric.fn,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def visualize(self, prediction, target, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        if "roc_auc" in self.metrics:
            visualize_roc_curves(
                prediction, target, output_dir,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)
        if "average_precision" in self.metrics:
            visualize_precision_recall_curves(
                prediction, target, output_dir,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)

    def write_feature_scores_to_file(self, output_path):
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.get_feature_from_index)
            for feature, score in feature_score_dict.items():
                if score is None:
                    feature_scores[feature] = None
                else:
                    feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["class"] + metric_cols)
        with open(output_path, 'w+') as file_handle:
            file_handle.write("{0}\n".format(cols))
            for feature, metric_scores in sorted(feature_scores.items()):
                if not metric_scores:
                    file_handle.write("{0}\t{1}\n".format(feature, "\t".join(["NA"] * len(metric_cols))))
                else:
                    metric_score_cols = '\t'.join(
                        ["{0:.4f}".format(s) for s in metric_scores.values()])
                    file_handle.write("{0}\t{1}\n".format(feature,
                                                          metric_score_cols))
        return feature_scores
