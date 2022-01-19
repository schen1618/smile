import os
from re import T
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold


def getBestSVM(
    Classifier, classifier_args, bags, labels, bounds, num_samples, num_splits=5
):
    def trainable():
        curr_best_auc = None
        curr_best_config = None
        # Hyperparameters
        # Iterative training function - can be any arbitrary training procedure.
        for i in range(num_samples):
            config = {v: np.random.uniform(*bounds[v]) for v in bounds}
            accuracies = []
            aucs = []
            skf = StratifiedKFold(n_splits=num_splits)
            for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
                train_bags = bags[train_index]
                train_labels = labels[train_index]
                test_bags = bags[test_index]
                test_labels = labels[test_index]
                classifier = Classifier(
                    **classifier_args,
                    **config,
                )
                classifier.fit(train_bags, train_labels)
                y_confidence = classifier.predict(test_bags)
                y = np.sign(y_confidence)
                accuracies.append(accuracy_score(test_labels, y))
                aucs.append(roc_auc_score(test_labels, y_confidence))
            if curr_best_auc is None and curr_best_config is None:
                curr_best_auc = np.average(aucs)
                curr_best_config = config
            else:
                if curr_best_auc < np.average(aucs):
                    curr_best_auc = np.average(aucs)
                    curr_best_config = config
        return curr_best_config, curr_best_auc

    # points_to_evaluate = [{"C": 25.6445, "gamma": 98.7764}]
    best_config, best_auc = trainable()

    print("Best config: ", best_config)
    print("AUC: ", best_auc)
    return Classifier(**classifier_args, **best_config), best_config
