import numpy as np


class CalculateF1Score:
    def __init__(self, Y, y_pred, class_num, verbose=False):
        self.class_num = class_num
        self.epsilon = 10 ** -10
        self.y_true_normalized = self.normalize_y(Y)
        self.y_pred_normalized = self.normalize_y(y_pred)
        self.class_true_positives = [
            np.sum(self.class_true_positive(i)) for i in range(class_num)
        ]
        if verbose:
            print(self.calculateScore())

    def calculateScoreByClass(self):
        return [self.class_f_measure(i) for i in range(self.class_num)]

    def calculateScore(self):
        return self.macro_f_measure()

    def normalize_y(self, y):
        if len(y.shape) == 1:
            return y.astype("int32")
        else:
            return np.argmax(y, axis=1)

    def class_true_positive(self, class_label):
        return np.logical_and(
            self.y_true_normalized == class_label, self.y_pred_normalized == class_label
        )

    def class_precision(self, class_label):
        return self.class_true_positives[class_label] / (
            np.sum(self.y_pred_normalized == class_label) + self.epsilon
        )

    def class_recall(self, class_label):
        return self.class_true_positives[class_label] / (
            np.sum(self.y_true_normalized == class_label) + self.epsilon
        )

    def class_f_measure(self, class_label):
        precision = self.class_precision(class_label)
        recall = self.class_recall(class_label)
        return (2 * precision * recall) / (precision + recall + self.epsilon)

    def macro_precision(self):
        return sum([self.class_precision(i) for i in range(self.class_num)]) / (
            self.class_num + self.epsilon
        )

    def macro_recall(self):
        return sum([self.class_recall(i) for i in range(self.class_num)]) / (
            self.class_num + self.epsilon
        )

    def macro_f_measure(self):
        return sum([self.class_f_measure(i) for i in range(self.class_num)]) / (
            self.class_num + self.epsilon
        )
