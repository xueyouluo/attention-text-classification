from math import log
import os
import json

class Judger:
    # Initialize Judger, with the path of label
    def __init__(self, label_path):
        self.label_dic = {}

        f = open(label_path, "r")
        self.task_cnt = 0
        for line in f:
            self.label_dic[line.strip()] = self.task_cnt
            self.task_cnt += 1

    # Gen new results according to the truth and users output
    def gen_new_result(self, result, truth, label):
        s1 = set(label)
        s2 = set()
        for name in truth:
            s2.add(self.label_dic[name])

        for a in range(0, self.task_cnt):
            in1 = a in s1
            in2 = a in s2
            if in1:
                if in2:
                    result[a]["TP"] += 1
                else:
                    result[a]["FP"] += 1
            else:
                if in2:
                    result[a]["FN"] += 1
                else:
                    result[a]["TN"] += 1

        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_value(res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def get_f1(self, arr):
        sumf = 0
        y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for x in arr:
            p, r, f = self.get_value(x)
            sumf += f
            for z in x.keys():
                y[z] += x[z]

        _, __, f_ = self.get_value(y)
        macro_f1 = f_
        micro_f1 = sumf * 1.0 / len(arr)
        return macro_f1, micro_f1

    def evaluation(self, truth_path, outputs):
        result = []
        for a in range(0, self.task_cnt):
            result.append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        inf = open(truth_path, "r")
        outputs = iter(outputs)
        for line in inf:
            ground_truth = json.loads(line)["labels"]
            user_output = next(outputs)

            result = self.gen_new_result(result, ground_truth, user_output)

        macro_f1, micro_f1 = self.get_f1(result)
        return macro_f1, micro_f1
