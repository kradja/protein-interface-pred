from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve
import pandas as pd
import os
from pathlib import Path
from src.utils import visualization_utils


class BinaryClassEvaluation:
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path,
                 output_prefix):
        self.df = df
        self.evaluation_settings = evaluation_settings
        self.evaluation_output_file_base_path = evaluation_output_file_base_path
        self.visualization_output_file_base_path = visualization_output_file_base_path
        self.output_prefix = output_prefix

        self.evaluation_output_file_path = os.path.join(evaluation_output_file_base_path, output_prefix)
        Path(os.path.dirname(self.evaluation_output_file_path)).mkdir(parents=True, exist_ok=True)

        self.visualization_output_file_path = os.path.join(visualization_output_file_base_path, output_prefix)
        Path(os.path.dirname(self.visualization_output_file_path)).mkdir(parents=True, exist_ok=True)

        self.evaluation_metrics_df = None
        self.roc_curves_df = None
        self.pr_curves_df = None
        self.itr_col = "itr"
        self.experiment_col = "experiment"
        self.y_true_col = "y_true"
        self.itrs = df["itr"].unique()
        self.y_pred_column = "y_pred"
        self.pos_label = evaluation_settings["pos_label"]
        self.neg_label = evaluation_settings["neg_label"]

    def execute(self):
        experiments = self.df[self.experiment_col].unique()
        result = []
        roc_curves = []
        pr_curves = []
        for experiment in experiments:
            experiment_df = self.df[self.df[self.experiment_col] == experiment]
            for itr in self.itrs:
                result_itr = {self.itr_col: itr, self.experiment_col: experiment}
                df_itr = experiment_df[experiment_df[self.itr_col] == itr]
                if self.evaluation_settings["auroc"]:
                    roc_curve_itr, auroc_itr = self.compute_auroc(df_itr)
                    # individual ROC curves
                    roc_curve_itr[self.itr_col] = itr
                    roc_curve_itr[self.experiment_col] = experiment
                    roc_curves.append(roc_curve_itr)
                    result_itr["auroc"] = auroc_itr
                if self.evaluation_settings["auprc"]:
                    pr_curve_itr, auprc_itr = self.compute_auprc(df_itr)
                    # individual Precision-Recall curves
                    pr_curve_itr[self.itr_col] = itr
                    pr_curve_itr[self.experiment_col] = experiment
                    pr_curves.append(pr_curve_itr)
                    result_itr["auprc"] = auprc_itr
                if self.evaluation_settings["accuracy"]:
                    acc_itr = self.compute_accuracy(df_itr)
                    result_itr["accuracy"] = acc_itr
                if self.evaluation_settings["f1"]:
                    f1_itr = self.compute_f1(df_itr)
                    result_itr["f1"] = f1_itr
                result.append(result_itr)
        self.evaluation_metrics_df = pd.DataFrame(result)
        self.evaluation_metrics_df.to_csv(self.evaluation_output_file_path + "_evaluation_metrics.csv")

        if len(roc_curves) > 0:
            self.roc_curves_df = pd.concat(roc_curves, ignore_index=True)
            self.roc_curves_df.to_csv(self.evaluation_output_file_path + "_roc_curves.csv")
        if len(roc_curves) > 0:
            self.pr_curves_df = pd.concat(pr_curves, ignore_index=True)
            self.pr_curves_df.to_csv(self.evaluation_output_file_path + "_pr_curves.csv")

        self.plot_visualizations()
        return

    def plot_visualizations(self):
        if self.evaluation_settings["accuracy"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "accuracy", self.visualization_output_file_path + "_accuracy_boxplot.png")
        if self.evaluation_settings["f1"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "f1", self.visualization_output_file_path + "_f1_boxplot.png")
        if self.evaluation_settings["auroc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auroc",
                                         self.visualization_output_file_path + "_auroc_boxplot.png",baseline=0.5)
            # visualization_utils.curve_plot(df=self.roc_curves_df, x_col="fpr", y_col="tpr",
            #                                color_group_col=self.experiment_col, style_group_col=self.itr_col,
            #                                output_file_path=self.visualization_output_file_path + "_roc_curves.png")
        if self.evaluation_settings["auprc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auprc",
                                         self.visualization_output_file_path + "_auprc_boxplot.png", baseline=0.1)
            # visualization_utils.curve_plot(df=self.pr_curves_df, x_col="recall", y_col="precision",
            #                                color_group_col=self.experiment_col, style_group_col=self.itr_col,
            #                                output_file_path=self.visualization_output_file_path + "_precision_recall_curves.png")
        return

    def compute_accuracy(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return accuracy_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred)

    def compute_f1(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return f1_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred, pos_label=self.pos_label)

    def compute_auroc(self, df_itr):
        # The function roc_auc_score returns {1 - true_AUROC_score}
        # It considers the compliment of the prediction probabilities in the computation of the area
        # roc_auc_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr["Human"].values)

        # Hence we use roc_curve to compute fpr, tpr followed by auc to compute the AUROC.
        fpr, tpr, _ = roc_curve(y_true=df_itr[self.y_true_col].values, y_score=df_itr[self.y_pred_column].values)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr}), auc(fpr, tpr)

    def compute_auprc(self, df_itr):
        auprc_score = average_precision_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr[self.y_pred_column].values, pos_label=self.pos_label)
        precision, recall, _ = precision_recall_curve(y_true=df_itr[self.y_true_col].values, probas_pred=df_itr[self.y_pred_column].values, pos_label=self.pos_label)
        return pd.DataFrame({"precision": precision, "recall": recall}), auprc_score

    def convert_probability_to_prediction(self, df_itr, threshold=0.5):
        y_pred_prob = df_itr[self.y_pred_column].values
        y_pred = [self.pos_label if y >= threshold else self.neg_label for y in y_pred_prob]
        return y_pred
