# -*- coding: UTF-8 -*-
import statistics
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import eli5
import xgboost
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
from missingpy import MissForest
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Set the maximum number of rows, columns, and width of pandas output table
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class MLPipline():

    def __init__(self):
        self.model_lr = LogisticRegression()
        self.model_knn = KNeighborsClassifier()
        self.model_dt = DecisionTreeClassifier()
        self.model_rf = RandomForestClassifier()
        self.model_et = ExtraTreesClassifier()
        self.model_svc = SVC(probability=True)
        self.model_mlp = MLPClassifier(max_iter=4000)
        self.model_gdbt = GradientBoostingClassifier()
        self.model_xgb = xgboost.XGBClassifier()
        self.model_aboost = AdaBoostClassifier()
        self.model_aboost_lr = AdaBoostClassifier(base_estimator=self.model_lr)
        self.model_bagging = BaggingClassifier(base_estimator=self.model_xgb)
        self.estimaters = [('lr', self.model_lr), ('knn', self.model_knn), ('dt', self.model_dt), ('rf', self.model_rf),
                      ('et', self.model_et), ('svc', self.model_svc), ('mlp', self.model_mlp), ('gdbt', self.model_gdbt),
                      ('xgb', self.model_xgb)]
        self.model_voting = VotingClassifier(estimators=self.estimaters, voting='soft')
        self.model_list = [self.model_lr, self.model_knn, self.model_svc, self.model_dt, self.model_mlp, self.model_rf, self.model_et, self.model_gdbt, self.model_xgb,
                      self.model_aboost]

    # split x and y
    def split_x_y(self, scaler_data):
        X = scaler_data.iloc[:, 2:]
        Y = scaler_data['Target']
        return X, Y

    # Normalization
    def stand_data(self, data):
        scaler_data = MinMaxScaler().fit_transform(data)
        scaler_data = pd.DataFrame(scaler_data, columns=data.columns)
        return scaler_data

    # Calculate the mean and standard deviation
    def CI_95(self, cal_list):
        mean = statistics.mean(cal_list)
        std = statistics.stdev(cal_list)
        return mean, std

    # lasso
    def lasso(self, data, labels, threshold):
        estimator = LassoCV(random_state=7)
        sf = SelectFromModel(estimator=estimator, threshold=threshold)
        select_data = sf.fit_transform(data, labels)
        select_index = data.columns[sf.get_support()]
        return select_data, select_index

    # DCA
    def calculate_net_benefit_model(self, thresh_group, y_pred_score, y_label):
        net_benefit_model = np.array([])
        for thresh in thresh_group:
            y_pred_label = y_pred_score > thresh
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
            n = len(y_label)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model = np.append(net_benefit_model, net_benefit)
        return net_benefit_model

    # DCA
    def calculate_net_benefit_all(self, thresh_group, y_label):
        net_benefit_all = np.array([])
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total = tp + tn
        for thresh in thresh_group:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            net_benefit_all = np.append(net_benefit_all, net_benefit)
        return net_benefit_all

    # plot DCA
    def plot_DCA(self, ax, thresh_group, net_benefit_model, net_benefit_all):
        # Plot
        ax.plot(thresh_group, net_benefit_model, color='crimson', label='Model')
        ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
        ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

        # Fill，the model is shown to be better than treat all and treat none
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

        # Figure Configuration
        ax.set_xlim(0, 1)
        ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
        ax.set_xlabel(
            xlabel='Threshold Probability',
            fontdict={'family': 'Times New Roman', 'fontsize': 15}
        )
        ax.set_ylabel(
            ylabel='Net Benefit',
            fontdict={'family': 'Times New Roman', 'fontsize': 15}
        )
        ax.grid('major')
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        ax.legend(loc='upper right')
        return ax

    # plot violin
    def plot_violinplot(self, data_violin):
        sns.violinplot(x='group',
                       y='score',
                       width=0.4,
                       data=data_violin,
                       order=['Train set', 'Test set'],
                       hue='category2',
                       split=False,
                       palette='muted',
                       inner='box',
                       scale='area'
                       )
        plt.legend(loc='lower right', ncol=2)
        plt.show()

    # 5-fold train and test generate
    def five_fold(self, X_train, Y_train):
        kf = KFold(n_splits=5, shuffle=True, random_state=7)
        for train_index, test_index in kf.split(X_train):
            Xtrain, Xtest = X_train.loc[train_index], X_train.loc[test_index]
            Ytrain, Ytest = Y_train.loc[train_index], Y_train.loc[test_index]
            yield Xtrain, Ytrain, Xtest, Ytest

    # model evaluation by 5-fold cross validation
    def model_evaluation(self, model_list, X_train, Y_train, *args):
        kf = KFold(n_splits=5, shuffle=True, random_state=7)
        result = []
        for model in [model_list]:
            acc_list = []
            precision_list = []
            recall_list = []
            f1score_list = []
            auc_list = []
            for train_index, test_index in kf.split(X_train):
                Xtrain, Xtest = X_train.loc[train_index], X_train.loc[test_index]
                Ytrain, Ytest = Y_train.loc[train_index], Y_train.loc[test_index]
                model.fit(Xtrain, np.ravel(Ytrain))
                if args:
                    Xtest = args[0]
                    Ytest = args[1]
                y_pred = model.predict(Xtest)
                y_score = model.predict_proba(Xtest)[:, 1]
                acc = accuracy_score(Ytest, y_pred)
                precision = precision_score(Ytest, y_pred)
                recall = recall_score(Ytest, y_pred)
                f1score = f1_score(Ytest, y_pred)
                auc = roc_auc_score(Ytest, y_score)
                acc_list.append(acc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1score_list.append(f1score)
                auc_list.append(auc)
            name = model.__class__.__name__
            acc_mean, acc_std = self.CI_95(acc_list)
            precision_mean, precision_std = self.CI_95(precision_list)
            recall_mean, recall_std = self.CI_95(recall_list)
            f1score_mean, f1score_std = self.CI_95(f1score_list)
            auc_mean, auc_std = self.CI_95(auc_list)
            result.append(
                [name, acc_mean, acc_std, precision_mean, precision_std, recall_mean, recall_std, f1score_mean, f1score_std,
                 auc_mean, auc_std])

        result = pd.DataFrame(result, columns=['model_name', 'acc_mean', 'acc_std', 'precision_mean', 'precision_std',
                                               'recall_mean', 'recall_std',
                                               'f1score_mean', 'f1score_std', 'auc_mean', 'auc_std'])
        return result

    # missForest filling
    def miss_filing(self, data):
        forestimp = MissForest(random_state=7)
        missing_filling_data = forestimp.fit_transform(data)
        s = pd.DataFrame(missing_filling_data, columns=data.columns)
        return s

    # plot roc curve
    def roc_curve(self, x_train, y_train, x_test, y_test):
        roc_auc_list = []
        fpr_list = []
        tpr_list = []
        for model in self.model_list:
            model.fit(x_train, np.ravel(y_train))
            y_score = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr_list[0], tpr_list[0], color='darkorange', lw=lw, label='Logistic Regression (area = %0.2f)' % roc_auc_list[0])
        plt.plot(fpr_list[1], tpr_list[1], color='black', lw=lw, label='KNeighbors (area = %0.2f)' % roc_auc_list[1])
        plt.plot(fpr_list[2], tpr_list[2], color='blue', lw=lw, label='SVC (area = %0.2f)' % roc_auc_list[2])
        plt.plot(fpr_list[3], tpr_list[3], color='brown', lw=lw, label='DecisionTree (area = %0.2f)' % roc_auc_list[3])
        plt.plot(fpr_list[4], tpr_list[4], color='coral', lw=lw, label='MLP (area = %0.2f)' % roc_auc_list[4])
        plt.plot(fpr_list[5], tpr_list[5], color='gold', lw=lw, label='RandomForest (area = %0.2f)' % roc_auc_list[5])
        plt.plot(fpr_list[6], tpr_list[6], color='maroon', lw=lw, label='ExtraTrees (area = %0.2f)' % roc_auc_list[6])
        plt.plot(fpr_list[7], tpr_list[7], color='pink', lw=lw, label='GradientBoosting Tree (area = %0.2f)' % roc_auc_list[7])
        plt.plot(fpr_list[8], tpr_list[8], color='red', lw=lw, label='XGB (area = %0.2f)' % roc_auc_list[8])
        plt.plot(fpr_list[9], tpr_list[9], color='darkorchid', lw=lw, label='Final model(area=%0.2f)' % roc_auc_list[9])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # 绘制标准线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    # plot calibration curve
    def a(self, y_test, y_prob):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10, normalize=True)
        plt.figure(figsize=(10, 10), dpi=60)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Best model")
        plt.xlabel('Threshold', size=20)
        plt.ylabel("Fraction of positives", size=20)
        plt.ylim([-0.05, 1.05])
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc="lower right")
        plt.show()

    # Expected calibration error
    def expected_calibration_error(y, proba, bins='fd'):
        bin_count, bin_edges = np.histogram(proba, bins=bins)
        n_bins = len(bin_count)
        bin_edges[0] -= 1e-8
        bin_id = np.digitize(proba, bin_edges, right=True) - 1
        bin_ysum = np.bincount(bin_id, weights=y, minlength=n_bins)
        bin_probasum = np.bincount(bin_id, weights=proba, minlength=n_bins)
        bin_ymean = np.divide(bin_ysum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)
        bin_probamean = np.divide(bin_probasum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)
        ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
        return ece

    # brier score
    def brier_score(self, y_prob, y_test):
        y_prob_new = []
        for i in y_prob:
            stand = (i - min(y_prob)) / (max(y_prob) - min(y_prob))
            y_prob_new.append(stand)
        brier_score = brier_score_loss(y_test, y_prob_new)
        return brier_score

    # SHAP
    def shap_explainer(self, md, x_train, y_train, x_test):
        model = md.fit(x_train, y_train)
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, show=False)
        plt.show()

    # Permutation Importance
    def permutation_importance(self, md, x_train, y_train, x_test, y_test):
        model = md.fit(x_train, y_train)
        perm = PermutationImportance(model, random_state=7).fit(x_test, y_test)
        el5 = eli5.show_weights(perm)
        return el5

