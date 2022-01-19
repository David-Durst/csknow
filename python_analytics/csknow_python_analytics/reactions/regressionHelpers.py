from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics, preprocessing, pipeline
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from collections import Counter
import numpy as np

def makeLogReg(df, cols, name, grouped_str, plot_folder):
    plot_str = grouped_str + '_' + name.lower().replace(' ', '_') + '.png'

    plt.clf()
    X_df = df[cols]
    y_series = df['hacking']

    #lr_model = SVC(gamma='auto')
    lr_model = LogisticRegression()
    scalar = preprocessing.StandardScaler()
    pipe = pipeline.Pipeline([('transformer', scalar), ('estimator', lr_model)])
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(pipe, X_df, y_series, cv=skf)
    print(f'''{name} {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}''')

    good_roc_figure = plt.figure(figsize=(10,10))
    good_roc_axis = good_roc_figure.gca()
    kfoldROCCurve(pipe, skf, X_df, y_series, good_roc_axis, name)
    good_roc_figure.savefig(plot_folder + 'roc_' + plot_str)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, stratify=y_series, test_size=0.2)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(10, 10))
    sn.set(font_scale=2.0)
    confusion_matrix_heatmap = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 24}, ax=ax)
    confusion_matrix_heatmap.set_yticklabels(confusion_matrix_heatmap.get_ymajorticklabels(), fontsize=24)
    confusion_matrix_heatmap.set_xticklabels(confusion_matrix_heatmap.get_xmajorticklabels(), fontsize=24)
    confusion_matrix_heatmap.set_ylabel("Actual", fontsize=24)
    confusion_matrix_heatmap.set_xlabel("Predicted", fontsize=24)
    plt.suptitle(name + ' Labeled Confusion Matrix', fontsize=30)
    confusion_matrix_figure = confusion_matrix_heatmap.get_figure()
    confusion_matrix_figure.savefig(plot_folder + 'confusion_' + plot_str)
    sn.reset_orig() # disable seaborn style

    result = X_test.copy()
    result['label'] = y_test
    result['pred'] = y_pred
    bad_results = result[result['label'] != result['pred']]
    print(df.iloc[bad_results.index])

def kfoldROCCurve(pipe: pipeline.Pipeline, skf: StratifiedKFold, X_df, y_series, ax, name):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(skf.split(X_df, y_series)):
        pipe.fit(X_df.iloc[train], y_series[train])
        viz = RocCurveDisplay.from_estimator(
            pipe,
            X_df.iloc[test],
            y_series[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=name + "Receiver Operating Characteristic",
    )
    ax.legend(loc="lower right")