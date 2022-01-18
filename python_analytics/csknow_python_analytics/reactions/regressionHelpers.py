from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing, pipeline
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from collections import Counter

def makeLogReg(df, cols, name, grouped_str, plot_folder):
    plot_str = grouped_str + '_' + name.lower().replace(' ', '_') + '.png'

    plt.clf()
    X_df = df[cols]
    y_series = df['hacking']

    lr_model = LogisticRegression()
    scalar = preprocessing.StandardScaler()
    pipe = pipeline.Pipeline([('transformer', scalar), ('estimator', lr_model)])
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(pipe, X_df, y_series, cv=skf)
    print(f'''{name} {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}''')

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, stratify=y_series, test_size=0.2)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_pred_prob = pipe.predict_proba(X_test)
    y_pred_prob_pos = y_pred_prob[:, 1]

    # no skill prediction
    (most_common_class_value, most_common_class_count) = \
        Counter(y_test).most_common(1)[0]
    no_skill_pred = [most_common_class_value for _ in range(len(y_test))]
    print(f'''percent hacking: {most_common_class_count * 1.0 / len(y_test)}''')

    # calculate auc scores
    no_skill_auc = metrics.roc_auc_score(y_test, no_skill_pred)
    y_pred_auc = metrics.roc_auc_score(y_test, y_pred_prob_pos)
    print(f'''No Skill: ROC AUC={no_skill_auc}''')
    print(f'''Logistic: ROC AUC={y_pred_auc}''')

    no_skill_fpr, no_skill_tpr, _ = metrics.roc_curve(y_test, no_skill_pred)
    y_pred_fpr, y_pred_tpr, _ = metrics.roc_curve(y_test, y_pred_prob_pos)

    #mpl.style.use('default')
    roc_figure = plt.figure(figsize=(10,10))
    roc_axis = roc_figure.gca()
    roc_axis.plot(no_skill_fpr, no_skill_tpr, linestyle='--', label='No Skill')
    roc_axis.plot(y_pred_fpr, y_pred_tpr, linestyle='-', marker='o', label='Logistic')
    roc_axis.legend()
    roc_figure.savefig(plot_folder + 'roc_' + plot_str)
    plt.clf()

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