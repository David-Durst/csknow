from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing, pipeline
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

def makeLogReg(df, cols, name, plot_folder):
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

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(10, 10))
    sn.set(font_scale=2.0)
    confusion_matrix_heatmap = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 24}, ax=ax)
    confusion_matrix_heatmap.set_yticklabels(confusion_matrix_heatmap.get_ymajorticklabels(), fontsize=24)
    confusion_matrix_heatmap.set_xticklabels(confusion_matrix_heatmap.get_xmajorticklabels(), fontsize=24)
    confusion_matrix_heatmap.set_ylabel("Actual", fontsize=24)
    confusion_matrix_heatmap.set_xlabel("Predicted", fontsize=24)
    plt.title(name + ' Labeled Confusion Matrix', fontsize=30)
    confusion_matrix_figure = confusion_matrix_heatmap.get_figure()
    confusion_matrix_figure.savefig(plot_folder + name + '_grouped_confusion_matrix__hand_vs_cpu__hacking_vs_legit.png')

    result = X_test.copy()
    result['label'] = y_test
    result['pred'] = y_pred
    bad_results = result[result['label'] != result['pred']]
    print(df.iloc[bad_results.index])