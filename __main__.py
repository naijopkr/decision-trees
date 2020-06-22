import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/kyphosis.csv')
df.head()

sns.pairplot(df, hue='Kyphosis')

from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision trees
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

# Tree visualization
from IPython.display import Image
from io import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns[1:])
features

dot_data = StringIO()
export_graphviz(
    dtree,
    out_file=dot_data,
    feature_names=features,
    filled=True,
    rounded=True
)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

# Random forests
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

y_rfc_pred = rfc.predict(X_test)

# Evaluate
def print_cm(pred_cm):
    pred_false, pred_true = pred_cm
    tn, fn = pred_false
    fp, tp = pred_true
    print(
        f'TN\tFN\tFP\tTP\n{tn}\t{fn}\t{fp}\t{tp}'
    )

dtree_cm = confusion_matrix(y_test, y_pred)
print_cm(dtree_cm)

rfc_cm = confusion_matrix(y_test, y_rfc_pred)
print_cm(rfc_cm)

dtree_cr = classification_report(y_test, y_pred)
rfc_cr = classification_report(y_test, y_rfc_pred)

print(dtree_cr)
print(rfc_cr)
