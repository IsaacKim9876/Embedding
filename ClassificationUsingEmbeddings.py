# imports
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# load data
datafile_path = "/Users/isaackim/Python/TestDataSets/data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to array

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.Score, test_size=0.2, random_state=42
)

# train random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)


"""
decision_trees = clf.estimators_
for i, tree in enumerate(decision_trees):
    from sklearn.tree import export_graphviz
    import pydotplus
    from IPython.display import Image
    dot_data = export_graphviz(tree,
                               out_file=None,
                               feature_names=df.embedding.values[0].astype(str),
                               filled=True,
                               rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree" + str(i) + ".pdf")
    
"""