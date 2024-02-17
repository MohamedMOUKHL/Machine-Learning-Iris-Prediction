# from sklearn.datasets import load_iris
# import pandas as pd
# import pandas.plotting as pdpl
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)


# iris_dataset = load_iris()
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], 
# random_state=0)

# #print("X_train shape: {}".format(X_train.shape))
# #print("X_test shape: {}".format(X_test.shape))

# iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
# grr = pdpl.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8)



# knn.fit(X_train, y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, 
# n_jobs=1, n_neighbors=1, p=2, weights='uniform')


# y_pred = knn.predict(X_test)

# # Évaluez la performance du modèle
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# confusion_mat = confusion_matrix(y_test, y_pred)

# #print(f"Accuracy: {accuracy}")
# #print("Classification Report:\n", report)
# #print("Confusion Matrix:\n", confusion_mat)




# nouvelles_mesures = [[5.1, 3.5, 1.4, 0.2]]  
# predictions = knn.predict(nouvelles_mesures)
# #print("Predictions:", predictions)


# with open('knn_model.pkl', 'wb') as model_file:
#     pickle.dump(knn, model_file)