import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data', header=None, names=column_names)


print("Descriptive statistics")
print(df.info())


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




print("perceptron parameters")


mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100),
                    max_iter= 1000,
                    activation='logistic',
                    solver='sgd',
                    random_state=42)


# mlp = MLPClassifier()


mlp.fit(X_train_scaled, y_train)


mlp_pred = mlp.predict(X_test_scaled)


print("\nperceptron results:")
print("accuracy:", accuracy_score(y_test, mlp_pred))
print("\nconfusion matrix")
print(confusion_matrix(y_test, mlp_pred))
print("\nclassification reports:")
print(classification_report(y_test, mlp_pred))


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('"true" label')
    plt.xlabel('predicted Label')
    plt.show()


plot_confusion_matrix(y_test, mlp_pred, 'confusion matrix')
