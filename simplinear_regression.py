import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load dataset
dataset = datasets.load_wine(as_frame=True)

# Set data and target
x, y = dataset.data, dataset.target

# Create dataframe
data = pd.concat([x, y], axis=1)
data.rename(columns={"target": "Wine quality"}, inplace=True)

# Scatter matrix plot
scatter_matrix_fig = plt.figure(figsize=(15, 15))
axes = pd.plotting.scatter_matrix(data, alpha=0.5, figsize=(15, 15), diagonal="kde")

# Adjust axes for readability
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(45)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_horizontalalignment('right')

scatter_matrix_fig.suptitle("Scatter Matrix of Wine Dataset", fontsize=20)
scatter_matrix_fig.savefig("scatter_matrix.jpg", dpi=300)
plt.close(scatter_matrix_fig)

# Correlation matrix plot
correlation_matrix = data.corr()

fig, ax = plt.subplots(figsize=(12, 10))  # Adjust size for clarity
sns.heatmap(
    correlation_matrix,
    ax=ax,
    cmap="vlag",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 10},
    cbar=True,
    square=True,
    linewidths=0.5,
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.columns,
)

# Rotate labels for better visibility
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.title("Correlation Matrix for Wine Dataset", fontsize=16)
fig.savefig("correlation_matrix.jpg", dpi=300)
plt.close(fig)

# Simple linear regression
simple_x = x.alcalinity_of_ash.to_numpy().reshape(len(x), 1)
simple_x_train, simple_x_test, y_train, y_test = train_test_split(
    simple_x, y, random_state=0, test_size=0.2)

model = LinearRegression()
model.fit(simple_x_train, y_train)
simple_y_pred = model.predict(simple_x_test)

# Linear regression plot
fig, ax = plt.subplots(figsize=(15, 7))
plt.scatter(simple_x_test, y_test, color="black", label="Real Values")
plt.scatter(simple_x_test, simple_y_pred, color="red", label="Predicted Values")
plt.plot(simple_x_test, simple_y_pred, color="blue", label="Regression Line")
plt.legend()
plt.title("Wine Quality by Alkalinity of Ash")
ax.set_xlabel("Alkalinity of Ash (AOA)")
ax.set_ylabel("Wine Quality")
fig.savefig("linear_regression.jpg", dpi=300)
plt.close(fig)

