import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

# Loading the dataset
dataset = datasets.load_wine(as_frame=True)

# Setting data and target
x, y = dataset.data, dataset.target

# Creating dataframe
data = pd.concat([x, y], axis=1)
data.rename(columns={"target": "Wine quality"}, inplace=True)  # Rename target column

print(dataset.DESCR)

# Inspection
# Select some sample indices
sample_indices = np.linspace(0, len(x) - 4, 4, dtype=int)
sample_indices = [index for i in sample_indices for index in range(i, i + 4)]

# Make sure indices do not go out of bounds
sample_indices = sample_indices[:len(x)]  # Truncate in case it exceeds the data length

sample_data = data.iloc[sample_indices, :]
# Select and print the data table for those sample indices
styled_data = sample_data.style.set_properties(**{
    "text-align": "center",
}).set_properties(**{
    "border-left": "4px solid black"
}, subset=['Wine quality']).set_table_styles([
    dict(selector="th", props=[("font-size", "13px")]),
    dict(selector="td", props=[("font-size", "11px")]),
]).background_gradient()

# Save the styled DataFrame as an HTML file
styled_data.to_html('styled_output.html')
pd.set_option('float_format', '{:g}'.format)
data.describe()

# Save descriptive statistics as a CSV
data.describe().to_csv("data_description.csv")

# Scatter matrix
scatter_matrix_fig, axes = plt.subplots(figsize=(15, 15))
axes = pd.plotting.scatter_matrix(data, figsize=(15, 15))

# Fix y-axis label formatting
new_labels = [
    round(float(i.get_text()), 2) for i in axes[0, 0].get_yticklabels()
]
_ = axes[0, 0].set_yticklabels(new_labels)
scatter_matrix_fig.savefig("scatter_matrix.jpg")
plt.close(scatter_matrix_fig)

# Calculate correlation matrix using NumPy
correlation_matrix = np.corrcoef(data.values.T)

# Plot correlation matrix using seaborn
fig, ax = plt.subplots(figsize=(8, 8))
tick_labels = list(x.columns) + ['Wine quality']
hm = sns.heatmap(
    correlation_matrix,
    ax=ax,
    cbar=True,  # Show colorbar
    cmap="vlag",  # Specify colormap
    vmin=-1,  # Min. value for colormapping
    vmax=1,  # Max. value for colormapping
    annot=True,  # Show the value of each cell
    square=True,  # Square aspect ratio in cell sizing
    fmt='.2f',  # Float formatting
    annot_kws={'size': 12},  # Font size of the values displayed within the cells
    xticklabels=tick_labels,  # x-axis labels
    yticklabels=tick_labels)  # y-axis labels
plt.tight_layout()
fig.savefig("correlation_matrix.jpg")
plt.close(fig)

# Simple linear regression
from sklearn.model_selection import train_test_split

# Create a vector of the single predictor values
simple_x = x.alcalinity_of_ash.to_numpy().reshape(len(x), 1)

# Split for simple linear regression
simple_x_train, simple_x_test, y_train, y_test = train_test_split(simple_x,
                                                                  y,
                                                                  random_state=0,
                                                                  test_size=0.2)

model = LinearRegression()
_ = model.fit(simple_x_train, y_train)
simple_y_pred = model.predict(simple_x_test)

# Plot simple linear regression
fig, ax = plt.subplots(figsize=(15, 7))

# Plot real values scatter plot
_ = plt.scatter(simple_x_test, y_test, color="black", label="Real Values")

# Plot predicted values scatter plot
_ = plt.scatter(simple_x_test,
                simple_y_pred,
                color="red",
                label="Predicted Values")

# Plot regression line
_ = plt.plot(simple_x_test,
             simple_y_pred,
             color="blue",
             label="Regression Line")

# Show legend
_ = plt.legend()

# Set title
title = "Wine quality by alkalinity of ash"
plt.title(title)

# Set axis labels
ax.set_xlabel("Alkalinity of Ash (AOA)")
_ = ax.set_ylabel("Wine Quality")
fig.savefig("linear_regression.jpg")
plt.close(fig)
