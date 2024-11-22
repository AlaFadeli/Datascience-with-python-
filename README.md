# Linear Regression Example: Alcohol Quality by Alkalinity of Ash

This repository demonstrates a basic implementation of simple linear regression to predict alcohol quality based on a specific chemical feature, "alkalinity of ash," from the **Wine Dataset**. The code visualizes the results using scatter plots for both real and predicted values, along with the regression line.

## Code Explanation

The provided code performs the following steps:

### 1. **Loading and Inspecting the Dataset**
- The **Wine Dataset** is loaded using `sklearn.datasets.load_wine`, which contains data on various chemical properties of wine, along with the target variable, which is the quality of the wine (discrete class).
  
- The data is combined into a **Pandas DataFrame**, and an initial inspection of the dataset is performed using `.describe()` to provide summary statistics.

- The dataset is then analyzed with a **scatter matrix** to explore potential relationships between various features. A **correlation matrix** is computed and displayed using a heatmap to highlight the strength of the relationships between the features.

### 2. **Feature Selection and Data Preprocessing**
- A **single feature** ("alkalinity of ash") is selected for this simple linear regression. This feature is reshaped into a 2D array (`simple_x`) to fit the linear regression model.

- The data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`, where 80% of the data is used for training and 20% is reserved for testing.

### 3. **Training the Linear Regression Model**
- A **Linear Regression model** is created and trained using the `fit` method on the training data. Predictions for the test data are made using the `predict` method.

### 4. **Plotting the Results**
- The results are visualized using `matplotlib` by plotting the **real values** (black) and **predicted values** (red) against the test set.
  
- The **regression line** (blue) is also drawn to show the fitted relationship between the selected feature ("alkalinity of ash") and the target variable (wine quality).

### 5. **Code:**

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Loading the dataset
dataset = datasets.load_wine(as_frame=True)

# Setting data and target
x, y = dataset.data, dataset.target

# Creating dataframe
data = pd.concat([x, y], axis=1)

# Inspecting dataset
sample_indices = np.linspace(0, len(x) - 4, 4, dtype=int)
sample_indices = [index for i in sample_indices for index in range(i, i + 4)]
sample_indices = sample_indices[:len(x)]  # Ensure indices do not go out of bounds

sample_data = data.iloc[sample_indices, :]
styled_data = sample_data.style.set_properties(**{
    "text-align": "center",
}).set_properties(**{
    "border-left": "4px solid black"
}, subset=['target']).set_table_styles([dict(selector="th", props=[("font-size", "13px")]), 
                                          dict(selector="td", props=[("font-size", "11px")])]).background_gradient()

styled_data.to_html('styled_output.html')
pd.set_option('float_format', '{:g}'.format)
data.describe()

# Scatter matrix plot and correlation heatmap
axes = pd.plotting.scatter_matrix(data, figsize=(15, 15))

correlation_matrix = np.corrcoef(data.values.T)
fig, ax = plt.subplots(figsize=(8, 8))
tick_labels = list(x.columns) + ['diabetes']
hm = sns.heatmap(correlation_matrix, ax=ax, cbar=True, cmap="vlag", vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 12}, xticklabels=tick_labels, yticklabels=tick_labels)
plt.tight_layout()

# Train-test split for simple linear regression
simple_x = x.alcalinity_of_ash.to_numpy().reshape(len(x), 1)
simple_x_train, simple_x_test, y_train, y_test = train_test_split(simple_x, y, random_state=0, test_size=0.2)

# Fit model
model = LinearRegression()
model.fit(simple_x_train, y_train)
simple_y_pred = model.predict(simple_x_test)

# Plot results
fig, ax = plt.subplots(figsize=(15, 7))

# Plot real values scatter plot
_ = plt.scatter(simple_x_test, y_test, color="black", label="Real Values")

# Plot predicted values scatter plot
_ = plt.scatter(simple_x_test, simple_y_pred, color="red", label="Predicted Values")

# Plot regression line
_ = plt.plot(simple_x_test, simple_y_pred, color="blue", label="Regression Line")

# Show legend
_ = plt.legend()

# Set title
title = "Alcohol Quality by Alkalinity of Ash"
plt.title(title)

# Set axis labels
ax.set_xlabel("Alkalinity of Ash (AOA)")
_ = ax.set_ylabel("Alcohol Quality (AQ)")

plt.show()
```

## Issues with the Example: Low Correlation Coefficient

In this example, we demonstrate a basic linear regression model, which is highly dependent on the **correlation coefficient** (denoted as **r**) between the independent variable (alkalinity of ash) and the dependent variable (alcohol quality). For linear regression to be effective, a strong linear relationship between these two variables is essential for making accurate predictions.

However, in this case, the **correlation coefficient** is relatively low (around **0.5**), which signals a **weak linear relationship** between alkalinity of ash and alcohol quality. Ideally, a good linear regression model should exhibit a correlation coefficient closer to **1** (indicating a strong positive linear relationship) or **-1** (indicating a strong negative relationship).

### Why Is a Low Correlation Coefficient a Problem?

#### 1. **Weak Predictive Power**
   With a correlation coefficient of only **0.5**, the linear regression model struggles to capture the true relationship between the variables. This can lead to inaccurate predictions, as the model may fail to represent the underlying trend effectively.

#### 2. **Non-linear Relationships**
   A low correlation coefficient suggests that the relationship between the two variables might not be linear. In such cases, linear regression is not the most suitable model. More advanced techniques like **polynomial regression**, **decision trees**, or other **machine learning models** could potentially yield better results by capturing the non-linear relationship between the variables.

#### 3. **Risk of Overfitting or Underfitting**
   When the correlation between variables is weak, the linear regression model might either **overfit** or **underfit** the data:
   - **Overfitting** happens when the model captures noise or random fluctuations in the data, rather than the actual trend, leading to poor generalization.
   - **Underfitting** occurs when the model is too simple and fails to capture the significant patterns, leading to subpar performance.

### Conclusion

While the provided code offers a basic implementation of linear regression, the model's performance is limited due to the **low correlation coefficient** (around **0.5**). A stronger correlation would likely improve the accuracy of the predictions. To enhance the model, you might explore other features, utilize more complex models, or conduct further analysis to identify stronger relationships that lead to better prediction outcomes.

