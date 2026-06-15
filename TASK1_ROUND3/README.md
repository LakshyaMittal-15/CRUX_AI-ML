# Decision Tree Classifier from Scratch using Python

A complete implementation of a **Decision Tree Classifier from scratch** using **Python, NumPy, and Object-Oriented Programming (OOP)** without relying on Scikit-Learn's built-in Decision Tree model. The classifier is trained and evaluated on the Titanic dataset to predict passenger survival.

## 📖 Overview

Decision Trees are one of the most widely used supervised machine learning algorithms for classification and regression tasks. This project focuses on building a binary classification Decision Tree entirely from scratch to understand the underlying concepts such as:

- Entropy
- Information Gain
- Recursive Tree Construction
- Dataset Splitting
- Tree Traversal for Prediction

The model is trained on the Titanic dataset and predicts whether a passenger survived or not based on selected features.

## Objective

The objective of this project is to:

- Understand the internal working of Decision Trees.
- Implement entropy and information gain manually.
- Build a recursive tree structure using OOP.
- Perform classification without using Scikit-Learn's DecisionTreeClassifier.
- Evaluate model performance on unseen data.

##  Dataset

### Titanic Dataset

The Titanic dataset contains information about passengers aboard the RMS Titanic and whether they survived the disaster.

### Features Used

| Feature | Description |
|----------|-------------|
| pclass | Passenger Class |
| sex | Gender (Encoded) |
| age | Passenger Age |
| fare | Ticket Fare |
| embarked | Port of Embarkation (Encoded) |

### Target Variable

| Target | Meaning |
|----------|----------|
| 0 | Did Not Survive |
| 1 | Survived |

##  Project Architecture

### Node Class

The Node class represents each node in the decision tree.

A node can be:

#### Decision Node

Stores:

- Feature Index
- Threshold
- Information Gain
- Left Child Node
- Right Child Node

#### Leaf Node

Stores:

- Final Predicted Class

### DecisionTree Class

The DecisionTree class contains all logic required to build and use the classifier.

#### Methods Implemented

| Method | Purpose |
|----------|----------|
| build_tree() | Recursively constructs the tree |
| best_split() | Finds the optimal split |
| split() | Divides dataset into left and right subsets |
| gain() | Calculates gain|
| gini_gain() | Calculates gini gain |
| fit() | Trains the model |
| predict() | Predicts labels |
| predict_class() | Traverses tree for predictions |


##  Working Principle

### Step 1: Data Preparation

- Load Titanic dataset
- Select features and target
- Split into training and testing datasets


features = ["pclass", "sex", "age", "fare", "embarked"]
target = ["survived"]

X = titanic[features]
y = titanic[target]

### Step 2: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Step 3: Build Decision Tree

For every feature:

1. Find all unique thresholds.
2. Split dataset.
3. Compute entropy.
4. Calculate information gain.
5. Select split with maximum information gain.
6. Create child nodes recursively.

### Step 4: Create Leaf Nodes

A leaf node is created when:

- Maximum depth is reached.
- Information gain becomes zero.
- Number of samples becomes too small.

The most frequent class becomes the leaf value.

```python
leaf_value = Counter(y).most_common(1)[0][0]
```

### Step 5: Prediction

For every test sample:

1. Start at root node.
2. Compare feature value with threshold.
3. Move left or right.
4. Continue recursively.
5. Return class label at leaf node.


##  Project Structure

```text
DecisionTreeClassifier.ipynb
│
├── Data Loading
├── Data Preprocessing
├── Feature Selection
├── Train-Test Split
│
├── Node Class
│
├── DecisionTree Class
│   ├── build_tree()
│   ├── best_split()
│   ├── split()
│   ├── gain()
│   ├── gini_gain()
│   ├── fit()
│   ├── predict()
│   └── predict_class()
│
└── Model Evaluation
```

##  Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn (Train-Test Split Only)
- Collections Module
- Jupyter Notebook


##  Model Training

```python
decision_tree_model = DecisionTree(
    min_samples_split=5,
    max_depth=5
)

decision_tree_model.fit(X_train, y_train)
```

## Making Predictions

```python
predictions = decision_tree_model.predict(X_test)
```

## 📊 Model Evaluation

```python
accuracy = np.mean(predictions == y_test) * 100

print(f"Accuracy: {accuracy:.2f}%")
```

Example Output:

```text
Accuracy: 78.00%
```

##  Key Concepts Learned

Through this project, the following concepts were implemented and understood:

- Decision Tree Construction
- Recursive Programming
- Binary Trees
- Entropy Calculation
- Information Gain
- Dataset Partitioning
- Tree Traversal
- Classification Algorithms
- Machine Learning Fundamentals


##  Future Enhancements

Potential improvements include:

- Gini Index Implementation
- Tree Visualization
- Reduced Error Pruning
- Cost Complexity Pruning
- Random Forest Implementation
- Feature Importance Analysis
- Support for Multiclass Classification
- Cross Validation


## 👨‍💻 Author

**Lakshya Mittal**

This project was developed as part of learning and implementing machine learning algorithms from scratch to gain a deeper understanding of Decision Tree internals and object-oriented design in Python.

##  License

This project is intended for educational and learning purposes.

Feel free to use, modify, and extend the implementation for academic or personal projects.
