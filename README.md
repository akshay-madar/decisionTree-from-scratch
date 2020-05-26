Here I try to implement Decision Tree algo. from scratch for a classification problem. The model achieves slightly less accuracy as compared to the logistic regression that I have implemented on the same dataset, which can also be found in its own repo.

### Decision Tree algo:
The splits are performed based on information gain and gini impurity as below. The tree builds recursively until there is no info. gain. Please see the notebook file for complete code.

```
# Calculate the Gini Impurity for a list of rows
def gini(rows):

    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
```

```
# he uncertainty of the starting node, minus the weighted impurity of two child nodes
def info_gain(left, right, current_uncertainty):

    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
```
