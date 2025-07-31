# Import required libraries
import numpy as np # Numerical operations
import pandas as pd  # For data manipulation and analysis 
from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold  
from sklearn.neighbors import KNeighborsClassifier  # KNN model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc  
from sklearn.preprocessing import label_binarize

from ipywidgets import IntSlider, Dropdown, Button, HBox, VBox, Label, Layout   # For UI Control
from IPython.display import display, clear_output, HTML
import io  # Provides tools for handling input/output
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict
import time
  

################################ Shared Helper Functions #######################################################
# Customized function for calculating weights
def compute_weights(weight, dist = None, for_sklearn = False):
    if for_sklearn:
        if weight == 'one':
            return 'uniform'
        elif weight == 'inverse':
            return 'distance'
        else:
            return lambda d: 1 / (d**2 + 1e-5)

    else:
        if weight == "one":
            return np.ones(len(dist))
        elif weight == "inverse":
            return 1 / (dist + 1e-5)
        else:
            return 1 / (dist**2 + 1e-5)

# Calculate scores for each sample
def accumulate_scores(score_matrix, idx, neighbor_labels, weights_arr, class_indices):
    class_weight_sum = defaultdict(float)     
    # For each test sample, get the weighted score for each class
    for lbl, w in zip(neighbor_labels, weights_arr):
        class_weight_sum[lbl] += w
    for cls_idx in class_indices:
        score_matrix[idx, cls_idx] += class_weight_sum.get(cls_idx, 0.0)


# Calculate specificity
def compute_specificity(cm):
    specificity = []
    for i in range(len(cm)):  # For each class
        TP = cm[i][i]  # True positives for class i
        FP = sum(cm[:, i]) - TP  # False positives for class i
        FN = sum(cm[i, :]) - TP  # False negatives for class i
        TN = cm.sum() - (TP + FP + FN)  # True negatives for class i
        denom = TN + FP
        specificity.append(TN / denom if denom else 0.0)  # Avoid division by zero
    return np.array(specificity)


# Calculate accuracy, sensitivity, specificity, and non-error rate for input datasets
def print_eval_report(cm, y_true, y_pred, class_names, context):
    class_indices = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)

    print(f"{context}Accuracy\t{acc * 100:.2f}% ({int(acc * len(y_true))}/{len(y_true)})\n")

    recall_vals= recall_score(y_true, y_pred, labels=class_indices, average=None, zero_division=0)
    specificity_vals = compute_specificity(cm)

    for i, cls in enumerate(class_names):  # Loop through each class
        sens = recall_vals[i] * 100  # Sensitivity = Recall
        
        spec = specificity_vals[i] * 100
        ner = (recall_vals[i] + specificity_vals[i]) / 2 * 100  # Non-error rate
        

        # Extract confusion matrix values
        tp = cm[i][i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

    
        # Print table with metrics
        
        print(f"{cls:<10}{'Sensitivity':>20}{'Selectivity':>25}{'Non-Error Rate':>25}")
        print(f"{' ':<10}{sens:>14.2f}% ({tp}/{tp+fn}){'':>5}{spec:>11.2f}% ({tn}/{tn+fp}){ner:>21.2f}%")

    print("\n\n")


# This function plot confusion matrix heatmaps.
def plot_confusion_matrix_heatmap_with_metrics(title):
    # global train_cm, train_pred, train_true, val_cm, val_pred, val_true
    if "validation" in title.lower():
        cm = val_cm
        y_pred= val_pred
        y_true = val_true
    else:
        cm = train_cm
        y_pred = train_pred
        y_true = train_true
        
    total = cm.sum()
    n = len(class_names)

    # Create annotated matrix (n+1)x(n+1)
    annotated_cm = np.empty((n + 1, n + 1), dtype=object)

    # Fill confusion matrix cells
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            percent = round(count / total * 100)
            annotated_cm[i, j] = f"{count}/{total}\n({percent}%)"

    # Add recall (rightmost column)
    recalls = recall_score(y_true, y_pred, labels=range(n), average=None, zero_division=0)
    for i in range(n):
        correct = cm[i, i]
        total_true = cm[i].sum()
        recall_pct = round(recalls[i] * 100)
        annotated_cm[i, -1] = f"{correct}/{total_true if total_true else 1}\n({recall_pct}%)"

    # Add precision (bottom row)
    precisions = precision_score(y_true, y_pred, labels=range(n), average=None, zero_division=0)
    for j in range(n):
        correct = cm[j, j]
        total_pred = cm[:, j].sum()
        precision_pct = round(precisions[j] * 100)
        annotated_cm[-1, j] = f"{correct}/{total_pred if total_pred else 1}\n({precision_pct}%)"

    annotated_cm[-1, -1] = ""  # bottom-right corner

    # Extend the original confusion matrix with zeros to match shape
    extended_cm = np.zeros((n + 1, n + 1))
    extended_cm[:n, :n] = cm

    # Create extended label set
    xticklabels = list(class_names) + ["Recall"]
    yticklabels = list(class_names) + ["Precision"]

    # Plot
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(extended_cm, annot=annotated_cm, fmt="", cmap="Greens",
                xticklabels=xticklabels, yticklabels=yticklabels, cbar=False,
                linewidths=0, linecolor='gray')
    ax.xaxis.set_ticks_position('top')       # Move ticks to top

    # Determine number of rows and columns
    nrows, ncols = extended_cm.shape
    
    # Highlight last row and last column
    for i in range(nrows):
        for j in range(ncols):
            if i == nrows - 1 or j == ncols - 1:
                # Calculate patch position: seaborn heatmap uses (col, row) as (x, y)
                rect = plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True,
                    facecolor='white',   # Light yellow
                    edgecolor='black',
                    linewidth=0
                )
                ax.add_patch(rect)
    
    # Redraw annotations on top
    for t in ax.texts:
        t.set_zorder(10)
    
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()

# This function plots scores for each item.
# Plot score plots for training and testing samples
def plot_items_scores(title):
    if "validation" in title.lower():
        sample_scores = val_scores
    elif "training" in title.lower():
        sample_scores = train_score_avgs
    else:
        sample_scores = all_scores
        
    n_items = sample_scores.shape[0]
    n_classes = len(class_names)  # Can be 2, 3, or more

    # Generate distinct colors using a colormap
    cmap = plt.colormaps['tab10']  # Or try 'tab20', 'Set3', etc.
    class_colors = {i: cmap(i % cmap.N) for i in range(n_classes)}
    
    # Initialize bar positions for stacking
    pos_bottoms = np.zeros(n_items)
    neg_bottoms = np.zeros(n_items)

    display(HTML('<div style="text-align:center;"><strong>Score Report</strong></div>'))

    plt.figure(figsize=(14, 6))

    for cls in range(n_classes):
        values = []
        bottoms = []
        colors = []

        for i in range(n_items):
            score = sample_scores[i, cls]
            is_true_class = (cls == y_encoded[i])

            if is_true_class:
                values.append(score)
                bottoms.append(pos_bottoms[i])
                pos_bottoms[i] += score  # Accumulate for next class
            else:
                values.append(-score)
                bottoms.append(neg_bottoms[i])
                neg_bottoms[i] -= score  # Accumulate downward
            colors.append(class_colors[cls])

        # Draw bars for this class across all items
        plt.bar(range(n_items), values, bottom=bottoms, color=colors, width=0.6)

    #print(sample_scores)
    plt.xlabel('Item ID')
    plt.ylabel('Score')
    plt.title(f'Scores per Item ({title})')
    
    # Custom legend
    legend_elements = [Patch(facecolor=class_colors[i], label=f'{class_names[i]}(Class {i})') for i in range(n_classes)]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=n_classes)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

   
    
####################################### Functions Unique to the Grid Search ################################################
# This function performs grid search over a range of K values (1 to 20) for a KNN classifier using Leave-One-Out Cross-Validation (LOOCV).
# It searches for the optimal number of neighbors (k) that optimizes for the overall accuracy.

def run_grid_accuracy(X, y, metric, weights, file):
    global class_names, y_encoded, class_indices, file_name
    file_name = file
    class_names = list(dict.fromkeys(y))
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_index[label] for label in y])
    class_indices = list(range(len(class_names)))
    
    # === Grid Search for best K ===
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid = GridSearchCV(
        KNeighborsClassifier(metric=metric, weights=compute_weights(weights, for_sklearn=True)),
        param_grid,
        cv=LeaveOneOut()
    )
    # Time the grid search
    start = time.time()
    grid.fit(X, y)
    end = time.time()
    
    # Sort by mean_test_score descending
    grid_scores = grid.cv_results_
    n_top = 3
    results = []
    for i in range(len(grid_scores['params'])):
        mean = grid_scores['mean_test_score'][i]
        std = grid_scores['std_test_score'][i]
        params = grid_scores['params'][i]
        results.append((mean, std, params))

    results = sorted(results, key=lambda x: -x[0])
    
    print(f"GridSearchCV took {end - start:.2f} seconds for {len(results)} candidate parameter settings.")
    
    for rank, (mean, std, params) in enumerate(results[:n_top], start=1):
        print(f"\nModel with rank {rank}:")
        print(f"Mean validation score: {mean:.3f} (std: {std:.3f})")
        print(f"Parameters: {params}")

    # Display results for best k
    best_k = grid.best_params_['n_neighbors']
    best_score = grid.best_score_
    print(f"\nBest K: {best_k} (Mean LOOCV Accuracy: {best_score * 100:.2f}%)\n")
    evaluate_final(X, y, best_k, metric, weights, class_names)

# This function performs grid search over a range of K values (1 to 20) for a KNN classifier using Leave-One-Out Cross-Validation (LOOCV).
# It searches for the optimal number of neighbors (k) that optimizes for AUC score and report the ROC curvs for the top 3 values.

def run_grid_roc(X, y, metric, weights, file):
    global class_names, y_encoded, class_indices, file_name
    file_name = file
    class_names = list(dict.fromkeys(y))
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_index[label] for label in y])
    class_indices = list(range(len(class_names)))
    n_classes = len(class_indices)

    k_values = list(range(1, 21))
    auc_scores = []
    prob_dict = {}  # store probs per k for top-3 ROC plot
    truth_dict = {}

    start = time.time()

    for k in k_values:
        probs = []
        truths = []

        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]

            model = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric,
                weights=compute_weights(weights, for_sklearn=True)
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[0]
            probs.append(proba)
            truths.append(y_test[0])

        if n_classes == 2:
            auc = roc_auc_score(truths, [p[1] for p in probs])
        else:
            auc = roc_auc_score(truths, probs, multi_class='ovr', average='macro')

        auc_scores.append(auc)
        prob_dict[k] = np.array(probs)
        truth_dict[k] = np.array(truths)

    end = time.time()

    # Reporting top results
    results = [(auc_scores[i], 0.0, {'n_neighbors': k_values[i]}) for i in range(len(k_values))]
    results = sorted(results, key=lambda x: -x[0])
    print(f"Custom ROC Grid Search took {end - start:.2f} seconds for {len(results)} candidate parameter settings.")

    n_top = 3
    for rank, (mean, std, params) in enumerate(results[:n_top], start=1):
        print(f"\nModel with rank {rank}:")
        print(f"Mean validation AUC: {mean:.4f}")
        print(f"Parameters: {params}")

    # === Plot macro-average ROC curves for top 3 K ===
    print("\nPlotting ROC curves for top 3 K values...")
    plt.figure(figsize=(5, 4))
    for _, _, param in results[:n_top]:
        k = param['n_neighbors']
        probs = prob_dict[k]
        truths = truth_dict[k]

        if n_classes == 2:
            # Binary classification: plot a single ROC curve
            y_true_bin = np.array(truths)
            probs_pos = np.array([p[1] for p in probs])  # Probability for positive class
            fpr, tpr, _ = roc_curve(y_true_bin, probs_pos)
            auc_val = auc_scores[k - 1]
            plt.plot(fpr, tpr, label=f"K={k} (AUC={auc_val:.3f})")

        else:
            # Multi-class: plot macro-averaged ROC
            y_true_bin = label_binarize(truths, classes=class_indices)
            fpr = dict()
            tpr = dict()
            for i in class_indices:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            
            all_fpr = np.unique(np.concatenate([fpr[i] for i in class_indices]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in class_indices:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            auc_val = auc_scores[k - 1]
            plt.plot(all_fpr, mean_tpr, label=f"K={k} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Top 3 K values")
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Final model on full dataset
    best_k = results[0][2]['n_neighbors']
    best_auc = results[0][0]
    print(f"\nBest K: {best_k} (Mean LOOCV AUC: {best_auc:.4f})\n")
    evaluate_final(X, y, best_k, metric, weights, class_names)

# The function below evaluates the final KNN model on the entire dataset, using the best K from grid search.
# It first reports overall accuracy, then sensitivity, selectivity (specificity), and non-error rate for each class.
def evaluate_final(X, y, k, metric, weights, class_names):
    global train_true, train_pred, train_cm, all_scores

    # Final Model Evaluation on Full Dataset
    class_indices = list(range(len(class_names)))
    model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=compute_weights(weights, for_sklearn=True))
    model.fit(X, y_encoded)
    y_pred = model.predict(X)

    # Store results
    train_true = y_encoded
    train_pred = y_pred
    train_cm = confusion_matrix(train_true, train_pred, labels=class_indices)
    
    # Calculate scores 
    n_classes = len(set(y_encoded))
    all_scores = np.zeros((len(X), n_classes))

    # Get neighbors of each sample (including self)
    neigh_dists, neigh_indices = model.kneighbors(X, n_neighbors=k)

    # Compute scores for each sample
    for i in range(len(X)):
        neighbors = neigh_indices[i]
        dists = neigh_dists[i]
        weights_arr = compute_weights(weights, dists)
        neighbor_labels = y_encoded[neighbors]
        accumulate_scores(all_scores, i, neighbor_labels, weights_arr, class_indices)
 
    
    display(HTML('<div style="text-align:left;"><strong>Summary Report for the Full Dataset</strong></div>'))

    print(f"\nRunning Final KNN with k={k}")
    print("Model trained and evaluated on the full dataset")
    print(f"Input file name: {file_name}\n")
    
    print_eval_report(train_cm, train_true, train_pred, class_names, context = "")



############################################# Functions Unique to the Notebook Requiring User Inputs ################

# Obtain classification results for cross validation
def run_cross_validation(X, y_encoded, k, metric, weights, class_indices, n):
    splitter = LeaveOneOut() if n == 1 else StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=compute_weights(weights, for_sklearn=True))
    n_classes = len(class_indices)

    val_true, val_pred = [], []
    sample_scores = np.zeros((len(X), n_classes))
    train_score_sums = np.zeros((len(X), n_classes))
    train_score_counts = np.zeros(len(X))
    train_labels = [None] * len(X)

    for train_idx, test_idx in splitter.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        model.fit(X_train, y_train)
        val_preds = model.predict(X_test)
        val_pred.extend(val_preds)
        val_true.extend(y_test)

        neigh_dists_test, neigh_indices_test = model.kneighbors(X_test, n_neighbors=k)
        for i, (test_i, neighbors, dists) in enumerate(zip(test_idx, neigh_indices_test, neigh_dists_test)):
            weights_arr = compute_weights(weights, dists)
            accumulate_scores(sample_scores, test_i, y_train[neighbors], weights_arr, class_indices)

        train_preds = model.predict(X_train)
        neigh_dists_train, neigh_indices_train = model.kneighbors(X_train, n_neighbors=k)
        for i, (train_i, neighbors, dists) in enumerate(zip(train_idx, neigh_indices_train, neigh_dists_train)):
            weights_arr = compute_weights(weights, dists)
            accumulate_scores(train_score_sums, train_i, y_train[neighbors], weights_arr, class_indices)
            train_score_counts[train_i] += 1
            train_labels[train_i] = y_encoded[train_i]
    return val_true, val_pred, sample_scores, train_score_sums, train_score_counts, train_labels


# Calculate overall accuracy, sensitivity and selectivity for each class
def run_knn(X, y, k, metric, weights, n, input_file):
    global train_cm, train_true, train_pred, val_cm, val_true, val_pred, train_score_avgs, val_scores, class_names, file_name, y_encoded

    class_names = list(dict.fromkeys(y))       # This preserves the order of class names in the dataset
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_index[label] for label in y])            # Encode class labels as indices
    class_indices = list(range(len(class_names)))        # A range of indices for class names (eg. 0,1,2)
    file_name = input_file
    
    # Get scores and predicted class labels for training and testing samples via cross validation
    val_true, val_pred, val_scores, train_score_sums, train_score_counts, train_labels = run_cross_validation(
        X, y_encoded, k, metric, weights, class_indices, n
    )
    cv_method = "Leave One Out" if n==1 else f"{n}-folds"
    train_score_avgs = train_score_sums / train_score_counts[:, np.newaxis]
    
    train_pred = np.argmax(train_score_avgs, axis = 1)
    train_true = [lbl for lbl in train_labels]           
        
    train_cm = confusion_matrix(train_true, train_pred, labels=class_indices)
    val_cm = confusion_matrix(val_true, val_pred, labels=class_indices)      
   
    
    display(HTML('<div style="text-align:left;"><strong>Summary Report</strong></div>'))
    
    print(f"\nRunning KNN with k={k}")
    print(f"Cross-Validation was performed using {cv_method}")        
    print(f"Input file name: {file_name}\n")

    print_eval_report(train_cm, train_true, train_pred, class_names, context = "Training ")
    print_eval_report(val_cm, val_true, val_pred, class_names, context = "Validation ")




