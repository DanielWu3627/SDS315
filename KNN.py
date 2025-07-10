# Imports
import argparse  # For command-line argument parsing
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import LeaveOneOut, StratifiedKFold  # Cross-validation strategies
from sklearn.neighbors import KNeighborsClassifier  # KNN model
from sklearn.preprocessing import LabelEncoder  # For encoding class labels
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score  # Evaluation metrics
import sys  # Access system-specific parameters and functions
import os  # Interact with the operating system (e.g., file paths)

# Set the directory containing CSV files
CSV_DIR = r"."  # Path to the folder with CSV files

# Function to list all CSV files in the directory
def list_csv_files():
    return [f for f in os.listdir(CSV_DIR) if f.lower().endswith('.csv')]  # Return all files ending in .csv (case-insensitive)

# CLI argument setup
parser = argparse.ArgumentParser(description="Run KNN classification with cross-validation.")  # Initialize argument parser
parser.add_argument("-f", required=True, help="Name of the CSV data file (must be in the default directory).")  # Argument for file name
parser.add_argument("-k", type=int, required=True, help="Number of nearest neighbors (k). Must be >= 1.")  # Argument for K value
parser.add_argument("-n", type=int, required=True, help="1 for LOOCV, or any integer > 1 for number of folds in K-Fold.")  # Argument for cross-validation choice
args = parser.parse_args()  # Parse the command-line arguments

script_name = os.path.basename(sys.argv[0])  # Extract the script name

# Validation
available_csvs = list_csv_files()  # Get list of all valid CSV files
if not args.f.lower().endswith('.csv') or args.f not in available_csvs or args.k < 1 or args.n < 1:
    # Validate arguments: correct file extension, file must exist, k and n must be >= 1
    print(
    "\nPlease type:\n"
    "- -f for CSV file name (e.g. -f BreastTissue.csv)\n"
    "- -k for number of nearest neighbors (must be ≥ 1, e.g. -k 5)\n"
    "- -n for loocv or kfold (type -n 1 for loocv, or any integer > 1 for kfold)\n"
    "\nExample:\n"
    "To apply 5 nearest neighbors on 'BreastTissue.csv' and use 3 fold cross validation to check performance, type:\n"
    "python " + script_name + " -f BreastTissue.csv -k 5 -n 3 ", file=sys.stderr
)

    print("\nAvailable CSV files:", file=sys.stderr)  # List available CSV files to guide the user
    for f in available_csvs:
        print(f"  - {f}", file=sys.stderr)
    print("\nTIP: You can also save the output to a file using '> output.txt' (e.g., python KNN.py -f BreastTissue.csv -k 5 -n 3 > output.txt)", file=sys.stderr)
    sys.exit(1)  # Exit program if validation fails

# Begin output
print(f"\n=== Results for {args.f} | k={args.k} | {'LOOCV' if args.n == 1 else f'{args.n}-Fold CV'} ===")
print(f"\nRunning KNN with k={args.k}")  # Notify user of chosen k value

# Load and prepare dataset
csv_path = os.path.join(CSV_DIR, args.f)  # Construct full file path
data = pd.read_csv(csv_path)  # Read CSV file into DataFrame
X = data.drop(columns=['NAME', 'CLASS'])  # Extract feature columns (exclude NAME and CLASS)
y = data['CLASS']  # Target labels

# Encode labels
label_encoder = LabelEncoder()  # Create label encoder object
y_encoded = label_encoder.fit_transform(y)  # Transform string labels into integers
class_names = sorted(label_encoder.classes_)  # Sorted list of original class labels
class_indices = label_encoder.transform(class_names)  # Corresponding encoded class indices

# Setup cross-validation
if args.n == 1:
    splitter = LeaveOneOut()  # Use Leave-One-Out CV if n=1
    print(f"Cross-Validation was performed using Leave-One-Out\n")
else:
    splitter = StratifiedKFold(n_splits=args.n, shuffle=True, random_state=42)  # Use Stratified K-Fold CV otherwise
    print(f"Cross-Validation was performed using {args.n} folds\n")

k = args.k  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)  # Initialize KNN model with specified k

val_true, val_pred = [], []  # Lists to store true and predicted validation labels
train_votes = [[] for _ in range(len(X))]  # Votes from KNN predictions on training data for each sample
train_labels = [None for _ in range(len(X))]  # True labels for training data

# Perform cross-validation
for train_idx, test_idx in splitter.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # Split features
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]  # Split targets

    model.fit(X_train, y_train)  # Train KNN model
    val_preds = model.predict(X_test)  # Predict on validation/test set

    val_pred.extend(val_preds)  # Accumulate predictions
    val_true.extend(y_test)  # Accumulate true labels

    train_preds = model.predict(X_train)  # Predict on training set (for internal eval)
    for idx, pred in zip(train_idx, train_preds):
        train_votes[idx].append(pred)  # Store predictions (votes)
        train_labels[idx] = y_encoded[idx]  # Store true label

# Majority voting for training predictions
train_pred_majority, train_true_majority = [], []  # Lists for final training predictions and true labels
for votes, true_label in zip(train_votes, train_labels):
    if votes:
        majority_vote = max(set(votes), key=votes.count)  # Most common prediction
        train_pred_majority.append(majority_vote)
        train_true_majority.append(true_label)

# Metrics
train_cm = confusion_matrix(train_true_majority, train_pred_majority, labels=class_indices)  # Confusion matrix (training)
val_cm = confusion_matrix(val_true, val_pred, labels=class_indices)  # Confusion matrix (validation)

def compute_specificity(cm):  # Function to calculate specificity from confusion matrix
    specificity = []
    for i in range(len(cm)):
        TP = cm[i][i]  # True Positive
        FP = sum(cm[:, i]) - TP  # False Positive
        FN = sum(cm[i, :]) - TP  # False Negative
        TN = cm.sum() - (TP + FP + FN)  # True Negative
        denom = TN + FP
        specificity.append(TN / denom if denom else 0.0)  # Avoid divide-by-zero
    return np.array(specificity)

# Accuracy
train_acc = accuracy_score(train_true_majority, train_pred_majority)  # Training accuracy
val_acc = accuracy_score(val_true, val_pred)  # Validation accuracy
print(f"Training\t{train_acc * 100:.2f}% ({int(train_acc * len(train_true_majority))}/{len(train_true_majority)})")  # Print training accuracy
print(f"Validation\t{val_acc * 100:.2f}% ({int(val_acc * len(val_true))}/{len(val_true)})")  # Print validation accuracy

# Recall, Precision, Specificity
train_recall = recall_score(train_true_majority, train_pred_majority, labels=class_indices, average=None, zero_division=0)  # Recall per class (training)
train_precision = precision_score(train_true_majority, train_pred_majority, labels=class_indices, average=None, zero_division=0)  # Precision per class (training)
train_specificity = compute_specificity(train_cm)  # Specificity per class (training)

val_recall = recall_score(val_true, val_pred, labels=class_indices, average=None, zero_division=0)  # Recall per class (validation)
val_precision = precision_score(val_true, val_pred, labels=class_indices, average=None, zero_division=0)  # Precision per class (validation)
val_specificity = compute_specificity(val_cm)  # Specificity per class (validation)

# Print formatted per-class metrics
for i, cls in enumerate(class_names):
    train_sens = train_recall[i] * 100  # Training sensitivity (recall)
    val_sens = val_recall[i] * 100  # Validation sensitivity
    train_spec = train_specificity[i] * 100  # Training specificity
    val_spec = val_specificity[i] * 100  # Validation specificity
    train_ner = (train_recall[i] + train_specificity[i]) / 2 * 100  # Training non-error rate (NER)
    val_ner = (val_recall[i] + val_specificity[i]) / 2 * 100  # Validation NER

    # Extract confusion matrix values
    tp_train = train_cm[i][i]
    fn_train = train_cm[i, :].sum() - tp_train
    fp_train = train_cm[:, i].sum() - tp_train
    tn_train = train_cm.sum() - (tp_train + fn_train + fp_train)

    tp_val = val_cm[i][i]
    fn_val = val_cm[i, :].sum() - tp_val
    fp_val = val_cm[:, i].sum() - tp_val
    tn_val = val_cm.sum() - (tp_val + fn_val + fp_val)

    print("")
    print(f"{cls}")  # Class name
    print(f"{'':<10}{'Sensitivity':>20}{'Selectivity':>25}{'Non-Error Rate':>25}")  # Headers
    print(f"{'Training':<10}{train_sens:>14.2f}% ({tp_train}/{tp_train+fn_train}){train_spec:>17.2f}% ({tn_train}/{tn_train+fp_train}){train_ner:>21.2f}%")
    print(f"{'Validation':<10}{val_sens:>14.2f}% ({tp_val}/{tp_val+fn_val}){val_spec:>17.2f}% ({tn_val}/{tn_val+fp_val}){val_ner:>21.2f}%")

# Print Confusion Matrices
def format_confusion_matrix(cm, labels, true_labels, pred_labels, dataset_name):
    print(f"\n{dataset_name} Confusion Matrix")
    print(f"{'':<16}{'Predicted Class'}")  # Header row
    header = f"{'True Class':<16}" + ''.join(f"{lbl:<16}" for lbl in labels) + "Recall"  # Column headers
    print(header)
    recalls = recall_score(true_labels, pred_labels, labels=class_indices, average=None, zero_division=0)  # Recall values for all classes
    for i, label in enumerate(labels):
        row = f"{label:<16}"  # Start new row with class name
        for j in range(len(labels)):
            cell = cm[i][j]  # Cell value
            row += f"{cell}/{cm.sum()} ({cell/cm.sum():.0%})".ljust(16)  # Cell as count and percentage
        row += f"{cm[i][i]}/{cm[i].sum()} ({recalls[i]*100:.0f}%)".ljust(16)  # Recall at end
        print(row)
    row = f"{'Precision':<16}"
    precisions = precision_score(true_labels, pred_labels, labels=class_indices, average=None, zero_division=0)  # Precision values
    for i in range(len(labels)):
        col_sum = cm[:, i].sum()
        correct = cm[i][i]
        row += f"{correct}/{col_sum if col_sum > 0 else 1} ({precisions[i]*100:.0f}%)".ljust(16)
    print(row)

format_confusion_matrix(train_cm, class_names, train_true_majority, train_pred_majority, "Training")  # Training confusion matrix
format_confusion_matrix(val_cm, class_names, val_true, val_pred, "Validation")  # Validation confusion matrix


