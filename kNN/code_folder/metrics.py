import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    # ACCURACY
    all_values = y_pred.shape[0]
    true_p_true_n = sum(abs(np.array(list(map(int, y_true))) - y_pred) == 0)
    accuracy = true_p_true_n / all_values
    
    # PRECISION
    true_p = sum(np.array(list(map(int, y_true[y_pred == 1]))))
    precision = true_p / sum(y_pred == 1)
    
    # RECALL
    true_p_false_n = sum(np.array(list(map(int, y_true))) == 1)
    recall = true_p / true_p_false_n
    
    # F1
    f1 = (precision*recall)*2/(precision + recall)

    return accuracy, precision, recall, f1

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    all_values = y_pred.shape[0]
    # Переведем на всякий случай оба вектора в числовые форматы
    num_y_pred = np.array(list(map(int, y_pred)))
    num_y_true = np.array(list(map(int, y_true)))
    m_accuracy = sum((num_y_pred - num_y_true) == 0) / all_values
    return m_accuracy

def r_squared(y_true, y_pred):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
#     return 1 - sum((y_true - y_pred)**2)/sum((y_true - np.mean(y_true))**2)
    return 1 - sum(np.square(y_true - y_pred))/sum(np.square(y_true - np.mean(y_true)))

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    all_values = y_pred.shape[0]
    return sum((y_true - y_pred)**2)/all_values
    
def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    all_values = y_pred.shape[0]
    return sum(abs(y_true - y_pred))/all_values
    