from sklearn.metrics import r2_score, precision_recall_curve, recall_score, precision_score, auc, classification_report, fbeta_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss

def cramers_v(var1,var2):

    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None))
    chi2 = ss.chi2_contingency(crosstab)[0]
    n = crosstab.sum()
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

# Métricas: F2, R2, Accuracy, Precision, Recall, F1, Support, Matriz de Confusion
def plot_metrics(y_test, y_pred, yhat=None, Y=None):

    try:
        if yhat.all() != None:

            print(f'\nF2 Score: {fbeta_score(y_test, y_pred, beta=2, average="macro")}\n') # F2-Score

            print(f'R2 Score: {r2_score(y_test, y_pred)}\n') # R2

            print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}\n') # Accuracy

            print(classification_report(y_test, y_pred, labels= [0, 1])) # Tabla con precision, recall, f1-score, support

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', values_format='.2g') # Confusion Matrix
            disp.figure_.suptitle("Percentage Confusion Matrix")

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred) # Confusion Matrix
            disp.figure_.suptitle("Confusion Matrix")
            # print(f"Confusion matrix:\n{disp.confusion_matrix}") # devuelve la matrix en texto

            # retrieve just the probabilities for the positive class
            pos_probs = yhat[:, 1]
            # calculate the no skill line as the proportion of the positive class
            no_skill = len(Y[Y==1]) / len(Y)
            # plot the no skill precision-recall curve
            fig = plt.figure()
            ax = plt.axes()
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            # calculate model precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, pos_probs)
            # convert to f score
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            # plot the model precision-recall curve
            ax.plot(recall, precision, marker='.', label='Model')
            # Best Threshold
            ax.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision') 
            plt.title('Precision-Recall Curve')
            # show the legend
            plt.legend()
            # show the plot
            plt.show()

            print(f'Model PR AUC: {auc(recall, precision)}')
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            
    except AttributeError:
            print(f'\nF2 Score: {fbeta_score(y_test, y_pred, beta=2, average="macro")}\n') # F2-Score

            print(f'R2 Score: {r2_score(y_test, y_pred)}\n') # R2

            print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}\n') # Accuracy

            print(classification_report(y_test, y_pred, labels= [0, 1])) # Tabla con precision, recall, f1-score, support

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', values_format='.2g') # Confusion Matrix
            disp.figure_.suptitle("Percentage Confusion Matrix")

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred) # Confusion Matrix
            disp.figure_.suptitle("Confusion Matrix")
            # print(f"Confusion matrix:\n{disp.confusion_matrix}") # devuelve la matrix en texto
def plot_recall_precission(recall_precision):

    plt.figure(figsize=(15, 5))
    ax = sns.pointplot(x = [element[0] for element in recall_precision], y=[element[1] for element in recall_precision],
                     color="r", label='recall', scale=1)
    ax = sns.pointplot(x = [element[0] for element in recall_precision], y=[element[2] for element in recall_precision],
                     color="b", label='precission')
    ax.set_title('recall-precision versus threshold')
    ax.set_xlabel('threshold')
    ax.set_ylabel('probability')
    ax.legend()

    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if(i%5 == 0) or (i%5 ==1) or (i%5 == 2) or (i%5 == 3):
            labels[i] = '' # skip even labels
            ax.set_xticklabels(labels, rotation=45, fontdict={'size': 10})
    plt.show()

def plot_threshold(model, x_train_scaled, x_test_scaled, y_train, y_test):
        # predicción de TRAIN
    pd_train_predicted = pd.DataFrame(model.predict_proba(x_train_scaled), 
                                    index=x_train_scaled.index, columns = ['y_predicted_0', 'y_predicted']).drop(['y_predicted_0'],axis=1)
    pd_train_predicted_final = pd.concat([x_train_scaled, pd_train_predicted, y_train],axis=1)

    prob_predictions = pd_train_predicted_final.y_predicted.values
    recall_precision = []

    for threshold in np.arange(0.01, 0.99, 0.01):
        
        given_threshold = [1 if value>threshold else 0 for value in prob_predictions]
        recall_precision.append([threshold, recall_score(pd_train_predicted_final.isFraud, given_threshold),
                                precision_score(pd_train_predicted_final.isFraud, given_threshold)])

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plot_recall_precission(recall_precision)