import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def make_predictions(self, model, X_data, threshold=0.5):
        Y_pred = model.predict(X_data)
        Y_pred = Y_pred > threshold
        return Y_pred

    def compute_confusion_matrix(self, Y_test, Y_pred):
        con_mat = tf.math.confusion_matrix(labels=Y_test, predictions=Y_pred).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        return con_mat_norm

    def subplot_confusion_matrix(self, confusion_matrix, subplot_title='',
                                 subplot_index=1, subplot_rows=1, subplot_cols=1):
        np_confusion_matrix = np.array([np.array(r).astype('float') for r in confusion_matrix])

        np_confusion_matrix = np.around(np_confusion_matrix.astype('float') /
                                        np_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        label_names = ['Bad', 'Good']
        df_confusion_matrix = pd.DataFrame(confusion_matrix,
                                        index=label_names, 
                                        columns=label_names)

        ax = plt.subplot(subplot_rows, subplot_cols, subplot_index)
        sns.heatmap(df_confusion_matrix, cmap=plt.cm.Blues, annot=True)
        if subplot_title != '':
            ax.set_title(subplot_title)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
