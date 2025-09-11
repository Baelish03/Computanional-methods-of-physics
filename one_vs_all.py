import numpy as np
import pandas as pd

class OneVsAll:
    def one_vs_all(self, test_matrix):
        """
        Multi-class classification using one-vs-all strategy.

        Each perceptron corresponds to a single class.
        Predictions are collected from all classifiers and the
        final class is chosen based on scores.

        Parameters
        ----------
        test_matrix : np.ndarray
            Test feature matrix.

        Returns
        -------
        np.ndarray
            Final predicted class labels.
        """
        self.indexes, self.predictions = [], [None] * self.length
        for a in range(self.length):
            self.indexes.append(a)
            self.predictions[a] = self.test_prediction(test_matrix,
                                                       self.trained_weights_matrix[a])
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] is None and df["pred"][i][j] == 1:
                    final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] is None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float)
