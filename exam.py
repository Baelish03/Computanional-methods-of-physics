from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statistics as st
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers

class Perceptron:
    def __init__(self, features_matrix, labels_vec):
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0] #105
        self.features_len = features_matrix.shape[1] #2
        self.step = START
        self.weights_vec = np.zeros(shape=self.features_len + 1)
        self.labels_vec = labels_vec
        self.dummylabel_vec = np.zeros(shape=self.samples_len) #calculated label, shape = 105

        self.trained_weights_matrix = TRAINED_wEIGHTS_MATRIX
        self.trained_scores = SCORES

    def activation_func(self, features_vec):  #SECOND STEP: CALCULATION OF LABEL
        return np.where(np.dot(self.weights_vec[1:], features_vec) + self.weights_vec[0] >= 0, 1, 0) #sum from 1 to features_len of feature * weight + first wight
        
    def weights_vec_update(self, features_vec, dummylabel, label): #THIRD STEP: RECALCULATION OF WEIGHT IN THE INNER CYCLE
        self.weights_vec[0] += LR * (label - dummylabel)
        self.weights_vec[1:] += LR * (label - dummylabel) * features_vec
        return self.weights_vec

    def weights_training(self): #FIRST STEP: OUTER CYCLE
        while self.step <= EPOCHS:
            for i in range(self.samples_len): #iteration on samples INNER CYCLE
                self.dummylabel_vec[i] = self.activation_func(self.features_matrix[i]) #calculate a label for each sample
                self.weights_vec = self.weights_vec_update(self.features_matrix[i], self.dummylabel_vec[i], self.labels_vec[i])
            self.step += 1
        return self.weights_vec

    def train_prediction(self, trained_weights_vec):
        return (np.where(np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0] >= 0, 1, 0))
    
    def test_prediction(self, test_matrix, trained_weights_vec):
        return (np.where(np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0] >= 0, 1, 0))
            
    def train_score(self, trained_weights_vec):
        return np.mean(self.labels_vec == self.train_prediction(trained_weights_vec))
     
    def one_vs_all(self, test_matrix):
        self.indexes, self.predictions = [], [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            self.indexes.append(a); self.predictions[a] = self.test_prediction(test_matrix, self.trained_weights_matrix[a]); 
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] == None and df["pred"][i][j] == 1:
                        final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] == None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float) 
        #https://stackoverflow.com/questions/40809503/python-numpy-typeerror-ufunc-isfinite-not-supported-for-the-input-types if it works, it works

class Adaline:
    def __init__(self, features_matrix, labels_vec):
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0] 
        self.features_len = features_matrix.shape[1] 
        self.step = START
        self.weights_vec = np.zeros(shape=self.features_len + 1)
        self.dummylabel_vec = np.zeros(shape=self.samples_len)
        self.labels_vec = labels_vec
        self.error = 0
        self.gradient_vec = np.zeros(shape=self.features_len) #the gradient vector has same shape as weights vector

        self.trained_weights_matrix = TRAINED_wEIGHTS_MATRIX
        self.trained_scores = SCORES

    def activation_func(self, features_vec): #SECOND STEP: CALCULATION OF LABEL
        return np.where(np.dot(self.weights_vec[1:], features_vec) + self.weights_vec[0] >= 0, 1, 0)
        
    def error_func (self, dummylabel, label): #THIRD STEP: CALCULATION OF ERROR
        return ((label - dummylabel) ** 2) #a single error: sum in weights_training
       
    def gradient(self, features_vec, dummylabel, label): #FOURTH STEP: GRADIENT CALCULATION
        return (label - dummylabel) * (-1) * features_vec #a single gradient: sum in weights_training
        
    def weights_vec_update(self): #LAST INNER CYCLE STEP: WEIGHTS UPDATES
        self.weights_vec[1:] -= LR * self.gradient_vec
        self.weights_vec[0] += LR * sum(LABELS_VEC - self.dummylabel_vec) 
        return self.weights_vec
        
    def weights_training(self): #FIRST STEP: OUTER CYCLE
        while self.step <= EPOCHS:
            self.error = 0
            self.gradient_vec = np.zeros(shape=self.features_len) 
            for i in range(self.samples_len): #iteration on samples INNER CYCLE
                self.dummylabel_vec[i] = self.activation_func(self.features_matrix[i]) #calculate a label for each sample
                self.error += self.error_func (self.dummylabel_vec[i], self.labels_vec[i]) #SUM OF ALL ERRORS
                self.gradient_vec += self.gradient(self.features_matrix[i], self.dummylabel_vec[i], self.labels_vec[i]) #SUM OF ALL GRADIENT VECTORS
            self.weights_vec = self.weights_vec_update()
            self.step += 1
        #print("Number of wrong label:", self.error)
        return self.weights_vec
    
    def train_prediction(self, trained_weights_vec):
        return (np.where(np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0] >= 0, 1, 0))
    
    def test_prediction(self, test_matrix, trained_weights_vec):
        return (np.where(np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0] >= 0, 1, 0))
            
    def train_score(self, trained_weights_vec):
        return np.mean(self.labels_vec == self.train_prediction(trained_weights_vec))
    
    def one_vs_all(self, test_matrix):
        self.indexes, self.predictions = [], [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            self.indexes.append(a); self.predictions[a] = self.test_prediction(test_matrix, self.trained_weights_matrix[a]); 
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] == None and df["pred"][i][j] == 1:
                        final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] == None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float)

class Logistic_Regression:
    def __init__(self, features_matrix, labels_vec):
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0] 
        self.features_len = features_matrix.shape[1]
        self.step = START
        self.weights_vec = np.zeros(self.features_len + 1)
        self.dummylabel_vec = np.zeros(shape=self.samples_len)
        self.labels_vec = labels_vec
        self.gradient_vec = np.zeros(shape=self.features_len)
        self.p_vec = np.zeros(shape=self.samples_len)

        self.trained_scores = SCORES
        self.trained_weights_matrix = TRAINED_wEIGHTS_MATRIX

    def sigmoid(self, test_vec, weights_vec): #SECOND STEP: Z CALCULATION AND SIGMOID
        z = np.dot(test_vec, weights_vec[1:]) + weights_vec[0]
        return 1 / (1 + np.e ** (-z))
    
    def activation_func(self, p): #THIRD STEP
        return np.where(p > 0.5, 1, 0)

    def gradient(self, features_vec, dummylabel, label): #FOURTH STEP: GRADIENT CALCULATION
        return (label - dummylabel) * (-1) * features_vec #a single gradient: sum in weights_training
        
    def weights_vec_update(self): #LAST INNER CYCLE STEP: WEIGHTS UPDATES
        self.weights_vec[1:] -= LR * self.gradient_vec
        self.weights_vec[0] += LR * sum(self.labels_vec - self.dummylabel_vec) 
        return self.weights_vec
        
    def weights_training(self): #FIRST STEP: OUTER CYCLE
        while self.step <= EPOCHS:
            self.gradient_vec = np.zeros(shape=self.features_len) 
            for i in range(self.samples_len): #iteration on samples INNER CYCLE
                self.p_vec[i] = self.sigmoid(self.features_matrix[i], self.weights_vec) #calculate probability for each sample 
                #IMPORTANT: in the training phase, sigmoid uses the TRAINING weights, for the prediction it uses ALREADY TRAINED weights
                self.dummylabel_vec[i] = self.activation_func(self.p_vec[i]) #calculate a label for each sample
                self.gradient_vec += self.gradient(self.features_matrix[i], self.dummylabel_vec[i], self.labels_vec[i]) #SUM OF ALL GRADIENT VECTORS
            self.weights_vec = self.weights_vec_update()
            self.step += 1
        #print(list(self.p_vec))
        return self.weights_vec
    
    def train_prediction(self, trained_weights_vec):
        z = np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0]
        return (np.where(1 / (1 + np.e ** (-z)) > 0.5, 1, 0)) 
    
    def test_prediction(self, test_matrix, trained_weights_vec):
        z = np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0]
        return (np.where(1 / (1 + np.e ** (-z)) > 0.5, 1, 0)) 
                
    def train_score(self, trained_weights_vec):
        return np.mean(self.labels_vec == self.train_prediction(trained_weights_vec))
    
    def one_vs_all(self, test_matrix):
        self.indexes, self.predictions = [], [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            self.indexes.append(a); self.predictions[a] = self.test_prediction(test_matrix, self.trained_weights_matrix[a]); 
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        #print(df)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] == None and df["pred"][i][j] == 1:
                        final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] == None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float)

class Support_Vector_Machine:
    def __init__(self, features_matrix, kernel_type, labels_vec):
        self.new_label = np.where(np.array(labels_vec) == 1, 1, -1)  
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0] #100
        self.features_len = features_matrix.shape[1] #4
        self.step = START
        self.gradient_vec = np.zeros(shape=self.features_len)
        self.alpha = np.zeros(self.samples_len)
        self.weights = np.zeros(self.features_len + 1)
        self.gamma = 0.1 #1 / self.features_len
        self.b = 0
        #self.C = 1 #tolerance for the old method: over 1 is useless, under 1 (> 0 obv) makes SVM more aggressive (faster, but weaker); example: with 0.1 and 1/10 of the epoch it runs as well as the adaline
        self.kernel_type = kernel_type

        self.trained_scores = SCORES
        self.trained_weights_matrix = TRAINED_wEIGHTS_MATRIX

    def kernel(self, input1, input2): #FIRST STEP: KERNEL CALCULATION
        if self.kernel_type == 'lineare':
            return np.dot(input1, input2.T) # for linear kernal
        elif self.kernel_type == 'gaussiana':
            return np.exp(-self.gamma * np.linalg.norm(input1[:, np.newaxis] - input2[np.newaxis, :], axis=2) ** 2) #e ^-gamma ||X-y|| ^2  for guassian'''
    
    def weights_training(self):  #SECOND STEP: OUTER CYCLE
        K = self.kernel(self.features_matrix, self.features_matrix)
        P = matrix(np.outer(self.new_label, self.new_label) * K)
        q = matrix(-np.ones(self.samples_len))
        #https://cvxopt.org/userguide/coneprog.html?highlight=qp#quadratic-cone-programs
        #x is surely a vector with samples lenght size (in this case 105), P = 105x105
        # x.T * 0.5 * P * x + q * x 
        #P is the matrix of quadratic part of the conic to resolve and q is a vector for the linear part
        G = matrix(np.vstack((-np.eye(self.samples_len), np.eye(self.samples_len)))) #np.eye is identical matrix, G is 2*105 x 105
        h = matrix(np.hstack((np.zeros(self.samples_len), np.ones(self.samples_len) * 1.0))) #h is 1 x 2*105
        #this part imposes the condition G * x + s = h, with s > 0, this means a G * x + h > 0, so it's a disequation 
        A = matrix(self.new_label, (1, self.samples_len), 'd') #A is 1 x 105 and "d" is double (np.float64)
        b = matrix(0.0)
        #A * x = b

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(solution['x']).flatten() #take the solutions of the x vector and make them 1 dimensional
        w = np.sum(np.dot(self.alpha * self.new_label[:, None], self.features_matrix), axis=0)

        sv = (self.alpha > 1e-4) #soft classifier
        b = np.mean(self.new_label[sv] - np.dot(self.features_matrix[sv], w))

        #print("Pesi:", w)
        #print("Intercetta:", b)
        return np.concatenate((np.expand_dims(b, axis =0),w))

    """ 
        y_mul_kernal = np.outer(self.new_label, self.new_label) * self.kernel(self.features_matrix, self.features_matrix) # yi yj K(xi, xj) MATRICE n x n
        while self.step <= EPOCHS:  
            self.gradient_vec = (np.ones(self.samples_len)  - np.dot(y_mul_kernal, self.alpha)) # 1 – yk ∑ αj yj K(xj, xk)  vettore n
            self.alpha += LR * self.gradient_vec # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
            self.alpha = np.clip(self.alpha, 0, self.C) # 0<α<C
            #loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal) # ∑αi – (1/2) ∑i ∑j αi αj yi yj K(xi, xj)     
            #print(loss)
            self.step += 1 
        alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0] #it return inedexes where conditions are satisfied
        # for intercept b, we will only consider α which are 0<α<C 
        b_list = []        
        for index in alpha_index:
            b_list.append(self.new_label[index] - (self.alpha * self.new_label).dot(self.kernel(self.features_matrix, self.features_matrix[index])))
        self.b = np.mean(b_list) # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }

        #(trained_alphas_vec[1:] * self.label).dot(self.kernel(self.features_matrix, self.features_matrix)) + trained_alphas_vec[0] 
        #above the formula for label preditcion = (weights) * (features) = (alpha * label * features) * features = alpha * label * kernel(features, features) 
        #IMPORTANT SVM RETURNS 1 OR -1 VECTORS ONLY FOR THE VECTORS AND VALUES > 1 OR < -1 FOR OTHER POINTS, SO FOR PREDICTION WE NEED TH SIGNES OF THE VALUES
        
        w = np.sum(np.dot(self.alpha * self.new_label[:, None], self.features_matrix), axis=0)
        return np.concatenate( (np.expand_dims(self.b, axis=0) , w) )
        """

    def train_prediction(self, trained_weights_vec):
        return -1 * np.sign(np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0])
    
    def test_prediction(self, test_matrix, trained_weights_vec):
        return -1 * np.sign(np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0])
                
    def train_score(self, trained_weights_vec):
        return np.mean(self.new_label == self.train_prediction(trained_weights_vec))
    
    def one_vs_all(self, test_matrix):
        self.indexes, self.predictions = [], [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            self.indexes.append(a); self.predictions[a] = self.test_prediction(test_matrix, self.trained_weights_matrix[a]); 
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        #print(df)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] == None and df["pred"][i][j] == 1.0:
                        final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] == None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float)

class Neural_Network:
    def __init__(self, features_matrix):
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0] 
        self.features_len = features_matrix.shape[1]
        self.label = LABELS_VEC
        self.step = START
        self.hidden_neurons = 250
        self.label_len = len(set(self.label))
        self.W1 = (np.random.rand(self.hidden_neurons, self.features_len) - 0.5)
        self.b1 = (np.random.rand(self.hidden_neurons, 1) - 0.5)
        self.W2 = (np.random.rand(self.label_len, self.hidden_neurons) - 0.5)
        self.b2 = (np.random.rand(self.label_len, 1) - 0.5)


    def sigmoid(self, Z): #SECOND STEP: Z CALCULATION AND SIGMOID
        return 1 / (1 + np.e ** (-Z))

    def sigmoid_deriv(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def forward_prop(self, input_matrix):
        input_matrix = input_matrix.T
        self.Z1 = self.W1.dot(input_matrix) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.Z1, self.A1, self.Z2, self.A2

    def one_hot(self):
        self.one_hot_Y = np.zeros((self.label.size, self.label.max()+1))
        self.one_hot_Y[np.arange(self.label.size), self.label] = 1
        self.one_hot_Y = self.one_hot_Y.T
        return self.one_hot_Y

    def backward_prop(self):
        self.one_hot_Y = self.one_hot()
        self.dZ2 = self.A2 - self.one_hot_Y
        self.dW2 = 1 / self.features_len * self.dZ2.dot(self.A1.T)
        self.db2 = 1 / self.features_len * np.sum(self.dZ2)
        self.dZ1 = self.W2.T.dot(self.dZ2) * self.sigmoid_deriv(self.Z1)
        self.dW1 = 1 / self.features_len * self.dZ1.dot(self.features_matrix)
        self.db1 = 1 / self.features_len * np.sum(self.dZ1)
        return self.dW1, self.db1, self.dW2, self.db2

    def update_params(self):
        self.W1 = self.W1 - LR * self.dW1
        self.b1 = self.b1 - LR * self.db1    
        self.W2 = self.W2 - LR * self.dW2  
        self.b2 = self.b2 - LR * self.db2    
        return self.W1, self.b1, self.W2, self.b2

    def gradient_descent(self):
        for i in range(EPOCHS):
            self.Z1, self.A1, self.Z2, self.A2 = self.forward_prop(self.features_matrix)
            self.dW1, self.db1, self.dW2, self.db2 = self.backward_prop()
            self.W1, self.b1, self.W2, self.b2 = self.update_params()
        return self.W1, self.b1, self.W2, self.b2
    
    def prediction(self, test_matrix):
        _, _, _, a2 = self.forward_prop(test_matrix)
        return np.argmax(a2, 0)

    def score(self):
        return np.sum(self.prediction(self.features_matrix) == self.label) / self.label.size
    
    def one_vs_all(self, test_matrix): #dummy function for plot
        return self.prediction(test_matrix)

def normalization(features_matrix):
    norm_features_matrix = np.zeros(shape=features_matrix.shape)
    for i in range(features_matrix.shape[1]):
        col_mean = st.mean(features_matrix[:,i])
        col_stdev = st.stdev(features_matrix[:,i])
        norm_features_matrix[:,i] = (features_matrix[:,i] - col_mean) / col_stdev
    return norm_features_matrix

def one_vs_all_label(labels_vec):
    target_list = list(set(labels_vec))
    target_matrix=[]
    for i in range(len(target_list)):
        single_target_vec = list(np.where(labels_vec == target_list[i], 1, 0))
        target_matrix.append(single_target_vec)
    return target_matrix

def plot(classifier, title=''):
        test_idx = range(len(TEST_MATRIX))
        markers = ('o', 's', '^', 'v', '<')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(GLOBAL_LABELS))])
        for idx, cl in enumerate(np.unique(GLOBAL_LABELS)):
            plt.scatter(x=GLOBAL_MATRIX[GLOBAL_LABELS == cl, 0], y=GLOBAL_MATRIX[GLOBAL_LABELS == cl, 1], 
                        alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor='black')      
                # highlight test examples
        if test_idx:
            # plot all examples
            X_test, y_test = TEST_MATRIX[test_idx, :], GLOBAL_LABELS[test_idx]

            plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                        alpha=1.0, linewidth=1, marker='o', s=100, label='Test set')        
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 200)
        yy = np.linspace(ylim[0], ylim[1], 200)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = classifier.one_vs_all(xy).reshape(XX.shape)
        ax.contourf(XX, YY, Z, alpha=0.3, cmap=cmap)
        plt.title(title)
        plt.xlabel('Petal length [standardized]')
        plt.ylabel('Petal width [standardized]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    DATASET = load_iris(as_frame=True)
    GLOBAL_MATRIX = normalization( DATASET.data.values [:,[2,3]])
    GLOBAL_LABELS = load_iris().target # shape = 150 labels, one for each sample
    FEATURES_MATRIX, TEST_MATRIX, GLOB_LABELS_VEC, TEST_LABELS = train_test_split(GLOBAL_MATRIX, GLOBAL_LABELS, test_size=0.3, random_state=1, stratify=GLOBAL_LABELS)
    LABELS_VEC = GLOB_LABELS_VEC

    START = 0 # begin of outer cycle
    EPOCHS = 1000 # end of outer cycle 1000 is the best bc perceptron is slower
    LR = 0.001

    ask= str(input("""1. Perceptron
  2. Adaline
  3. Logistic Regression
  4. Support Vector Machine (linear kernel)
  5. Support Vector Machine (gaussian kernel)
  6. Neural Network
  Select a classifier: """, ))

       
    if ask == "1":
        LABELS_MATRIX = one_vs_all_label(LABELS_VEC) #set the labels vector to do one vs all: example [0,0,1,1,2,2] bacomes [[1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1]]
        TRAINED_wEIGHTS_MATRIX = [None] * len(LABELS_MATRIX) #train a classifier for each class and append here
        SCORES = [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            LABELS_VEC = LABELS_MATRIX[a]
            p = Perceptron(FEATURES_MATRIX, LABELS_VEC)
            TRAINED_wEIGHTS_MATRIX[a] = list(p.weights_training()) #train weights and save them
            SCORES[a] = p.train_score(TRAINED_wEIGHTS_MATRIX[a]) #measure score and save it for each class
        PREDICTED_TRAIN_LABEL = p.one_vs_all(FEATURES_MATRIX) #put togeter all the classifier
        PREDICTED_TEST_LABEL = p.one_vs_all(TEST_MATRIX)
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Perceptron")

    elif ask == "2":
        LABELS_MATRIX = one_vs_all_label(LABELS_VEC)
        TRAINED_wEIGHTS_MATRIX = [None] * len(LABELS_MATRIX) 
        SCORES = [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            LABELS_VEC = LABELS_MATRIX[a]
            p = Adaline(FEATURES_MATRIX, LABELS_VEC)
            TRAINED_wEIGHTS_MATRIX[a] = list(p.weights_training())
            SCORES[a] = p.train_score(TRAINED_wEIGHTS_MATRIX[a])
        PREDICTED_TRAIN_LABEL = p.one_vs_all(FEATURES_MATRIX) 
        PREDICTED_TEST_LABEL = p.one_vs_all(TEST_MATRIX)
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Adaline")

    elif ask == "3":
        LABELS_MATRIX = one_vs_all_label(LABELS_VEC)
        TRAINED_wEIGHTS_MATRIX = [None] * len(LABELS_MATRIX) 
        SCORES = [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            LABELS_VEC = LABELS_MATRIX[a]
            p = Logistic_Regression(FEATURES_MATRIX, LABELS_VEC)
            TRAINED_wEIGHTS_MATRIX[a] = list(p.weights_training()) 
            SCORES[a] = p.train_score(TRAINED_wEIGHTS_MATRIX[a]) 
        PREDICTED_TRAIN_LABEL = p.one_vs_all(FEATURES_MATRIX) 
        PREDICTED_TEST_LABEL = p.one_vs_all(TEST_MATRIX)
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Logistic Regression")

    elif ask == "4":
        LABELS_MATRIX = one_vs_all_label(LABELS_VEC)
        TRAINED_wEIGHTS_MATRIX = [None] * len(LABELS_MATRIX)
        SCORES = [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            LABELS_VEC = LABELS_MATRIX[a]
            p = Support_Vector_Machine(FEATURES_MATRIX, "lineare", LABELS_VEC)
            TRAINED_wEIGHTS_MATRIX[a] = list(p.weights_training()) 
            SCORES[a] = p.train_score(TRAINED_wEIGHTS_MATRIX[a])
        PREDICTED_TRAIN_LABEL = p.one_vs_all(FEATURES_MATRIX) 
        PREDICTED_TEST_LABEL = p.one_vs_all(TEST_MATRIX)
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Logistic Regression with Linear Kernel")

    elif ask == "5":
        LABELS_MATRIX = one_vs_all_label(LABELS_VEC)
        TRAINED_wEIGHTS_MATRIX = [None] * len(LABELS_MATRIX) 
        SCORES = [None] * len(LABELS_MATRIX)
        for a in range(len(LABELS_MATRIX)):
            LABELS_VEC = LABELS_MATRIX[a]
            p = Support_Vector_Machine(FEATURES_MATRIX, "gaussiana", LABELS_VEC)
            TRAINED_wEIGHTS_MATRIX[a] = list(p.weights_training()) 
            SCORES[a] = p.train_score(TRAINED_wEIGHTS_MATRIX[a]) 
        PREDICTED_TRAIN_LABEL = p.one_vs_all(FEATURES_MATRIX) 
        PREDICTED_TEST_LABEL = p.one_vs_all(TEST_MATRIX)
        #print('Misclassified Train Samples: %d' % (PREDICTED_TRAIN_LABEL != GLOB_LABELS_VEC).sum(), " / ", len(GLOB_LABELS_VEC))
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Logistic Regression with Gaussian Kernel")
        
    elif ask == "6":
        p = Neural_Network(FEATURES_MATRIX)
        w = p.gradient_descent()
        PREDICTED_TEST_LABEL = p.prediction(TEST_MATRIX)
        print('Misclassified Test Samples: %d' % (PREDICTED_TEST_LABEL != TEST_LABELS).sum(), " / ", len(TEST_LABELS))
        plot(p, title="Neural Network")
        #print("score", p.score())
    
    else:
        print("Invalid input. Bye bye")
        exit()



