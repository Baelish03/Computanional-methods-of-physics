# Machine Learning Classifiers from Scratch
Code for my exam "Computational Methods of Physics". This project implements **several machine learning classifiers** from scratch (no `sklearn` classifiers, only dataset utilities)
1. **Perceptron**;
2. **Adaline (Adaptive Linear Neuron)**;
3. **Logistic Regression**;
4. **Support Vector Machine**
   - Linear kernel;
   - Gaussian (RBF) kernel;
5. **Neural Network**.
The program is not optimized and for more than 1000 epochs is very slow, because the goal is to understand the inner workings of classic ML algorithms by coding them manually and visualizing their decision boundaries.  

Each classifier is trained in a **one-vs-all** setting for multiclass classification and visualized on a 2D projection.

---

## Dataset
The code uses the **Iris dataset** from `sklearn.datasets`, restricted to **petal length** and **petal width** for easy visualization in 2D.  

- 150 samples  
- 3 classes (Setosa, Versicolor, Virginica)  
- Standardized features  

---

## Usage
Clone the repository and run:

```bash
python main.py
