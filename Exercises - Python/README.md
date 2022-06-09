<h1 align="center">Coursera Machine Learning Andrew NG in Python</h1>
Each of these folders contains my own version of the Coursera's <a href="https://www.coursera.org/learn/machine-learning">Machine learning</a> course exercises in Python.

My intention is not to replicate exactly the exercises but to have my own flavour of them.
I am also including variations of these exercises based on my own experimentations
😊.

To complete the exercises I have used some articles from the web that have been inspirational, helpful and many times a step-by-step guide of what to do. I am sincerely thankful to the AI community and the open-source mindset that drives it. I truly believe that this helps to increase the potential of this incredible technology.

# Contents

Following a description of the exercises plus the techniques and items that I have learned on each of them so it can be used in the future as models for new exercises:

## [Exercise 1 - Linear Regression](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex1%20-%20Linear%20Regression) 🍼
[ex1-Linear Regression](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex1%20-%20Linear%20Regression/ex1-Linear%20Regression.ipynb): Linear regression with single and multiple variables.
* Pandas:
  * Read CSV file
  * Convert data into a numpy array with the <code>iloc[:,:].to_numpy()</code> method
  * <code>describe()</code> method
* Matplotlib:
  * Basic 2D plotting: dataset, prediction line, markers, legend
  * 3D plot of the cost function surface
  * Contour diagram showing the cost function minumum and the level curves
* Numpy:
  * Zeros array
  * Ones array
  * append column to an existing array
* Scikit-learn:
  * <code>preprocessing.MinMaxScaler</code>: Data normalization
  * <code>preprocessing.StandardScaler</code>: Data standardization
* Python:
  * Import a .py file into the notebook
  * Function definition
  * Output: <code>print('The Cost value is %.2f' %cost)</code>

## [Exercise 2 - Logistic Regression](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex2%20-%20Logistic%20Regression) 🚀
[ex2-Logistic regression](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex2%20-%20Logistic%20Regression/ex2-Logistic%20regression.ipynb)
* Python:
  * Numpy formater:
```
grad = np.array([-0.1000125, -12.0092543, -11.2628234])
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('\t',grad.flatten())
```
  * Output: <code>print('Cost: {:.3f}'.format(cost))</code>
* Unconstrained Minimization algortihm <code>scipy.optimize</code>

[ex2m-Logistic regression with regularization](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex2%20-%20Logistic%20Regression/ex2m-Logistic%20regression%20with%20regularization.ipynb)
* Logistics regression with regularization incrementing the <b>polynomial features</b>
* Pandas:
  * <code>head()</code> and <code>describe()</code> methods
* Matplotlib:
  * Plotting a <b>non-linear decision boundary</b>: Using <code>np.linspace</code> to create a grid and <code>plt.contour</code> to plot the boundary
  * Write a greek character in the title: <code>plt.title('Training data with decision boundary ($\lambda$=' + str(lmbd) + ')')</code>

## [ex3 - MultiClass Classification + NN intro](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex3%20-%20MultiClass%20Classification%20%2B%20NN%20intro) 🚁
[ex3-MultiClass Classification](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex3%20-%20MultiClass%20Classification%20%2B%20NN%20intro/ex3-MultiClass%20Classification.ipynb): Use logistics regression to classify in more than two classes using one-vs-all classification.
* <code>scipy.io</code>
  * Loading a MatLab dataset with <code>loadmat</code>
  * <code>keys()</code> method to see the dataset structure.
* numpy:
  * <code>np.where</code> to convert data in an array
  * <code>np.random.choice(m, 100, replace=False)</code> to create a list of random indexes
* matplotlib:
  * Create a <code>plt.subplot</code> to visualize the MNIST sample
* Calculate accuracy with <code>print('Accuracy: {:.2f} %'.format(np.mean(pred == y) * 100))</code>

[ex3nn-MNIST on a NN already trained](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex3%20-%20MultiClass%20Classification%20%2B%20NN%20intro/ex3nn-MNIST%20on%20a%20NN%20already%20trained.ipynb): Into to Neural Networks using feedforward propagation.
* Replace all 10 by 0: <code>y[y == 10] = 0</code>
* <code>np.roll(x, 1, axis=0)</code> to swicth the first and the last rows
* Matplotlib: <code>plt.figure()</code> and <code>subplots</code>

[ex3-MNIST image visualization](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex3%20-%20MultiClass%20Classification%20%2B%20NN%20intro/ex3-MNIST%20image%20visualization.ipynb): Experimentation to process and visualize the MNIST dataset.
* <code>os</code> library: Open and read files
* matplotlib: <code>subplot</code> to create a grid with images

## [ex4 - Neural Networks](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex4%20-%20Neural%20Networks) 🌋
[ex4-neural networks](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex4%20-%20Neural%20Networks/ex4-neural%20networks.ipynb)
* Numpy:
  * <code>flatten</code>: convert into a 1D vector
  * <code>amax</code>: return the maximum from an axis or from an array

[ex4b-neural network class](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex4%20-%20Neural%20Networks/ex4b-neural%20network%20class.ipynb): Same as the previous one but creating a NN class
* Python:
  * Class creation and methods
  * <code>@staticmethod</code>
* <code>for i in reversed(range(a, b))</code>
* Numpy:
  * <code>np.argmax</code>: Returns the indices of the maximum values along an axis

[ex4c - Neural Network Class with MNIST dataset](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex4%20-%20Neural%20Networks/ex4c%20-%20Neural%20Network%20Class%20with%20MNIST%20dataset.ipynb): In this case we combine both "ex4b-neural network class" and "ex3-MNIST image visualization"

[Image processing experiments with Pillow](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex4%20-%20Neural%20Networks/Image%20processing%20experiments%20with%20Pillow.ipynb)
* Pillow:
  * Open an image and extract basic information
  * Show the image
  * Resizing and converting images to different color scales
  * Save converted images

## [ex5 - Regularized Linear Regression + Bias vs Variance](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex5%20-%20Regularized%20Linear%20Regression%20%2B%20Bias%20vs%20Variance) ⏰

[ex5-Regularized Linear Regression and Bias v.s. Variance](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex5%20-%20Regularized%20Linear%20Regression%20%2B%20Bias%20vs%20Variance/ex5-Regularized%20Linear%20Regression%20and%20Bias%20v.s.%20Variance.ipynb)

* Linear regression with regularization
* Bias-Variance balance
* Learning curves
* Cross validation dataset
* Polynomial regression using <code>sklearn.preprocessing</code>
  * <code>PolynomialFeatures</code>
  * <code>StandardScaler</code>
* Adjust with various regularization (&lambda;) parameters
* Selecting the optimal &lambda; checking the error on the cross validation set

[ex5b-Regularized Linear Regression and Bias v.s. Variance (with sklearn)](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex5%20-%20Regularized%20Linear%20Regression%20%2B%20Bias%20vs%20Variance/ex5b-Regularized%20Linear%20Regression%20and%20Bias%20v.s.%20Variance%20(with%20sklearn).ipynb)
* scikit-learn:
  * <code>sklearn.preprocessing.PolynomialFeatures</code>
  * <code>sklearn.preprocessing.StandardScaler</code>
  * <code>sklearn.linear_model.LinearRegression</code>
  * <code>sklearn.linear_model.Ridge</code> to apply regularization
  * <code>sklearn.pipeline.make_pipeline</code>
  * <code>sklearn.model_selection.learning_curve</code>
  * <code>sklearn.model_selection.validation_curve</code>

## [ex6 - SVM](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex6%20-%20SVM) 💣

[Ex6a-Support Vector Machines - Linear Model](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex6%20-%20SVM/Ex6a-Support%20Vector%20Machines%20-%20Linear%20Model.ipynb)
* Scikit-learn:
  * <code>sklearn.svm.SVC</code>
  * <code>kernel='linear'</code>
  * <code>C</code> as regularization parameter
* Plotting the decision boundary

[Ex6b-Support Vector Machines - nonlinear](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex6%20-%20SVM/Ex6b-Support%20Vector%20Machines%20-%20nonlinear.ipynb)
* Scikit-learn:
  * <code>sklearn.svm.SVC</code>
  * <code>kernel='rbf'</code>
  * <code>gamma</code> as parameter to determine the influence of each training example
  * <code>GridSearchCV</code> to find the optimal combination of hyperparameters
* Plotting the decision boundary

[Ex6c - Email Features extraction](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex6%20-%20SVM/Ex6c%20-%20Email%20Features%20extraction.ipynb)
* Text file parsing:
  * Read the file line by line
  * Regular Expression: <code>re.sub</code> find and replace the regular expression
* Python:
  * <code>lambda</code>
  * <code>if</code> one-liner
* Jupyter notebook:
  * Mathematical Capital letters, upperscript and underscript: <code>$R^{n}$</code> and <code>$x_i=1$</code>
* Natural Language Processing (NLP):
  * "Bag of words" (vocabulary list)
  * PorterStemmer algortihm

[Ex6c - Spam classification with SVM](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex6%20-%20SVM/Ex6c%20-%20Spam%20classification%20with%20SVM.ipynb)
* scikit-learn:
  * <code>GridSearchCV</code> for hyperparameters tuning
* NLP:
  * email features extraction

[MNIST classification with SVM](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex6%20-%20SVM/MNIST%20classification%20with%20SVM.ipynb)
Using the MNIST original dataset, process the files and run SVM to classify the numbers

## [ex7 - Kmeans + PCA](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA) 🐦
[Ex7 - Kmeans clustering](http://localhost:8888/notebooks/OneDrive/Personal%20Projects/Coursera%20-%20Machine%20Learning%20Andrew%20Ng/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA/Ex7%20-%20Kmeans%20clustering.ipynb)
* Kmeans:
  * My own implementation of a Kmeans class
* Matplotlib:
  * centroids visualization. Incremental steps to reach a stable centroid

[Ex7 - Kmeans clustering with scikit-learn + identify number of clusters](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA/Ex7%20-%20Kmeans%20clustering%20with%20scikit-learn%20%2B%20identify%20number%20of%20clusters.ipynb)
* scikit-learn:
  * <code>sklearn.cluster.KMeans</code>: Kmeans implementation
  * <code>sklearn.metrics.silhouette_score</code>: Silhouette score to detect number of clusters
* Kmeans: How to best select the number of clusters
  * Elbow method: Using the Sum of Squared Errors (SSE) for different values of <code>k</code>(number of clusters), identify the value of <code>k</code> where SSE decreases most rapidly.
  * Silhouette method: how similar a point is to its own cluster comparing to the other clusters. A high value is desireable.

[Ex7 - Image compression with Kmeans](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA/Ex7%20-%20Image%20compression%20with%20Kmeans.ipynb)
We reduce each of the RGB colors (24-bits -> milions of colors) to its closest color on a set of 16 colors.
* Matplotlib:
  * Show a grid of images
* PIL:
  * Manage images as an array with numpy
* numpy:
  * <code>astype</code> to convert an array on a type

[Ex7b - Principal Component Analysis](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA/Ex7b%20-%20Principal%20Component%20Analysis.ipynb)
* Our own implementation of PCA

[Ex 7c - Face Image Dataset PCA](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex7%20-%20Kmeans%20%2B%20PCA/Ex%207c%20-%20Face%20Image%20Dataset%20PCA.ipynb)
Identify the number of components required to represent a % of the model.
* scikit-learn:
  * <code>sklearn.decomposition.PCA</code>
  * <code>PCA(n_components=K)</code> & <code>pca.inverse_transform</code>
* motplotlib:
  * draw images
* numpy:
  * <code>np.cumsum</code> cummulative sum to identify the number of dimensions that describe a % of the model

## [ex8 - Anomaly Detection + Recommender Systems](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/tree/master/Exercises%20-%20Python/ex8%20-%20Anomaly%20Detection%20%2B%20Recommender%20Systems)

[Ex8 - Anomaly Detection](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex8%20-%20Anomaly%20Detection%20%2B%20Recommender%20Systems/Ex8%20-%20Anomaly%20Detection.ipynb)
* scikit-learn:
  * F1 score
* matplotlib:
  * contour lines
* numpy:
  * <code>count_nonzero</code>
  * <code>mean</code> and <code>var</code>

[Ex 8b - Recommender Systems](https://github.com/conradoca/Coursera---Machine-Learning-Andrew-Ng/blob/master/Exercises%20-%20Python/ex8%20-%20Anomaly%20Detection%20%2B%20Recommender%20Systems/Ex%208b%20-%20Recommender%20Systems.ipynb)
* <code>from scipy import optimize</code>: optimize using an array as parameters
* numpy:
  * <code>where</code>
  * <code>nanmean</code>
  * <code>nan_to_num</code>
  * <code>concatenate</code>
  * <code>column_stack</code>
