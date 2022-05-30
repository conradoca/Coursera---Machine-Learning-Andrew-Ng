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
