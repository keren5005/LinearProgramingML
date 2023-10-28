import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

def test_for_seprability(H, M):
    X = np.c_[H, np.ones(H.shape[0])]
    Y = np.c_[-M, -np.ones(M.shape[0])]
    A = np.r_[X, Y]
    b = -np.ones(A.shape[0]).reshape(-1, 1)
    c = np.zeros(A.shape[1])
    res = linprog(c=c, A_ub=A, b_ub=b, options={"disp": False})
    return res.success


def linprog_on_iris_dataset():
    '''
    This is a Python function that performs linear programming to test if two sets of data are linearly separable.
    It also visualizes the data and the linear separator if the data is linearly separable.
    '''
    from sklearn import datasets
    data = datasets.load_iris()

    # create a DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = pd.DataFrame(data.target)
    names = data.target_names

    dic = {0: 'setosa', 1: 'versicolor', 2: 'verginica'}

    sc = StandardScaler()
    sc.fit(df.iloc[:, [2, 3]].values)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            plt.clf()
            plt.figure(figsize=(10, 6))

            plt.title(data.target_names[i] + ' vs ' + data.target_names[j])
            plt.xlabel(data.feature_names[2])
            plt.ylabel(data.feature_names[3])

            H = df[df['Target'] == i].iloc[:, [2, 3]].values
            M = df[df['Target'] == j].iloc[:, [2, 3]].values

            H = sc.transform(H)
            M = sc.transform(M)

            rc = test_for_seprability(H, M)

            if rc:
                label = 'Linearly separable'
                color = 'green'
                # find the linear separator between H and M
                X = np.concatenate((H, M))
                y = np.concatenate((np.ones(H.shape[0]), -np.ones(M.shape[0])))
                clf = LinearSVC(random_state=0, tol=1e-5)
                clf.fit(X, y)
                w = clf.coef_[0]
                b = clf.intercept_[0]
                xp = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
                yp = -(w[0] / w[1]) * xp - b / w[1]
                plt.plot(xp, yp, color='green', linestyle='--')
            else:
                label = 'Not linearly separable'
                color = 'red'
            plt.scatter(H[:, 0], H[:, 1], label=data.target_names[i])
            plt.scatter(M[:, 0], M[:, 1], label=data.target_names[j])
            plt.annotate(data.target_names[i] + ': ' + label, xy=(H[:, 0].mean(), H[:, 1].mean()),
                         xytext=(3, 3), textcoords='offset points', color=color)
            plt.annotate(data.target_names[j] + ': ' + label, xy=(M[:, 0].mean(), M[:, 1].mean()),
                         xytext=(3, 3), textcoords='offset points', color=color)
            plt.legend()
            plt.show()


def linprog_on_simple_dataset():
    '''
    This function performs linear regression using the method from the SciPy library
    on 'simple dataset'. The function first standardizes the input x values and adds a column of
    ones to the input x matrix. It then creates the appropriate c, A, and b matrices for the linear programing,
    where c represents the objective function,  A represents the constraints, and b represents the right-hand side
    of the constraints.
    '''
    df = pd.read_csv('sr.csv')
    n = len(df)

    x = df['x'].values
    y = df['y'].values
    sc = StandardScaler()

    x = np.c_[np.ones(n), x.reshape(-1, 1)]
    y = y.reshape(-1, 1)

    m = x.shape[1]
    c_array = []
    bounds = []
    for i in range(m):
        c_array.append(0)
        bounds.append((None, None))
    for i in range(n):
        c_array.append(1)
        bounds.append((0, None))

    c = np.array(c_array)
    A_array = []
    b_array = []

    for r in range(n):
        row1 = [x[r][0]]
        row2 = [-x[r][0]]
        b_array.append(y[r][0])
        b_array.append(-y[r][0])
        for z in range(r):
            row1.append(0)
            row2.append(0)
        row1.append(-1)
        row2.append(-1)
        for z in range(n + m - len(row1)):
            row1.append(0)
            row2.append(0)
        A_array.append(row1)
        A_array.append(row2)
    A = np.array(A_array)
    b = np.array(b_array)

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, options={"disp": False})

    beta = []
    for i in range(m):
        beta.append(res.x[i])

    print(f'Beta = {beta}')

    #calc R^2
    y_hat = np.dot(x, np.array(beta))
    y = np.squeeze(y)
    ybar = np.mean(y)
    ssreg = np.sum(np.power(y - y_hat, 2))
    sstot = np.sum(np.power(y - ybar, 2))
    r2 = 1.0 - ssreg / sstot
    print(f'R2 = {r2}')


if __name__ == '__main__':
    print("Question 5 section B")
    linprog_on_iris_dataset()
    print("==========================================================")
    print("Question 5 section D")
    linprog_on_simple_dataset()
