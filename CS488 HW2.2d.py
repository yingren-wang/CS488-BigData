import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def lin_reg(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    beta = SS_xy / SS_xx
    alpha = m_y - beta * m_x

    return alpha, beta


def plot_lin_reg_model(x, y, a, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted response vector
    # y_pred = alpha + beta * x
    y_pred = a + b * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def main():
    # observations
    x = np.array([3,5,7,9,12,15,18])
    y = np.array([100,250,330,590,660,780,890])

    # estimating coefficients
    a, b = lin_reg(x, y)
    print("Estimated coefficients:\n alpha (slope intercept) = {}   "
          "\n beta (slope) = {}".format(a, b))

    # plotting regression line
    plot_lin_reg_model(x, y, a, b)

    # compare with sklearn
    X = np.array([3,5,7,9,12,15,18])
    Y = np.array([100,250,330,590,660,780,890])
    XX = np.reshape(X, (-1, 1))
    reg = LinearRegression().fit(XX, Y)

    # Coefficient of determination
    # c_det = reg.score(XX, Y)
    # print("Estimated Coefficient of determination = {}".format(c_det))

    # estimating coefficients
    print("Estimated coefficients:\n alpha (slope intercept) = {}   "
          "\n beta (slope) = {}".format(reg.intercept_, reg.coef_))

    # Predict a new data point
    # add a new point and see how the data change
    new_x = 30
    new_y = reg.predict(np.reshape(new_x, (-1, 1)))
    print("For new x = {} the estimated new y prediction = {}".format(new_x, new_y))

    # check if should invest in ABC or not
    if (float(new_y / 890)) > 1.5:
        print("Do not invest in ABC")
    else:
        print("Invest in ABC")

    # plotting regression line
    plot_lin_reg_model(np.append(x, new_x), np.append(y, new_y), reg.intercept_, reg.coef_)


if __name__ == "__main__": main()
