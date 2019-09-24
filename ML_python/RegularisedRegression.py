import random
import numpy as np
import pylab as plt
# read red wine data
red_data = np.loadtxt('winequality-red.csv', delimiter=';', dtype='str')
# store the value in matrix
red_matrix = red_data[1:, :].astype('float')
sample_number = red_matrix.shape[0]
l = list()
index = sample_number
# split data in to 70% train samples and 30% test samples
while index > 0:
    index -= 1
    l.append(index)
train_index = random.sample(l, int(sample_number*0.7))
test_index = list()
for i in l:
    if i not in train_index:
        test_index.append(i)
# select each sample from the whole data matrix
train = red_matrix[train_index, :]
test = red_matrix[test_index, :]
# create feature matrix for calculation
x_train = train[:, :11]
t_train = train[:, 11]
x_test = test[:, :11]
t_test = test[:, 11]
X = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
X_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
# calculate w with each lambada
for lamb in [0, 0.01, 0.1, 1, 10]:
    w = np.linalg.solve(np.dot(X.T, X) + lamb*np.identity(X.shape[1]), np.dot(X.T, t_train))
    # predict the result
    t_predict = np.dot(X_test, w)
    # int the result
    t_int_predict = t_predict.round()
    # calculate mean squared error
    MSE = ((t_int_predict-t_test)*(t_int_predict-t_test)/t_test.shape[0]).sum()
    print 'lambda =', lamb, '   MSE = ', MSE
    # plot true and predict result
    # plt.plot(t_predict, 'yo')
    plt.plot(t_int_predict, 'ro')
    plt.plot(t_test, 'bo')
    plt.xlabel('sample')
    plt.ylabel('quality')
    plt.show()


