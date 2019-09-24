import numpy as np
# read red wine data
red_data = np.loadtxt('winequality-red.csv', delimiter=';', dtype='str')
# store the value in matrix
red_matrix = red_data[1:, :].astype('float')
sample_number = red_matrix.shape[0]
l = list()
index = sample_number
while index > 0:
    index -= 1
    l.append(index)
# initial lambda, mean squared error and accuracy
best_lamb = 100
best_MSE = 100
best_accuracy = 0
# calculate w with each lambada
for lamb in [0, 0.01, 0.1, 1, 10]:
    s = 0
    s_accuracy = 0
    # split data in to 10 groups equally, and 90% for training and 10% for testing
    for K in range(10):
        test_index = l[K * int(sample_number * 0.1):(K + 1) * int(sample_number * 0.1)]
        train_index = list()
        for i in l:
            if i not in test_index:
                train_index.append(i)
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
        w = np.linalg.solve(np.dot(X.T, X) + lamb * np.identity(X.shape[1]), np.dot(X.T, t_train))
        # predict the result
        t_predict = np.dot(X_test, w)
        # int the result
        t_int_predict = t_predict.round()
        # calculate mean squared error
        MSE = ((t_int_predict - t_test) * (t_int_predict - t_test) / t_test.shape[0]).sum()
        s += MSE
        # calculate each accuracy
        accuracy = 1-(abs(t_int_predict-t_test)/t_test/t_test.shape[0]).sum()
        # sum the accuracy
        s_accuracy += accuracy

    print 'lambda =', lamb, ' MSE = ', s / 10, 'accuracy = ', s_accuracy/10
    # compare and choose best lambda based on minimum MSE
    if best_MSE > s/10:
        best_lamb = lamb
        best_MSE = s/10
        best_accuracy = s_accuracy/10
print 'best_lamb = ', best_lamb, 'best_MSE = ', best_MSE,  'best_accuracy = ', best_accuracy