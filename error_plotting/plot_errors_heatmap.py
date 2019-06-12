import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import csv

# plot_train = true -> plots train error, else plots test error
# creates the heatmap matrix. p1 -> row index, p2 -> col index
# p1_dim, p2_dim are needed to define size of matrix
# Code is not general. Only works for 25 errors w p1 = {0, 0.2, 0.4, 0.6, 0.8, 1.0}
def generate_heatmap_matrix(all_files, plot_train):

    # initialize matrix
    error_matrix = np.zeros((5, 5))

    for filename in all_files:
        data = csv.reader(open(filename, 'r'), delimiter=",")
        train_l, test_l = [], []

        # get params, train_l, test_l from file
        c = 0
        paramNames, paramVals = [], []
        for row in data:
            # first two rows are used to make param dict
            if(c == 0):
                paramNames = row
            elif(c == 1):
                paramVals = row
            else:
                train_l.append(float(row[0]))
                test_l.append(float(row[1]))
            c += 1

        # create params dictionary
        params = {}
        for i in range(len(paramNames)):
            params[paramNames[i]] = paramVals[i]

        # check if train loss and test loss have same dim
        if(len(train_l) != len(test_l)):
            print("error: train/test -> different dimensions")

        # sets y to train or test
        if(plot_train):
            err_val = train_l[-1]
        else:
            err_val = test_l[-1]

        # set error_matrix[p1][p2] = loss at last epoch
        # do nothing if not using swapout
        if(params['use_swapout'] == 'True'):
            i = int(float(params['swapout_p1']) * 4) # 0.25 * 4 -> 1, correct index
            j = int(float(params['swapout_p2']) * 4) 
            error_matrix[i][j] = err_val

    return error_matrix

if __name__ == '__main__':
    # put everything into a folder called errors at the same level as this py file
    path = 'to_plot'
    all_files = glob.glob(path + "/*.csv")

    # cnn_test_title = "PixelCNN Test Error: Swapout with Varying Probabilities"
    # cnn_train_title = "PixelCNN Train Error: Swapout with Varying Probabilities"

    print(generate_heatmap_matrix(all_files, True))
    print(generate_heatmap_matrix(all_files, False))

