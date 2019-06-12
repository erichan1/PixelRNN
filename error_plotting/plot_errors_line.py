import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import csv

# plots specifically for train_l vs. test_l
# plot_train = true -> plots train error, else plots test error
# remove_first = true -> remove first epoch from plot
def plot(all_files, title, plot_train, remove_first):
    errors = []

    for filename in all_files:
        data = csv.reader(open(filename, 'r'), delimiter=",")
        train_l, test_l = [], []


        # define x, y, params
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
        x = np.array([i for i in range(len(train_l))])

        # create params dictionary
        params = {}
        for i in range(len(paramNames)):
            params[paramNames[i]] = paramVals[i]

        # check if train loss and test loss have same dim
        if(len(train_l) != len(test_l)):
            print("error: train/test -> different dimensions")

        # sets y to train or test
        if(plot_train):
            y = train_l
        else:
            y = test_l

        # possibly removes first epoch
        if(remove_first):
            x = x[1:]
            y = y[1:]

        # set the label
        if(params['use_swapout'] == 'True'):
            label_ = 'swapout: p1 = {}, p2 = {}'.format(params['swapout_p1'], params['swapout_p2'])
        else:
            label_ = 'no swapout'

        # plot!  
        plt.plot(x, y, label = label_)


    plt.xlabel("Epochs")
    plt.ylabel("Error (Cross Entropy)")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    # put everything into a folder called errors at the same level as this py file
    path = 'to_plot'
    all_files = glob.glob(path + "/*.csv")

    cnn_test_title = "PixelCNN Test Error: Swapout with Varying Probabilities"
    cnn_train_title = "PixelCNN Train Error: Swapout with Varying Probabilities"
    # cnn_legend = ['p1 = 0.7, p2 = 0.2', 'no swapout', 'p1 = 0.6, p2 = 0.4', 'p1 = 0.3, p2 = 0.8',
    #     'p1 = 0.4, p2 = 0.9', 'p1 = 1.0, p2 = 1.0', 'p1 = 0.25, p2 = 0.75']

    plot(all_files, cnn_train_title, True, False)
    plot(all_files, cnn_test_title, False, False)
    plot(all_files, cnn_train_title, True, True)
    plot(all_files, cnn_test_title, False, True)


    # rnn_test_title = "PixelRNN Test Error: Swapout vs Resnet vs None"
    # rnn_train_title = "PixelRNN Train Error: Swapout vs Resnet vs None"
    # rnn_legend = ['None', 'Resnet', 'Swapout']

    # # plots, with first epoch
    # plot(all_files, rnn_train_title, rnn_legend, True, False)
    # plot(all_files, rnn_test_title, rnn_legend, False, False)

    # # plots, without first epoch
    # plot(all_files, rnn_train_title, rnn_legend, True, True)
    # plot(all_files, rnn_test_title, rnn_legend, False, True)

