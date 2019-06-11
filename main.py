import pixelrnn as cnn
import numpy as np

if __name__ == '__main__':
    # p1, p2, swapout, residual, max_epoch

    cnn.run_cnn(0, 0, False, False, 20)
    p1_arr = np.linspace(0,1,5)
    p2_arr = np.linspace(0,1,5)
    for p1 in p1_arr:
        for p2 in p2_arr:
            cnn.run_cnn(p1, p2, True, False, 20)