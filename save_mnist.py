import util
import numpy as np


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', fmt='%d')


def main():
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    save_to_csv(x_train, 'train.csv')
    save_to_csv(y_train, 'train_label.csv')
    save_to_csv(x_test, 'test.csv')
    save_to_csv(y_test, 'test_label.csv')


if __name__ == '__main__':
    main()