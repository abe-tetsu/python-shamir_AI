import time

import numpy as np
import pickle
from keras.datasets import mnist
import keras.utils
import shamir

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = 1000
Accuracy_image = 100


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print("load_data: OK")
    return (x_train, y_train), (x_test, y_test)


def load_weights(save_filename):
    with open(save_filename, 'rb') as f:
        data = pickle.load(f)

    print("trained weights was saved from " + data['trained_filename'])
    return data['weights']


def save_weights(weights, save_filename, trained_filename):
    with open(save_filename, 'wb') as f:
        pickle.dump({'weights': weights, 'trained_filename': trained_filename}, f)


# dataすべてAccuracy_image倍してintに変換する
def transform_data(x_train, x_test):
    x_train_scaled_images = (x_train * 0.1 * Accuracy_image).astype(int)
    x_test_scaled_images = (x_test * 0.1 * Accuracy_image).astype(int)
    print("transform_data: OK")

    return x_train_scaled_images, x_test_scaled_images


def compare_arrays(arr1, arr2):
    # 同じ長さでなければエラー
    if len(arr1) != len(arr2):
        raise ValueError("Both input lists must have the same length. len(arr1):", len(arr1), "len(arr2):", len(arr2))

    # 各要素の差が1以内かどうかをチェック
    for i in range(len(arr1)):
        if abs(arr1[i] - arr2[i]) > 2:
            print("index:", i)
            return False
    return True


def array_max(array):
    max = array[0]
    max_idx = 0
    for i in range(len(array)):
        if max < array[i]:
            max = array[i]
            max_idx = i
    return max_idx


def predict(x, weights):
    # 入力ベクトルの長さを確認
    if len(x) != 784:
        raise ValueError("Input vector must have a length of 784.", len(x))

    # 重み行列の寸法を確認
    if len(weights) != 784 or len(weights[0]) != 10:
        raise ValueError("Weights must be a 784x10 dimensional matrix.")

    # 出力値を格納する配列を初期化（10個の出力ノードに対応）
    output_values = [0] * 10

    # 各出力ノードに対して線形結合を計算
    for i in range(10):  # 出力ノードの数だけループ
        # 重みと入力の積の合計を求める
        sum_weighted_inputs = 0
        for j in range(784):  # 各入力ノードについて
            sum_weighted_inputs += x[j] * weights[j][i]
        # 計算された値を出力値に設定
        output_values[i] = sum_weighted_inputs
        # print(output_values)

    return output_values


def dot(x, y):
    if len(x) != len(y):
        print(len(x), len(y))
        raise ValueError("Both input lists must have the same length.")
    return sum(i * j for i, j in zip(x, y))


def outer(x, y):
    result = []
    for i in x:
        for j in y:
            result.append(i * j)

    # dw[7840] から dw[784][10]に変換
    new_result = []
    for i in range(0, len(result), 10):
        new_result.append(result[i:i + 10])

    return np.array(new_result, dtype=np.int64)


def debug_weight(loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3, P):
    for index in range(len(loaded_weights)):
        print("index:", index)
        dec = shamir.array_decrypt23(loaded_weights1[index], loaded_weights2[index], P)
        print("重み, 秘密分散前:", loaded_weights[index][0], loaded_weights[index][1], loaded_weights[index][2],
              loaded_weights[index][3], loaded_weights[index][4], loaded_weights[index][5], loaded_weights[index][6],
              loaded_weights[index][7], loaded_weights[index][8], loaded_weights[index][9])
        print("重み, 秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
        print("----------------------------")


def test_dec(loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3, P):
    count = 0
    for index in range(len(loaded_weights)):
        dec = shamir.array_decrypt23(loaded_weights1[index], loaded_weights2[index], P)
        if not compare_arrays(loaded_weights[index], dec):
            count += 1
            # indexだけ赤色で表示
            print("\033[31mindex:", index, "\033[0m")
            print("秘密分散前:", loaded_weights[index][0], loaded_weights[index][1], loaded_weights[index][2],
                  loaded_weights[index][3], loaded_weights[index][4], loaded_weights[index][5],
                  loaded_weights[index][6], loaded_weights[index][7], loaded_weights[index][8],
                  loaded_weights[index][9])
            print("秘密分散後:", dec[0], dec[1], dec[2], dec[3], dec[4], dec[5], dec[6], dec[7], dec[8], dec[9])
    return count
