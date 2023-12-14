import numpy as np
import time
import util
import shamir

P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight


def relu(x):
    return np.maximum(0, x)

# 秘密分散のReLUは、P/2より大きいかどうかで判断する
def relu_shamir(x):
    for i in range(len(x)):
        if x[i] > P / 2:
            return np.zeros(len(x))
    return x



def train_network(x_train, y_train, epochs, learning_rate):
    print("training start")
    print("k:", K)
    print("n:", N)
    print("p:", P)

    input_size = 784
    output_size = 10

    # 重みの初期値を秘密分散
    weights = (np.random.randn(input_size, output_size) + 1000) * np.sqrt(2. / input_size) * Accuracy_weight
    weights = weights.astype('int64')

    weights1 = []
    weights2 = []
    weights3 = []
    for weight_row in weights:
        weights1_row = []
        weights2_row = []
        weights3_row = []
        for weight in weight_row:
            # weight_tmp = np.round(weight * learning_rate)
            shares = shamir.encrypt(int(weight), K, N, P)
            weights1_row.append(shares[0])
            weights2_row.append(shares[1])
            weights3_row.append(shares[2])
        weights1.append(weights1_row)
        weights2.append(weights2_row)
        weights3.append(weights3_row)

    # 学習開始
    for epoch in range(epochs):
        counter = 0
        for x, y in zip(x_train, y_train):
            counter += 1
            if counter % 500 == 0:
                print("now:", counter)

            # xを秘密分散
            x1 = []
            x2 = []
            x3 = []
            for i in range(len(x)):
                shares = shamir.encrypt(int(x[i]), K, N, P)
                x1.append(shares[0])
                x2.append(shares[1])
                x3.append(shares[2])

            # yを秘密分散
            y1 = []
            y2 = []
            y3 = []
            for i in range(len(y)):
                shares = shamir.encrypt(int(y[i]), K, N, P)
                y1.append(shares[0])
                y2.append(shares[1])
                y3.append(shares[2])

            # -------------------------------------------

            # 順伝播の計算
            z = np.dot(x, weights)
            a = relu(z)

            # 誤差の計算
            dz = a - y
            dw = np.outer(x, dz)

            # 重みの更新
            weights = (weights - learning_rate * dw).astype(np.int64)

            # -------------------------------------------

            # zを秘密分散
            z1 = np.dot(x1, weights1)
            z2 = np.dot(x2, weights2)
            z3 = np.dot(x3, weights3)

            # zを再分配
            z1_transformed, z2_transformed, z3_transformed = shamir.array_convert_shamir(z1, z2, z3, K, N, P)

            a1 = relu_shamir(z1_transformed)
            a2 = relu_shamir(z2_transformed)
            a3 = relu_shamir(z3_transformed)

            a1 = np.array(a1, dtype=np.int64)
            a2 = np.array(a2, dtype=np.int64)
            a3 = np.array(a3, dtype=np.int64)

            # dzを計算
            dz1 = a1 - y1
            dz2 = a2 - y2
            dz3 = a3 - y3

            # dwを計算
            dw1 = np.outer(x1, dz1)
            dw2 = np.outer(x2, dz2)
            dw3 = np.outer(x3, dz3)

            # dwを再分配
            dw1_transformed, dw2_transformed, dw3_transformed = shamir.array_convert_shamir_2d(dw1, dw2, dw3, K, N, P)

            # 重みとバイアスの更新
            weights1 = (weights1 - dw1_transformed * learning_rate).astype(np.int64)
            weights2 = (weights2 - dw2_transformed * learning_rate).astype(np.int64)
            weights3 = (weights3 - dw3_transformed * learning_rate).astype(np.int64)

            # debug
            # for i in range(len(weights)):
            #     dec_weight = shamir.array_decrypt23(weights1[i], weights2[i], P)
            #     print("秘密分散前:", weights[i][0], weights[i][1], weights[i][2], weights[i][3], weights[i][4],
            #           weights[i][5], weights[i][6], weights[i][7], weights[i][8], weights[i][9])
            #     print("秘密分散後:", dec_weight[0], dec_weight[1], dec_weight[2], dec_weight[3], dec_weight[4],
            #           dec_weight[5], dec_weight[6], dec_weight[7], dec_weight[8], dec_weight[9])
            #     print("----------------------------")
                # if not util.compare_arrays(weights[i], dec_weight):
                #     print("index:", i)
                #     print("秘密分散前:", weights[i][0], weights[i][1], weights[i][2], weights[i][3], weights[i][4],
                #           weights[i][5], weights[i][6], weights[i][7], weights[i][8], weights[i][9])
                #     print("秘密分散後:", dec_weight[0], dec_weight[1], dec_weight[2], dec_weight[3], dec_weight[4],
                #           dec_weight[5], dec_weight[6], dec_weight[7], dec_weight[8], dec_weight[9])
                #     print("----------------------------")

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    return weights, weights1, weights2, weights3


def main():
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3 = train_network(x_train[:1000], y_train[:1000], epochs=1, learning_rate=0.0001)
    util.save_weights(weights, "weights.pkl", "training.py")
    util.save_weights(weights1, "weights1.pkl", "training.py")
    util.save_weights(weights2, "weights2.pkl", "training.py")
    util.save_weights(weights3, "weights3.pkl", "training.py")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
