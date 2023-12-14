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
            # 順伝播の計算
            z = np.dot(x, weights)
            a = relu(z)

            # 誤差の計算
            dz = a - y
            dw = np.outer(x, dz)

            # 重みの更新
            weights = (weights - learning_rate * dw).astype(np.int64)

            # -------------------------------------------

            # dzを秘密分散
            dz1 = []
            dz2 = []
            dz3 = []
            for i in range(len(dz)):
                # learning_rate倍して四捨五入
                dz_tmp = np.round(dz[i] * learning_rate)
                shares = shamir.encrypt(int(dz_tmp), K, N, P)
                dz1.append(shares[0])
                dz2.append(shares[1])
                dz3.append(shares[2])

            # xを秘密分散
            x1 = []
            x2 = []
            x3 = []
            for i in range(len(x)):
                shares = shamir.encrypt(int(x[i]), K, N, P)
                x1.append(shares[0])
                x2.append(shares[1])
                x3.append(shares[2])

            # dwを計算
            dw1 = np.outer(x1, dz1)
            dw2 = np.outer(x2, dz2)
            dw3 = np.outer(x3, dz3)

            # 重みとバイアスの更新
            weights1 = (weights1 - dw1).astype(np.int64)
            weights2 = (weights2 - dw2).astype(np.int64)
            weights3 = (weights3 - dw3).astype(np.int64)

            # debug
            # random_index = np.random.randint(0, len(weights))
            # dec_weight = shamir.array_decrypt33(weights1[random_index], weights2[random_index], weights3[random_index], P)
            # print("重み, 秘密分散前:", weights[random_index][0], weights[random_index][1], weights[random_index][2], weights[random_index][3], weights[random_index][4])
            # print("重み, 秘密分散後:", dec_weight[0], dec_weight[1], dec_weight[2], dec_weight[3], dec_weight[4])
            # print("-------")
            # for i in range(len(weights)):
            #     dec_weight = shamir.array_decrypt33(weights1[i], weights2[i], weights3[i], P)
            #     print("重み, 秘密分散前:", weights[i][0], weights[i][1], weights[i][2], weights[i][3], weights[i][4])
            #     print("重み, 秘密分散後:", dec_weight[0], dec_weight[1], dec_weight[2], dec_weight[3], dec_weight[4])
            #     print("-------")

                # if not util.compare_arrays(weights[i], dec_weight):
                #     print("counter:", counter)
                #     print("重み, 秘密分散前:", weights[i][0], weights[i][1], weights[i][2], weights[i][3], weights[i][4])
                #     print("重み, 秘密分散後:", dec_weight[0], dec_weight[1], dec_weight[2], dec_weight[3], dec_weight[4])
                #     print("-------")
                #     exit(1)

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    # 秘密の再分配
    for i in range(len(weights)):
        converted_weight1, converted_weight2, converted_weight3 = shamir.array_convert_shamir(weights1[i], weights2[i], weights3[i], K, N, P)
        weights1[i] = converted_weight1
        weights2[i] = converted_weight2
        weights3[i] = converted_weight3

    return weights, weights1, weights2, weights3


def main():
    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    weights, weights1, weights2, weights3 = train_network(x_train, y_train, epochs=1, learning_rate=0.0001)
    util.save_weights(weights, "weights.pkl", "training.py")
    util.save_weights(weights1, "weights1.pkl", "training.py")
    util.save_weights(weights2, "weights2.pkl", "training.py")
    util.save_weights(weights3, "weights3.pkl", "training.py")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time}[sec]")
