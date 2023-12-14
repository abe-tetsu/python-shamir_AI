import numpy as np
import time
import util
import shamir
import pandas as pd


P = pow(2, 62) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight


def relu(x):
    return np.maximum(0, x)


def train_network(x_train, y_train, epochs, learning_rate):
    print("学習数:", len(x_train))
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

            # dwを秘密分散
            dw1 = []
            dw2 = []
            dw3 = []
            for i in range(len(dw)):
                dw1_row = []
                dw2_row = []
                dw3_row = []
                for j in range(len(dw[i])):
                    # dw[i][j] * learning_rateを四捨五入
                    dw_tmp = np.round(dw[i][j] * learning_rate)
                    shares = shamir.encrypt(int(dw_tmp), K, N, P)
                    dw1_row.append(shares[0])
                    dw2_row.append(shares[1])
                    dw3_row.append(shares[2])

                dw1.append(dw1_row)
                dw2.append(dw2_row)
                dw3.append(dw3_row)

            dw1 = np.array(dw1, dtype=np.int64)
            dw2 = np.array(dw2, dtype=np.int64)
            dw3 = np.array(dw3, dtype=np.int64)

            # 重みとバイアスの更新
            weights1 = (weights1 - dw1).astype(np.int64)
            weights2 = (weights2 - dw2).astype(np.int64)
            weights3 = (weights3 - dw3).astype(np.int64)

        print(f"Epoch {epoch + 1}/{epochs}")
    print("training done")

    return weights, weights1, weights2, weights3


def recognition(random_idx, x_test, loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3):
    # 検出用画像データを秘密分散する
    test_image = x_test[random_idx]

    # 画像を秘密分散
    test_image_shares1 = []
    test_image_shares2 = []
    test_image_shares3 = []
    for i in range(len(test_image)):
        shares = shamir.encrypt(test_image[i], K, N, P)
        test_image_shares1.append(shares[0])
        test_image_shares2.append(shares[1])
        test_image_shares3.append(shares[2])

    test_image_shares1 = np.array(test_image_shares1)
    test_image_shares2 = np.array(test_image_shares2)
    test_image_shares3 = np.array(test_image_shares3)

    prediction0 = np.dot(test_image, loaded_weights)
    prediction1 = np.dot(test_image_shares1, loaded_weights1)
    prediction2 = np.dot(test_image_shares2, loaded_weights2)
    prediction3 = np.dot(test_image_shares3, loaded_weights3)

    # overflow対策
    prediction1 = np.array(prediction1, dtype=object)
    prediction2 = np.array(prediction2, dtype=object)
    prediction3 = np.array(prediction3, dtype=object)

    # 予測を復元
    prediction = []
    for i in range(len(prediction0)):
        shares = [prediction1[i], prediction2[i], prediction3[i]]
        prediction.append(shamir.decrypt(shares, P))

    return prediction, prediction0

def main():
    print("training start")
    print("k:", K)
    print("n:", N)
    print("p:", P)

    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    # 学習データ数を1000から10000まで1000刻みで増やす
    data_sizes = range(1000, 9001, 1000)

    # 結果を保存するためのDataFrameを初期化
    results_df = pd.DataFrame(columns=["学習データ数", "学習時間", "秘密分散前の正解率", "秘密分散後の正解率", "一致率"])

    epochs = 1
    learning_rate = 0.001

    for size in data_sizes:
        # 学習の開始時間を記録
        train_start = time.time()
        weights, weights1, weights2, weights3 = train_network(x_train[:size], y_train[:size], epochs, learning_rate)
        train_end = time.time()

        # テストの開始時間を記録
        test_start = time.time()
        correct_count = 0
        correct_count_before_shamir = 0
        correct_count_without_shamir = 0

        for i in range(len(x_test)):
            prediction_shamir, prediction = recognition(i, x_test, weights,weights1, weights2, weights3)
            if np.argmax(prediction_shamir) == np.argmax(y_test[i]):
                correct_count += 1
            if np.argmax(prediction) == np.argmax(y_test[i]):
                correct_count_without_shamir += 1
            if np.argmax(prediction) == np.argmax(prediction_shamir):
                correct_count_before_shamir += 1

        test_end = time.time()

        # 精度を計算
        accuracy = correct_count / len(x_test)
        accuracy_before_shamir = correct_count_before_shamir / len(x_test)
        accuracy_without_shamir = correct_count_without_shamir / len(x_test)

        # 結果をDataFrameに追加
        new_row = pd.DataFrame({
            "学習データ数": [size],
            "学習時間": [train_end - train_start],
            "秘密分散前の正解率": [accuracy_without_shamir * 100],
            "秘密分散後の正解率": [accuracy * 100],
            "一致率": [accuracy_before_shamir * 100]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        print(f"テストデータ数　　　　　　: {len(x_test)}")
        print(f"精度　　　　　　　　　　　: {accuracy * 100:.2f}%")
        print(f"秘密分散前の出力との一致率: {accuracy_before_shamir * 100:.2f}%")
        print(f"秘密分散なしの精度　　　　: {accuracy_without_shamir * 100:.2f}%")
        print(f"テスト時間　　　　　　　　: {test_end - test_start} seconds")
        print(f"データ数: {size}, 学習時間: {train_end - train_start} seconds, テスト時間: {test_end - test_start} seconds")

    # 結果をCSVファイルに保存
    results_df.to_csv("zikken2.csv", index=False)


if __name__ == '__main__':
    main()
