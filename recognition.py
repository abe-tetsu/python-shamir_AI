# 画像認識 ver3
import time

import numpy as np
import matplotlib.pyplot as plt
import shamir
import util

P = pow(2, 61) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight
Accuracy_image = util.Accuracy_image


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

    # debug
    # print("秘密分散前:", prediction0[0], prediction0[1], prediction0[2], prediction0[3], prediction0[4], prediction0[5],
    #       prediction0[6], prediction0[7], prediction0[8], prediction0[9])
    # print("秘密分散後:", prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], prediction[5],
    #       prediction[6], prediction[7], prediction[8], prediction[9])

    return prediction, prediction0


def main():
    (x_train, _), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)
    loaded_weights = util.load_weights("weights.pkl")
    loaded_weights1 = util.load_weights("weights1.pkl")
    loaded_weights2 = util.load_weights("weights2.pkl")
    loaded_weights3 = util.load_weights("weights3.pkl")

    # テストデータからランダムなインデックスを選択
    random_idx = np.random.randint(0, len(x_test))

    # 予測を実行
    prediction, prediction0 = recognition(random_idx, x_test, loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3)

    print(f"正解　　　　　　　　: {np.argmax(y_test[random_idx])}")
    print(f"予測結果(秘密分散前): {np.argmax(prediction0)}")
    print(f"予測結果(秘密分散後): {np.argmax(prediction)}")


    # 画像を表示
    # plt.imshow(x_test[random_idx].reshape(28, 28), cmap="gray")
    # plt.title(
    #     f"Before Shamir: {util.array_max(prediction0)}, After Shamir: {util.array_max(prediction)}, Actual: {np.argmax(y_test[random_idx])}")
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    main()
