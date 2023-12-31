import time
import numpy as np
import recognition
import util

P = pow(2, 61) - 1
K = 2
N = 3
Accuracy_weight = util.Accuracy_weight
Accuracy_image = util.Accuracy_image


def main():
    test_start = time.time()

    (x_train, y_train), (x_test, y_test) = util.load_data()
    x_train, x_test = util.transform_data(x_train, x_test)

    loaded_weights = util.load_weights("weights.pkl")
    loaded_weights1 = util.load_weights("weights1.pkl")
    loaded_weights2 = util.load_weights("weights2.pkl")
    loaded_weights3 = util.load_weights("weights3.pkl")

    correct_count = 0
    correct_count_before_shamir = 0
    correct_count_without_shamir = 0

    for i in range(len(x_test)):
        prediction_shamir, prediction = recognition.recognition(i, x_test, loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3)
        if np.argmax(prediction_shamir) == np.argmax(y_test[i]):
            correct_count += 1
        if np.argmax(prediction) == np.argmax(y_test[i]):
            correct_count_without_shamir += 1
        if np.argmax(prediction) == np.argmax(prediction_shamir):
            correct_count_before_shamir += 1

    accuracy = correct_count / len(x_test)
    accuracy_before_shamir = correct_count_before_shamir / len(x_test)
    accuracy_without_shamir = correct_count_without_shamir / len(x_test)
    test_end = time.time()
    print(f"テストデータ数　　　　　　: {len(x_test)}")
    print(f"精度　　　　　　　　　　　: {accuracy * 100:.2f}%")
    print(f"秘密分散前の出力との一致率: {accuracy_before_shamir * 100:.2f}%")
    print(f"秘密分散なしの精度　　　　: {accuracy_without_shamir * 100:.2f}%")
    print(f"テスト時間　　　　　　　　: {test_end - test_start} seconds")


if __name__ == '__main__':
    main()
