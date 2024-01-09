import util
import shamir

K = 2
N = 3
p = pow(2, 61) - 1

def test_dec(loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3, P):
    for index in range(len(loaded_weights)):
        dec = shamir.array_decrypt23(loaded_weights1[index], loaded_weights2[index], P)
        if not util.compare_arrays(loaded_weights[index], dec):
            print("index:", index)
            print("weights[index]:", loaded_weights[index])
            print("dec_weight:", dec)
            exit()
    print("pass!")


# weightを秘密分散する
def main():
    # weightをロード
    weights = util.load_weights("weights.pkl")

    # # weightを秘密分散するために、util.Accuracy_weight倍する
    # newWeights = []
    # for weight_row in weights:
    #     newWeights_row = []
    #     for weight in weight_row:
    #         newWeights_row.append(int(weight * util.Accuracy_weight))
    #     newWeights.append(newWeights_row)

    # weightを秘密分散
    weights_shares1 = []
    weights_shares2 = []
    weights_shares3 = []
    for i in range(len(weights)):
        shares = shamir.array_encrypt(weights[i], K, N, P)
        weights_shares1_row = []
        weights_shares2_row = []
        weights_shares3_row = []
        for j in range(len(shares)):
            weights_shares1_row.append(shares[j][0])
            weights_shares2_row.append(shares[j][1])
            weights_shares3_row.append(shares[j][2])
        weights_shares1.append(weights_shares1_row)
        weights_shares2.append(weights_shares2_row)
        weights_shares3.append(weights_shares3_row)

    test_dec(weights, weights_shares1, weights_shares2, weights_shares3, P)

    # 秘密分散したweightを保存
    util.save_weights(weights_shares1, "weights1.pkl", "convert_weight.py")
    util.save_weights(weights_shares2, "weights2.pkl", "convert_weight.py")
    util.save_weights(weights_shares3, "weights3.pkl", "convert_weight.py")


if __name__ == '__main__':
    main()
