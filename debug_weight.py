import util
import shamir

P = pow(2, 61) - 1


def main():
    loaded_weights = util.load_weights("weights.pkl")
    loaded_weights1 = util.load_weights("weights1.pkl")
    loaded_weights2 = util.load_weights("weights2.pkl")
    loaded_weights3 = util.load_weights("weights3.pkl")

    util.debug_weight(loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3, P)
    count = util.test_dec(loaded_weights, loaded_weights1, loaded_weights2, loaded_weights3, P)
    print(count)


if __name__ == '__main__':
    main()
