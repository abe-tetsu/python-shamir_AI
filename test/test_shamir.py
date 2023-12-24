import shamir
import unittest
import random

class TestShamir(unittest.TestCase):
    def test_shamir_encrypt(self):
        # 100回テストを繰り返す
        k = 2
        n = 3
        p = pow(2, 62) - 1
        for i in range(100):
            secret = random.randint(1, 100)
            shares = shamir.encrypt(secret, k, n, p)
            self.assertEqual(len(shares), n)

            dec_secret = shamir.decrypt(shares[:k], p)
            self.assertEqual(dec_secret, secret)


    def test_shamir_decrypt(self):
        secret = 100
        shares = [102, 104]
        p = 3359
        dec_secret = shamir.decrypt(shares, p)
        self.assertEqual(dec_secret, secret)

    def test_shamir23(self):
        secret = random.randint(1, 100)
        k = 2
        n = 3
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir34(self):
        secret = random.randint(1, 100)
        k = 3
        n = 4
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir45(self):
        secret = random.randint(1, 100)
        k = 4
        n = 5
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

    def test_shamir56(self):
        secret = random.randint(1, 100)
        k = 5
        n = 6
        p = 3359
        shares = shamir.encrypt(secret, k, n, p)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        self.assertEqual(dec_secret, secret)

class TestShamirPlus(unittest.TestCase):
    def test_shamir_minus(self):
        k = 2
        n = 3
        p = pow(2, 62) - 1
        secret = -100
        shares = shamir.encrypt(secret, k, n, p)
        print(shares)
        self.assertEqual(len(shares), n)

        dec_secret = shamir.decrypt(shares[:k], p)
        print(dec_secret)
        self.assertEqual(dec_secret, secret)

    def test_minus_multi(self):
        k = 2
        n = 3
        p = pow(2, 62) - 1
        secret1 = -100
        secret2 = -200
        shares1 = shamir.encrypt(secret1, k, n, p)
        shares2 = shamir.encrypt(secret2, k, n, p)

        res1 = shares1[0] * shares2[0]
        res2 = shares1[1] * shares2[1]
        res3 = shares1[2] * shares2[2]

        res = shamir.decrypt([res1, res2, res3], p)
        print(res)
        self.assertEqual(res, secret1 * secret2)

    def test_wari(self):
        k = 2
        n = 3
        p = pow(2, 62) - 1
        weight = 2000000
        accuracy = 100
        shares = shamir.encrypt(weight, k, n, p)
        accuracy_share = shamir.encrypt(accuracy, k, n, p)

        shares[0] = shares[0] // accuracy_share[0]
        shares[1] = shares[1] // accuracy_share[1]
        shares[2] = shares[2] // accuracy_share[2]

        res = shamir.decrypt(shares, p)
        print(res)
        self.assertEqual(res, weight // accuracy)

class TestAddShamir(unittest.TestCase):
    def test_秘密の再分配(self):
        k = 2
        n = 3
        p = pow(2, 62) - 1
        secret1 = 100
        shares1 = shamir.encrypt(secret1, k, n, p)

        secret2 = 2
        shares2 = shamir.encrypt(secret2, k, n, p)

        res1 = shares1[0] * shares2[0]
        res2 = shares1[1] * shares2[1]
        res3 = shares1[2] * shares2[2]
        res = shamir.decrypt([res1, res2, res3], p)
        self.assertEqual(res, secret1 * secret2)

        res = shamir.decrypt([res1, res2], p)
        self.assertNotEqual(res, secret1 * secret2)

        # 再分配
        new_shares = shamir.convert_shamir(res1, res2, res3, k, n, p)
        new_res = shamir.decrypt([new_shares[0], new_shares[1]], p)
        self.assertEqual(new_res, secret1 * secret2)

        new_res = shamir.decrypt([new_shares[0], new_shares[1], new_shares[2]], p)
        self.assertEqual(new_res, secret1 * secret2)

        secret3 = 10
        shares3 = shamir.encrypt(secret3, k, n, p)
        res1 = new_shares[0] * shares3[0]
        res2 = new_shares[1] * shares3[1]
        res3 = new_shares[2] * shares3[2]

        res = shamir.decrypt([res1, res2, res3], p)
        self.assertEqual(res, secret1 * secret2 * secret3)

        res = shamir.decrypt([res1, res2], p)
        self.assertNotEqual(res, secret1 * secret2 * secret3)

    def test_3つのシェアの掛け算の復元テスト(self):
        secret1 = 10
        secret2 = 20
        secret3 = 30
        answer = secret1 * secret2 * secret3

        k = 2
        n = 3
        p = pow(2, 62) - 1

        # secret1とsecret2の掛け算
        shares1 = shamir.encrypt(secret1, k, n, p)
        shares2 = shamir.encrypt(secret2, k, n, p)
        res1 = shares1[0] * shares2[0]
        res2 = shares1[1] * shares2[1]
        res3 = shares1[2] * shares2[2]

        # 復元テスト
        res = shamir.decrypt([res1, res2, res3], p)
        print("use 3 shares: secret1 * secret2:", res)
        self.assertEqual(res, secret1 * secret2)

        res = shamir.decrypt([res1, res2], p)
        print("use 2 shares: secret1 * secret2:", res)
        self.assertNotEqual(res, secret1 * secret2)

        # 再分配
        new_share1, new_share2, new_share3 = shamir.convert_shamir(res1, res2, res3, k, n, p)
        res = shamir.decrypt([new_share1, new_share2, new_share3], p)
        print("use 3 shares: secret1 * secret2:", res)
        self.assertEqual(res, secret1 * secret2)

        res = shamir.decrypt([new_share1, new_share2], p)
        print("use 2 shares: secret1 * secret2:", res)
        self.assertEqual(res, secret1 * secret2)

        # secret3を掛け算する
        shares3 = shamir.encrypt(secret3, k, n, p)
        res1 = new_share1 * shares3[0]
        res2 = new_share2 * shares3[1]
        res3 = new_share3 * shares3[2]

        # 復元テスト
        res = shamir.decrypt([res1, res2, res3], p)
        print("use 3 shares: secret1 * secret2 * secret3:", res)
        self.assertEqual(res, answer)

        res = shamir.decrypt([res1, res2], p)
        print("use 2 shares: secret1 * secret2 * secret3:", res)
        self.assertNotEqual(res, answer)

        # 再分配
        new_share1, new_share2, new_share3 = shamir.convert_shamir(res1, res2, res3, k, n, p)
        res = shamir.decrypt([new_share1, new_share2, new_share3], p)
        print("use 3 shares: secret1 * secret2 * secret3:", res)
        self.assertEqual(res, answer)

        res = shamir.decrypt([new_share1, new_share2], p)
        print("use 2 shares: secret1 * secret2 * secret3:", res)
        self.assertEqual(res, answer)


