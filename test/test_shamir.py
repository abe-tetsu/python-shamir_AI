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
        secret2 = 200
        shares1 = shamir.encrypt(secret1, k, n, p)
        shares2 = shamir.encrypt(secret2, k, n, p)

        res1 = shares1[0] * shares2[0]
        res2 = shares1[1] * shares2[1]
        res3 = shares1[2] * shares2[2]

        res = shamir.decrypt([res1, res2, res3], p)
        print(res)
        self.assertEqual(res, secret1 * secret2)

