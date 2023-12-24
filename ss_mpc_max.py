import random
p = pow(2, 62) - 1
k = 2
n = 3

def SS_shareGen(s):
    #	v[i] = s + (((a[k-1]x + a[k-2])x + a[k-3])x + a[k-4]....) + a[1])x

    v = [0]

    a = random.randrange(p)
    for i in range(1,n+1):
        v.append((a * i) % p)

    for j in range(k-2):
        a = random.randrange(p)
        for i in range(1,n+1):
            v[i] = ((v[i] + a) * i) % p

    for i in range(1,n+1):
        v[i] = (v[i] + s) % p

    return v

# Lagrange coefficient
def L(i):
    a = 1
    for j in range(1,n+1):
        if (j != i):
            a = (a * (0 - j) * pow(i - j, -1, p)) % p
    if a < 0:
        a = a + p
    return a

# reconstruct a secret from n shares
def SS_reconst(v):
    s = 0
    for i in range(1,n+1):
        s = (s + v[i] * L(i)) % p
    if s < 0:
        s = s + p
    return s

# convert shares of (2k-1,n)-SS into shares of (k,n)-SS
def SS_conv(v):
    vconv = [0] * (n+1)
    for i in range(1,n+1):
        vshare = SS_shareGen(v[i])
        for j in range(1,n+1):
            vconv[j] = (vconv[j] + vshare[j] * L(i)) % p
        del vshare
    return vconv

# compute share of s1 + s2 from shares of s1 and s2
def MPC_add(v1,v2):
    v = [0]
    for i in range(1,n+1):
        v.append((v1[i] + v2[i]) % p)
    return v

# compute share of c * s from shares of s and integer c
def MPC_scalar(c,v):
    vsmul = [0]
    for i in range(1,n+1):
        vsmul.append((c * v[i]) % p)
    return vsmul

# compute share of s1 * s2 from shares of s1 and s2
def MPC_mul(v1,v2):
    v = [0]
    for i in range(1,n+1):
        v.append((v1[i] * v2[i]) % p)
    vconv = SS_conv(v)
    return vconv

def all(s1,s2,s3,s4,s5):
    if s1 + s2 + s3 + s4 + s5 == 5:
        return 1
    else:
        return 0

def majority(s1,s2,s3,s4,s5):
    if (s1 + s2 + s3 + s4 + s5 >= 3):
        return 1
    else:
        return 0

def avarage(s1,s2,s3,s4,s5):
    return (s1 + s2 + s3 + s4 + s5) // 5

def max(s):
    max = 0
    for i in range(1,n):
        if s[max] < s[i]:
            max = i
    return max

def test():
    domainSize = 11
    while True:

        s = []

        for i in range(n):
            print('{}. '.format(i), end='')
            s.append(int(input('Input Point: ')))

        v = []
        vresult = SS_shareGen(0)

        remain = pow(domainSize, n) - 1

        for i in range(n):
            w = []
            for j in range(domainSize):
                if s[i] == j:
                    w.append(SS_shareGen(1))
                else:
                    w.append(SS_shareGen(0))
            v.append(w)

        print(v)
        ONE = SS_shareGen(1)

        for s0 in range(domainSize):
            vmul0 = MPC_mul(ONE, v[0][s0])
            for s1 in range(domainSize):
                vmul1 = MPC_mul(vmul0, v[1][s1])
                for s2 in range(domainSize):
                    vmul2 = MPC_mul(vmul1, v[2][s2])
                    for s3 in range(domainSize):
                        vmul3 = MPC_mul(vmul2, v[3][s3])
                        for s4 in range(domainSize):
                            vmul4 = MPC_mul(vmul3, v[4][s4])
                            vmul = MPC_scalar(max([s0,s1,s2,s3,s4]), vmul4)
                            vresult = MPC_add(vresult, vmul)
        result = SS_reconst(vresult)

        print('Max: {}'.format(result))

def main():
    secret = 100
    shares = SS_shareGen(secret)
    print(shares)

    weight1 = shares[0] // 10
    weight2 = shares[1] // 10
    weight3 = shares[2] // 10

    res = SS_reconst([weight1, weight2, weight3])
    print(res)

if __name__ == '__main__':
    main()