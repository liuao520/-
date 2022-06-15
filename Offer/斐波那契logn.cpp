#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int MOD = 1000000007;
//������Ļ�  ��û����ЩӦ�õ�����
void mul(int a[][2], int b[][2], int c[][2])
{
    int temp[][2] = {{0, 0}, {0, 0}};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                long long x = temp[i][j] + (long long)a[i][k] * b[k][j];
                temp[i][j] = x % MOD;
            }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            c[i][j] = temp[i][j]; //, cout << c[0][0] << "c";
}

int fibonacci(long long n)
{
    int x[2] = {1, 1}; //��ʼֵ

    //��A^(n - 1)����
    int res[][2] = {{1, 0}, {0, 1}}; //��λ��
    int t[][2] = {{1, 1}, {1, 0}};
    long long k = n - 1;
    while (k)
    {
        if (k & 1)            //����ǰk��ĩβ��1
            mul(res, t, res); //, cout << res[0][0] << "_";
        mul(t, t, t);
        k >>= 1; //ɾ��k��ĩβ
    }
    //��A^(n - 1)���X1
    int c[2] = {0, 0};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            long long r = c[i] + (long long)x[j] * res[j][i];
            c[i] = r % MOD;
        }
    return c[0];
}

int main()
{
    long long n;
    cin >> n;
    cout << fibonacci(n) << endl;

    return 0;
}