#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

void bubble_sort(vector<int> &q)
{
    // for (int i = q.size(); i > 0; i--)
    //     for (int j = 0; j + 1 <= i; j++)
    //     {
    //         if (q[j] > q[j + 1])
    //             swap(q[j], q[j + 1]);
    //     }
    //标记排序
    for (int i = q.size(); i > 0; i--)
    {
        bool flag = false;
        for (int j = 0; j + 1 < i; j++)
        {
            if (q[j] > q[j + 1])
            {
                swap(q[j], q[j + 1]);
                flag = true;
            }
        }
        if (!flag)
            break;
    }
}

void select_sort(vector<int> &q)
{
    for (int i = 0; i < q.size(); i++)
        for (int j = i + 1; j < q.size(); j++)
            if (q[i] > q[j])
                swap(q[i], q[j]);
}

void insert_sort(vector<int> &q)
{
    for (int i = 1; i < q.size(); i++)
    {
        int t = q[i], j;
        for (j = i - 1; j >= 0; j--)
            if (q[j] > t)
            {
                q[j + 1] = q[j];
            }
            else
                break;
        q[j + 1] = t;
    }
}

//冒泡排序的逆序对
void merge_sort(vector<int> &q, int l, int r)
{
    if (l >= r)
        return;

    // int res = 0;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    static vector<int> w;
    w.clear();

    int i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j])
            w.push_back(q[i++]);
        else
            w.push_back(q[j++]);
    // res += mid + 1 - i;

    while (i <= mid)
        w.push_back(q[i++]);
    while (j <= r)
        w.push_back(q[j++]);

    for (i = l, j = 0; j < w.size(); i++, j++)
        q[i] = w[j];
    // for (i = 0; i < w.size(); i++)
    //     q[i + l] = w[i];
}

void quick_sort(vector<int> &q, int l, int r)
{
    if (l >= r)
        return;

    int i = l - 1, j = r + 1, x = q[(l + r >> 1)];
    while (i < j)
    {
        do
            i++;
        while (q[i] < x);
        do
            j--;
        while (q[j] > x);

        if (i < j)
            swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}
//堆排序  基础操作 大堆
void push_down(vector<int> &heap, int size, int u) //当前下标
{
    int t = u, left = u * 2, right = u * 2 + 1;
    if (left <= size && heap[left] > heap[t])
        t = left;
    if (right <= size && heap[right] > heap[t])
        t = right;

    if (u != t)
    {
        swap(heap[u], heap[t]);
        push_down(heap, size, t);
    }
}
void push_up(vector<int> &heap, int u)
{

    while (u / 2 && heap[u / 2] < heap[u])
    {
        swap(heap[u / 2], heap[u]);
        u /= 2;
    }
}
void insert(vector<int> &heap, int size, int x)
{
    heap[++size] = x;
    push_up(heap, x);
}
void remove_top(vector<int> &heap, int &size)
{
    heap[1] = heap[size];
    size--;
    push_down(heap, size, 1);
}
void heap_sort(vector<int> &q, int n)
{
    int size = n;
    for (int i = 1; i <= n; i++)
        push_up(q, i);
    for (int i = 1; i <= n; i++)
    {
        swap(q[1], q[size]);
        size--;
        push_down(q, size, 1);
    }
}

void counting_sort(vector<int> &q, int n)
{
    vector<int> cnt(101, 0);
    for (int i = 1; i <= n; i++)
        cnt[q[i]]++;

    for (int i = 1, k = 1; i <= 100; i++)
        while (cnt[i])
        {
            q[k++] = i;
            cnt[i]--;
        }
}

//基数桶排序
int get_num(int x, int i)
{
    while (i--)
        x /= 10;
    return x % 10;
}
void radix_sort(vector<int> &q, int n)
{
    vector<vector<int>> cnt(10);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 10; j++)
            cnt[j].clear();

        for (int j = 1; j <= n; j++)
            cnt[get_num(q[j], i)].push_back(q[j]);

        for (int j = 0, k = 1; j < 10; j++)
            for (int x : cnt[j])
                q[k++] = x;
    }
}

int main()
{
    int n;
    vector<int> q;

    cin >> n;
    // for (int i = 0, t; i < n; i++)
    // {
    //     cin >> t;
    //     q.push_back(t);
    // }

    // quick_sort(q, 0, q.size() - 1);

    // for (auto x : q)
    //     cout << x << " ";
    // cout << flush;
    //堆排序 计数排序 循环从1开始
    q.resize(n + 1);
    for (int i = 1; i <= n; i++)
        cin >> q[i];

    radix_sort(q, n);
    for (int i = 1; i <= n; i++)
        cout << q[i] << ' ';
    cout << flush;

    return 0;
}