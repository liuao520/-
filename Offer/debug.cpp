#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

int duplicateInArray(vector<int> &nums)
{
    int l = 1, r = nums.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1; //[l,mid][mid+1, r]
        int count = 0;        //统计左半区间的数的个数
        for (auto x : nums)
            cout << mid << " " << l << " ", count += (x >= l && x <= mid);
        if (count > mid - l + 1)
            r = mid;
        else
            l = mid + 1;
    }
    return r;
}

int main()
{

    int x;
    vector<int> q;
    while (cin >> x)
    {
        q.push_back(x);
        if (cin.get() == '\n') //完美
            break;
    }
    cout << duplicateInArray(q) << endl;

    return 0;
}