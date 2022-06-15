

# 剑指offer的题解

## STL

```c++
vector,变长数组
    size()  返回元素个数 
    empty() 返回是否为空 
    clear() 清空 
    front()/back() 返回头/尾元素 
    push_back()/pop_back()从尾部插入元素/删除尾部元素 
    begin()/end() 返回头部的下标/返回尾部的下标的+1

        pair(int,int)
            first,第一个元素
            second,第二个元素


----------


string,字符串
    size()  返回元素个数 
    empty() 返回是否为空 
    clear() 清空
    substr(a,b) 返回从下标为a开始长度为b的字符串，b如果不写或者b大于a后面元素的长度，就返回下标a后面的所有元素



----------



queue,队列
    size()  返回元素个数 
    empty() 返回是否为空
    push() 向队尾插入一个元素
    front() 返回队头元素
    back() 返回队尾元素
    pop() 弹出队头元素
    清空：q=queue<int>();



----------



priority_queue 优先队列,堆，默认是大根堆
    push() 插入一个元素
    top() 返回栈顶元素
    pop() 弹出栈顶元素
    定义成小根堆的方式：priority_queue<int,vector<int>,greater<int>> q; 



----------



stack,栈
    size()  返回元素个数  
    empty() 返回是否为空
    push() 向栈顶 插入一个元素
    top() 返回栈顶元素
    pop() 弹出栈顶元素



----------



deque,双端队列
    size()  返回元素个数 
    empty() 返回是否为空 
    clear() 清空
    front() 返回第一个元素
    back() 返回最后一个元素
    push_back() 向最后插入一个元素
    pop_back() 弹出最后一个元素
    push_front() 像向队头插入一个元素
    pop_front() 弹出第一个元素
    begin()/end() 返回头部的下标/返回尾部的下标的+1



----------



set,multiset,map,multimap, 基于平衡二叉树（红黑树），动态维护有有序序列
    size()  返回元素个数 
    empty() 返回是否为空
    clear() 清空
    begin()/end() 返回头部的下标/返回尾部的下标的

    set/multiset
        insert() 插入一个数
        find() 查找一个数
        count() 返回某一个数的个数
        erase()
            输入的是一个数，删除所有的x
            输入的是一个迭代器，删除这个迭代器
        lower_bound()/upper_bound()
            lower_bound() 返回大于等于x的最小的迭代器 
            upper_bound() 返回大于x的最小的迭代器

    map/multimap
        insert() 插入一个数(pair)
        erase() 
            输入的是一个数(pair)，删除所有的x(pair)
            输入的是一个迭代器，删除这个迭代器



----------



unordered_set,unordered_multiset,unordered_map,unordered_multimap   哈希表
    和上面类似
    不支持lower_bound()/upper_bound(),迭代器的++，--



----------



bitset,压位
    bitset<10000> s; 10000为个数
    支持位运算：~ , & , | , ^ , >> , << , == , != 
    count() 返回有多少个1
    any() 判断是否至少有一个1
    none() 判断是否全为0
    set(),把所有位置成1
        set(k,v) 将第k位变成v
    reset() 把所有位变成0
    filp() 等价于~
        filp(k) 把第k位取反 

作者：有马公生
链接：https://www.acwing.com/blog/content/21593/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 二分

```c++
一、查找精确值
从一个有序数组中找到一个符合要求的精确值（如猜数游戏）。如查找值为Key的元素下标，不存在返回-1。

//这里是left<=right。
//考虑这种情况：如果最后剩下A[i]和A[i+1]（这也是最容易导致导致死循环的情况)首先mid = i,
//如果A[mid] < key，那么left = mid+1 = i +1，如果是小于号，则A[i + 1]不会被检查，导致错误
int left = 1,right = n;
while(left <= right)
{
   //这里left和right代表的是数组下标，所有没有必要改写成mid = left + (right - left)/2;
  //因为当代表数组下标的时候，在数值越界之前，内存可能就已经越界了
  //如果left和right代表的是一个整数，就有必要使用后面一种写法防止整数越界
        int mid = (left + right) / 2;
    if(A[mid] == key)
      return mid;
    else if(A[mid] > key)//这里因为mid不可能是答案了，所以搜索范围都需要将mid排除
      right = mid - 1;
    else
      left = mid + 1;
}
return -1;
二、查找大于等于/大于key的第一个元素
这种通常题目描述为满足某种情况的最小的元素。

int left = 1,right = n;
while(left < right)
{
  //这里不需要加1。我们考虑如下的情况，最后只剩下A[i],A[i + 1]。
  //首先mid = i，如果A[mid] > key，那么right = left = i，跳出循环，如果A[mid] < key，left = right = i + 1跳出循环，所有不会死循环。
  int mid = (left + right) / 2;
  if(A[mid] > key)//如果要求大于等于可以加上等于，也可以是check(A[mid])
    right = mid;
  //因为找的是大于key的第一个元素，那么比A[mid]大的元素肯定不是第一个大于key的元素，因为A[mid]已经大于key了，所以把mid+1到后面的排除
  else
    left = mid + 1;
  //如果A[mid]小于key的话，那么A[mid]以及比A[mid]小的数都需要排除，因为他们都小于key。不可能是第一个大于等于key的元素，
}
三、查找小于等于/小于key的最后一个元素
这种通常题目描述为满足某种情况的最大的元素。如Leetcode69题，求sqrt(x)向下取整就是这种模板。

int left = 1, right = n;
while(left < right)
{
  //这里mid = (left + right + 1) / 2;
  //考虑如下一种情况，最后只剩下A[i],A[i + 1]，如果不加1，那么mid = i，如果A[mid] < key，执行更新操作后，left = mid，right = mid + 1，就会是死循环。
  //加上1后，mid = i + 1,如果A[mid] < key，那么left = right = mid + 1,跳出循环。如果A[mid] > key，left = mid = i，跳出循环。
  int mid = (left + right + 1) / 2;
  if(A[mid] < key)
    left = mid;//如果A[mid]小于key，说明比A[mid]更小的数肯定不是小于key的最大的元素了，所以要排除mid之前的所有元素
  else
    right = mid - 1;//如果A[mid]大于key，那么说明A[mid]以及比A[mid]还要大的数都不可能小于key，所以排除A[mid]及其之后的元素。
}
四、总结
最后两种情况的循环跳出条件是left<right，为什么不是小于等于呢？因为我们的区间变换思路是不断的舍去不可能是解的区间，最后只剩下一个数就是我们的解。而第一种情况就算最后只剩一个数也有可能不是解，所以需要使用小于等于。

查找精确值，循环条件是小于等于；查找满足情况的最大最小值，循环条件是小于。
查找满足条件的最大数，mid = (right + left + 1) / 2；查找满足条件的最小数，mid = (right + left)/2
mid = left + (right - left) / 2，不是适用于所有的情况。
如果存在没有解的情况，比如从[1,2,3,4,5]找出大于等于6的第一个数，我们只需要将最后剩下的数单独进行一次判断就可以了。 
```



### [13. 找出数组中重复的数字](https://www.acwing.com/problem/content/14/)

#### 思路：

把每个数放到对应的位置上，即让 `nums[i] = i`。（每个数都要在1~n-1范围内）

从前往后遍历数组，对于当前遍历到的数，如果坑和值不等，把值交换到正确的位置上，之后交换到了对应位置上，判断是否有`nums[i]！= i`，说明出现多次。

#### 题解：

```C++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int n = nums.size();
        for(auto x : nums)
            if(x < 0 || x >= n)
                return -1;
        
        for(int i = 0; i <n; i++){
            while(nums[i] != i  && nums[nums[i]] != nums[i]) swap(nums[nums[i]],nums[i]);//已经占了一轮坑
            if(nums[i] != i ) return nums[i];
        }
        return -1;
    }
};
```

### [14. 不修改数组找出重复的数字](https://www.acwing.com/problem/content/15/)

#### 思路：

##### 二分思路：

二分本质是二段性：

```
先写一个check函数
判定在check的情况下（true和false的情况下），如何更新区间。
在check(m)==true的分支下是:
	其中确定mid在哪里  然后要得到的条件的位置在哪里
范围[1,mid][mid+1,r] 在右边区间找到mid，在[l,m]找答案 r=mid的情况，中间点的更新方式是m=(l+r)/2 
范围[1,mid-1][mid,r] 在左边区间找到mid，在[m,r]找答案l=mid的情况，中间点的更新方式是m=(l+r+1)/2 
这种方法保证了：
1. 最后的l==r
2. 搜索到达的答案是闭区间的，即a[l]是满足check()条件的。
```

1：分支和抽屉原理

每次会将区间长度缩小一半，一共会缩小 O(logn)次。每次统计两个子区间中的数时需要遍历整个数组，时间复杂度是 O(n)。所以总时间复杂度是 O(nlogn)

分支划分数的取值范围左右两个区间，一定至少存在一个区间，区间中数的个数大于区间长度。依次类推，每次可以把区间长度缩小一半，直到区间长度为1时，即题解。

```C++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int l = 1, r = nums.size()-1;
        while(l < r){
            int mid = l + r >> 1;//[l,mid][mid+1, r]//区间是指 数的取值范围
            int count =0;//统计数组中位于左半区间的数的个数
            for(auto x: nums) count += ( x >= l && x <= mid);
            if(count >  mid - l + 1) r = mid;//区间中数的个数大于区间长度
            else l = mid + 1;
        }  
        return l;
    }
};
```

2：双指针思路  

慢指针每次走一格，刚好遍历到链表尾部（即环起点）处结束，因此复杂度为O(n)O(n)
空间复杂度分析：O(1)

1. 如何判断链表是否存在环？
双指针，一快（每次跑两格）一慢（每次跑一格），从链表首部开始遍历，两个指针最终都会进入环内，由于快指针每次比慢指针多走一格，因此快指针一定能在环内追上慢指针。而如果链表没环，那么快慢指针不会相遇。
2. 对于有环的链表，如何找到环的起点？
基于第一点，快慢指针相遇时，我们可以证明相遇的点与环起点的距离，一定和链表首部与环起点的距离相等

```c++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        //0一定是一个链表的首部，因为所有元素值的范围在1 - n-1之间，即没有节点指向0节点。
        int f = 0, s = 0;
        //元素的下标代表节点地址，元素的值代表next指针
        //重复的元素(任意一个环的起点)意味着两个节点的next指针一样，即指向同一个节点
        //因此存在环，且环的起点即重复的元素。
        while(f == 0 || f != s){
            cout <<f << " "<< s<<endl;
            f = nums[nums[f]];
            s = nums[s];
            cout <<f << "_"<< s<<endl;
        }
        f = 0;
        while(f != s){
            cout <<endl<<f << " "<< s<<endl;
            f = nums[f];
            s = nums[s];
            cout <<f << "_"<< s<<endl;
        }
        return s;
    }
};
```

### [15. 二维数组中的查找](https://www.acwing.com/problem/content/16/)

#### 思路：

暴力

从第1行第n（n=array.size();）列开始搜索（数组的规律）
如果要找的数比当前的数小，那么我们找下一行的数，列不变
如果要找的数比当前的数大，那么我们找前一列的数，行不变

#### 题解：

```c++
class Solution {
public:
    bool searchArray(vector<vector<int>> array, int target) {
        if(array.empty()) return false;
        int i = 0, j = array[0].size() - 1;
        while(i < array.size() && j >= 0){
            if(array[i][j] == target) return true;
            if(array[i][j] > target) j --;
            else i++;
        }
        return false;
    }
};
```

### 4：替换空格

```C++
class Solution {
public:
    string replaceSpaces(string &str) {
    string res;
    for(auto x : str){
        if(x == ' ')
            res += "%20";
        else res += x;
    }
    return res;
    }
};
```

### 5：从尾到头打印链表

```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while(head){
            res.push_back(head->val);
            head = head->next;
        }
    // reverse(res.begin(), res.end());
    return   vector<int>(res.rbegin(), res.rend());
    }
};
```

### [18. 重建二叉树](https://www.acwing.com/problem/content/23/)

#### 思路：

前序遍历数组的第一个就是根节点，其余分别是左子树和右子树；中序遍历数组则是左子树+根节点+右子树。所以我们首先从前序遍历数组中找到根节点（第一个），然后在中序遍历中找到根节点，那么中序遍历数组就被分出来了，分成[左子树 | 根节点 | 右子树]。那么我们就可以每次找到左子树的节点个数，将中序数组分成左右子树进行递归，既可以构造二叉树

#### 题解：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //记录每个值在中序遍历中的位置，这样我们在递归到每个节点时，在中序遍历中查找根节点位置的操作
    unordered_map<int, int> pos;
    
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        for(int i = 0; i < n; i++)
            pos[inorder[i]] = i;
        return dfs(preorder, 0 , n - 1, inorder , 0 , n - 1);
    }
    
    TreeNode *dfs(vector<int> &preorder, int pl, int pr, vector<int>&inorder, int il, int ir){
        if(pl > pr) return nullptr;
        TreeNode* root = new TreeNode(preorder[pl]);//前序遍历数组的第一个就是根节点
        int k = pos[root->val] - il;//根的左子树的节点数
        root -> left = dfs(preorder, pl + 1, pl + 1 + k - 1, inorder, il, il + k - 1);
        root -> right = dfs(preorder, pl + 1 + k -1 + 1, pr, inorder, il + k + 1, ir);
        return root;
    }
};
```

### [19. 二叉树的下一个节点](https://www.acwing.com/problem/content/31/)

#### 思路：

如果当前节点有右儿子，则右子树中最左侧的节点就是当前节点的后继。 
如果当前节点没有右儿子，则需要沿着father域一直向上找，找到第一个是其father左儿子的节点，该节点的father就是当前节点的后继。 

#### 题解：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode *father;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL), father(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* inorderSuccessor(TreeNode* p) {
        //如果该节点有右子树，那么下一个节点就是其右子树中最左边的节点
        if(p -> right) {
            p = p -> right;
            while(p -> left) p = p -> left;
            return p;
        }
        //节点不存在右子树且节点是其父节点的左节点 直接就是父节点
        //节点不存在右子树且节点是其父节点的右节点 沿着父节点找第一个有左儿子的father节点（后继节点）
        while(p -> father && p == p -> father -> right) p = p -> father;//节点不存在右子树且节点是其父节点的右节点
        return p -> father;
        //节点不存在右子树且节点是其父节点的左节点
        //else if(p -> right == nullptr && p->father &&  p == p -> father -> left) return p ->father;
        //else {
            //节点不存在右子树且节点是其父节点的右节点
            //while(p -> father && p == p -> father -> right) p = p -> father;
            //return p ->father;
        //}
    }
};
```

### [20. 用两个栈实现队列](https://www.acwing.com/problem/content/36/)

#### 思路：

暴力

#### 题解：

```c++
class MyQueue {
public:

    stack<int> stk, cache;
    /** Initialize your data structure here. */
    MyQueue() {
    }
    /** Push element x to the back of queue. */
    void push(int x) {
        stk.push(x);    
    }
    //将一个栈弹出并压入另个栈中
    void copy(stack<int> &a, stack<int> &b){
        while(a.size()){
            b.push(a.top());
            a.pop();
        }
    }
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        copy(stk, cache);
        int res = cache.top();
        cache.pop();
        copy(cache, stk);
        return res;
    }
    /** Get the front element. */
    int peek() {
        copy(stk, cache);
        int res = cache.top();
        copy(cache, stk);
        return res;
    }
    /** Returns whether the queue is empty. */
    bool empty() {
        return stk.empty();
    }
};
/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */
```

### 7：斐波那契数列

为了避免考虑整数溢出问题，我们求 $a_n\%p$ 的值，$p=10^9+7p$。

#### 1.递归 

时间复杂度是$O(2^n)$

```c++
class Solution {
public:
    const int N = 100000;
    int Fibonacci(int n) {
        if(n == 0) return 0;
        if(n == 1) return 1;
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }
};
```

#### 2.记忆化搜索

开一个大数组记录中间结果，如果一个状态被计算过，则直接查表，否则再递归计算。
总共有 n 个状态，计算每个状态的复杂度是 O(1)，所以时间复杂度是 O(n)。
一秒内算 $n=10^7$毫无压力，但由于是递归计算，递归层数太多会爆栈，大约只能算到 $n=10^5$级别。

```c++
onst int N = 100000;
int a[N];
int Fibonacci(int n)
{
    if (a[n]) return a[n];
    if(n == 0) return 0;
    if(n == 1) return 1;
    a[n] = Fibonacci(n - 1) + Fibonacci(n - 2);
    return a[n];
}
```

#### 3.递推

开一个大数组，记录每个数的值。用循环递推计算。

时间复杂度是 O(n)。

```c++
int Fibonacci(int n) {
	int a[N];
	dp[0] = 0,dp[1] = 1;
    for(int i=2;i<=n;i++) 
    	dp[i] = dp[i-1] + dp[i-2];
    return a[n];
    }
```

#### 4.递归+滚动变量

递推时只需要记录前两项的值即可，没有必要记录所有值，所以可以用滚动变量递推。

时间复杂度还是 O(n)，但空间复杂度变成了 O(1)。

```c++
class Solution {
public:
    int Fibonacci(int n) {
        int a = 0, b = 1;
        while (n -- ) {
            int c = a + b;
            a = b, b = c;
        }
        return a;
    }
};
```

#### 5.矩阵运算 + 快速幂

求$m^k\%p$,时间复杂度$O(logk)$

```c++
int qmi(int m, int k, int p)
{
    int res = 1 % p, t = m;
    while (k)
    {
        if (k&1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}
```

时间复杂度$O(logk)$

利用矩阵运算的性质将通项公式变成幂次形式，然后用平方倍增（快速幂）的方法求解第 n 项

递推+滚动变量 1秒内最多可以算到 $10^8$ 级别,而矩阵运算+快速幂 可以n 在 long long 范围内都可以在1s内算出来。

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int MOD = 1000000007;
//传数组的话  就没有那些应用的事了
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
    int x[2] = {1, 1}; //初始值

    //求A^(n - 1)次幂
    int res[][2] = {{1, 0}, {0, 1}}; //单位阵
    int t[][2] = {{1, 1}, {1, 0}};
    long long k = n - 1;
    while (k)
    {
        if (k & 1)            //若当前k的末尾是1
            mul(res, t, res); //, cout << res[0][0] << "_";
        mul(t, t, t);
        k >>= 1; //删掉k的末尾
    }
    //对A^(n - 1)左乘X1
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
```

### 22:[旋转数组的最小数字](https://www.acwing.com/problem/content/20/)


#### 思路：

1.二分查找（二分段的本质）

![image-20220608120328049](C:\Users\hider\AppData\Roaming\Typora\typora-user-images\image-20220608120328049.png)

除了最后水平的一段（黑色水平那段）之外，其余部分满足二分性质：竖直虚线左边的数满足 nums[i]≥nums[0]；而竖直虚线右边的数不满足这个条件。
分界点就是整个数组的最小值

数组完全单调情况：当我们删除最后水平的一段之后，如果剩下的最后一个数大于等于第一个数，则说明数组完全单调。

二分的时间复杂度是 O(logn)，删除最后水平一段的时间复杂度最坏是 O(n)

#### 题解：

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = nums.size() - 1;
        if(n < 0) return -1;
        while(nums[n] == nums[0]) n--;
        if(nums[n] >= nums[0]) return nums[0];//排除只有单调增的情况
        int l = 0, r = n;
        while(l < r){
            int mid = l + r >>1;
            if(nums[mid] < nums[0])//说明mid在右半段，最小值在【left mid】之间
                r = mid;
            else l = mid + 1;
        }
        return nums[l];
    }
};
```

2.优化的暴力

分界点左边的数肯定比右边的数小解题，否则就是全相同或者完全单调 那最小元素为nums[0]

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        if(nums.empty()) return -1;
        int lowest = nums[0];
        for(int i = 1; i < nums.size(); i ++){
            if(nums[i - 1] > nums[i]){
                lowest = nums[i];
                break;
            }
        }
        return lowest;
    }
};
```

### 9:[矩阵中的路径](https://www.acwing.com/problem/content/21/)

#### 思路：

典型的暴搜例子  枚举路径
我们按照搜索的模板来定义

1 寻找起始点 （第一个符合条件的字母） ,添加一个同样大小的多维数组 维护该点是否访问过
2 深度搜索（四个方向扩展）
3 设置正确的退出点 不符合条件 坐标越界 字符串长度越界等
4 继续下一轮搜索 记得将访问数组中对应的点恢复原样

#### 题解：

```c++
class Solution {
public:
//枚举路径
    bool hasPath(vector<vector<char>>& matrix, string &str) {
        for(int i = 0; i < matrix.size(); i++)
            for(int j = 0; j < matrix[i].size(); j++)
                if(dfs(matrix, str, 0, i, j))
                    return true;
        return false;
    }
    
    bool dfs(vector<vector<char>> &matrix, string& str, int u, int x, int y){
        if(matrix[x][y] != str[u]) return false;//该点字符与字符串不同 直接返回
        if(u == str.size() - 1) return true;//检测是否完全符合str字符串
        if (matrix[x][y] == '*') return false;  //该坐标已经访问 直接返回
        
        int dx[4] ={-1, 0, 1, 0}, dy[4] = {0, 1, 0 ,-1};
        char t = matrix[x][y];
        matrix[x][y] = '*';
        //四个方向的遍历 任意一个符合条件则返回成功
        for(int i = 0; i < 4; i++){
            int a = x + dx[i], b = y + dy[i];
            if(a >= 0 && a < matrix.size() && b >= 0 && b < matrix[a].size())
                if(dfs(matrix,str,u + 1, a , b))
                    return true;
        }
        matrix[x][y] = t;
        return false;
    }
    
};
```

### 10.[机器人的运动范围](https://www.acwing.com/problem/content/22/)

#### 思路：

从 (0, 0) 点开始，每次朝上下左右四个方向扩展新的节点。

```markdown
扩展时需要注意新的节点需要满足如下条件：
	之前没有遍历过，这个可以用个bool数组来判断；
	没有走出边界；
	横纵坐标的各位数字之和小于 k；
最后答案就是所有遍历过的合法的节点个数。
```

#### 题解

```c++
class Solution {
public:
    bool get_sum(pair<int,int> p,int threshold){
        int s = 0;
        while(p.first) s += p.first % 10, p.first /= 10;
        while(p.second) s+= p.second %10, p.second /= 10;
        return s <= threshold;
    }
    int movingCount(int threshold, int rows, int cols)
    {
        if(rows == 0 || cols == 0) return 0;
        vector<vector<bool>> st(rows,vector<bool>(cols,false));
        queue<pair<int,int>> q;
        
        q.push({0,0});
        st[0][0] = true;
        
        int res = 0;
        int dx[4] = {-1, 0 ,1, 0}, dy[4] = {0, 1, 0 , -1};
        
        while(q.size()){
            auto t = q.front();
            q.pop();
            res ++;
            for(int i =0; i < 4; i++){
                int a = dx[i] + t.first, b = dy[i] + t.second;
                if(a >= 0 &&a< rows && b>= 0 && b< cols && get_sum(t,threshold) && !st[a][b]){
                    q.push({a,b});
                    st[a][b] = true;
                }
            }
        }
        return res;
    }
};
```

### 11.[剪绳子](https://www.acwing.com/problem/content/24/)

#### 思路

$N = \sum n_i,求y=max(\prod n_i)，y_{max} = x^{N/x}->x^{1/x}=e^{lnx/x}  取极值最大值，x=3$ 

分类讨论

其中2是一个点  当length=4时 ！

#### 题解

```c++
class Solution {
public:
    int maxProductAfterCutting(int length) {
        if(length <= 3) return length - 1;
        int res = 1;
        if(length % 3 ==1) res = 4, length -= 4;
        else if(length % 3 ==2) res = 2, length -= 2;
        
        while(length) {
            res *= 3;
            length -= 3;
        }
        return res;
    }
};
```

### 12：[二进制中1的个数](https://www.acwing.com/problem/content/25/)

#### 思路：

```c++
求 n的第 k位数字: n >> k & 1
返回 n的最后一位1: lowbit(n) = n & -n
```

1.迭代

```markdown
如果 n 在二进制表示下末尾是1，则在答案中加1；
将 n 右移一位，也就是将 n 在二进制表示下的最后一位删掉
```

在C++中如果我们右移一个负整数，系统会自动在最高位补1，这样会导致 n 永远不为0，就死循环了。
解决办法是把 nn 强制转化成无符号整型，这样 n 的二进制表示不会发生改变，但在右移时系统会自动在最高位补0。

```c++
class Solution {
public:
    int NumberOf1(int n) {
        unsigned int un = n;
        int res = 0;
        while(un){
            res += un & 1;
            un >>= 1;
        }
        return res;
    }
};
```

2.n&（n-1）

```markdown
n&(n-1)的结果：n最右边的1变成
比如n为6  110&101->100  循环直到n为0为止
```

```c++
class Solution {
public:
    int NumberOf1(int n) {
        int res =0;
        while(n){
            n = n&(n-1);
            res++;
        }
        return res;
    }
};
```

### 13:[数值的整数次方](https://www.acwing.com/problem/content/26/)

#### 思路

直接快速幂模板计算  把指数按照二进制进行拆解，然后如果当前位是1，乘

#### 题解：

```c++
class Solution {
public:
    typedef long long LL;
    double Power(double base, int exponent) {
        bool is_minus = exponent < 0;
        double res = 1;
        //当exponent等于负无穷时，取相反数会超出int范围。
        LL k = abs(LL(exponent)); //将exponent转成 long long 后 才取绝对值！ 
        while(k){
            if(k & 1) res *= base;
            k >>= 1;
            base *= base;
        }
        if (is_minus) res = 1 / res;
        return res;
    }
};
```

### 14：在O(1)时间删除链表结点 

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void deleteNode(ListNode* node) {
        //node -> val = node -> next -> val;
        //node -> next = node ->next -> next;
        *node = *(node->next);
    }
};
```

### 15：[删除链表中重复的节点](https://www.acwing.com/problem/content/27/)

#### 思路：

为了方便处理边界情况（头结点可能会被删除），我们定义一个虚拟元素 dummy 指向链表头节点。
然后从前往后扫描整个链表，每次扫描元素相同的一段，如果这段中的元素个数多于1个，则将整段元素直接删除。时间复杂度是 O(n) 

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplication(ListNode* head) {
        auto dummy = new ListNode(-1);
        dummy -> next = head;
        
        auto p = dummy;
        while(p -> next){
            auto q = p -> next;
            while(q && p -> next -> val == q -> val) q = q -> next;
            //这段中的元素个数多于1个(重复元素)，则将整段元素直接删除。 否则不重复
            if(p -> next -> next == q) p = p -> next;
            else p -> next = q;
        }
        return dummy->next;
    }
};
```

### 16：[正则表达式匹配](https://www.acwing.com/problem/content/28/)

#### 思路：

#### 1.递归dp

```markdown
状态表示：f[x][y]表示p从y开始到结尾是否能匹配s从x开始到结尾
状态转移：
	1:如果当前的p[y]是正常字母 且与s[x]相等，那么，f[x+1][y+1]是否匹配就依赖于f[x][y]
		f[i][j] = (s[i] == p[j] && f[i + 1][j + 1])
	2:如果当前的p[y]是'.'的话,那么,也与s[x]相等(因为.匹配任何英文字符)，f[x+1][y+1]是否匹配就依赖于 f[x][y]
		f[i][j] = f[i + 1][j + 1]
	(上述的两个情况相似，合并判断)
	
	3：比如b*，判断p[y + 1] 是否是 ‘*’，这里面即表示s[x]的重复0次还是重复多次分为两种情况：
		3.1：如果重复是0次，那么当前的 p[y]和p[y + 1]== *都可以忽略不计(即如果是aa*的话，后面的a*就可以忽略)， 则f[x][y]=f[x][y+2]
		3.2：如果是重复多次,那么当前的p[y]==s[x]或者p[y]=='.'，则f[x][y]=f[x+1][y]。(即无论xx后面有多少个相同字符，都匹配p[y]=='*)
		
```

```c++
class Solution {
public:
    vector<vector<int>>f;
    int n, m;
    bool isMatch(string s, string p) {
        n = s.size();
        m = p.size();
        f = vector<vector<int>>(n + 1, vector<int>(m + 1, -1));//考虑到s,p都为空的情况
        return dp(0, 0, s, p);
    }
    bool dp(int x, int y, string &s, string &p)
    {
//利用到的f[x][y] 有值说明已经计算过，直接返回不需要再去递归。利用到已经计算到的值，截枝不让重复递归
        if (f[x][y] != -1) return f[x][y];
        if (y == m)
            return f[x][y] = x == n;
// x<n 防止了下次进入dp后 "if (f[x][y] != -1)" 的越界; 因为下面进入dp(x+1,,,,)都用first先做了限制
        bool first_match = x < n && (s[x] == p[y] || p[y] == '.');
        if (y + 1 < m && p[y + 1] == '*')
        {// y+2 在进入下一次dp，不会越界;因为进入下一次dp前提是y+1是‘*’,*位于p最后一个位时，数组f可以到最后一位加一
            f[x][y] = dp(x, y + 2, s, p) || (first_match && dp(x + 1, y, s, p));
        }
        else
        // y递增加1，总会遇到的y==m后返回
           f[x][y] = first_match && dp(x + 1, y + 1, s, p);
        return f[x][y];
    }
};

```

#### 2.记忆化搜索dp

```markdown
状态表示:所有满足s[1~i] 与p[1~j]匹配的集合
状态属性:集合是否为空
状态计算：
1 p[j-1] 是字母 而且与 s[i-1] 相等，那么当前dp[i][j]是否匹配就依赖于dp[i-1][j-1]
2 p[j-1] 是'. 那么肯定与s[i-1]相等， 当前dp[i][j]是否匹配 就依赖于 dp[i-1][j-1]
情况1 2 类似 
3 p[j-1] 是'*' ,那么根据s[i]表示的前面字母的多次重复还是0次重复 分为两种情况
3.1 如果是0次重复 那么当前的p[j-1] == ‘*’ 和 p[j-2] 都可以忽略不计。 那么 dp[i][j] = dp[i][j-2]
3.2 如果是多次重复 那么 p[j-2] 与s[i-1] 相等 或者p[j-2]==’.’ 那么dp[i][j] = dp[i-1][j]

边界:第一个串为空，第二个串有可能匹配
第二个串为空，第一个串不可能匹配
```

![1055_046c5f3b63-1](剑指offer.assets/1055_046c5f3b63-1-16462070851323.png)

```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        int n = s.size(), m = p.size();
        s = ' ' + s, p = ' ' + p;
        vector<vector<bool>> f(n + 1, vector<bool>(m + 1));
        f[0][0] = true;
        for (int i = 0; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ )
            {
                if (j + 1 < p.size() && p[j + 1] == '*') continue;
                if (i && p[j] != '*')
                {
                    f[i][j] = f[i - 1][j - 1] && (s[i] == p[j] || p[j] == '.');
                }
                else if (p[j] == '*')
                {
                    f[i][j] = f[i][j - 2] || i && f[i - 1][j] && (s[i] == p[j - 1] || p[j - 1] == '.');
                }
            }
        return f[n][m];
    }
};

```

### 17:[表示数值的字符串](https://www.acwing.com/problem/content/29/)

#### 题解：

#### 常规列举

```c++
class Solution {
public:
    bool isNumber(string s) {
        int i = 0;
        while (i < s.size() && s[i] == ' ') i ++ ;
        int j = s.size() - 1;
        while (j >= 0 && s[j] == ' ') j -- ;
        if (i > j) return false;
        s = s.substr(i, j - i + 1);

        if (s[0] == '-' || s[0] == '+') s = s.substr(1);
        if (s.empty() || s[0] == '.' && s.size() == 1) return false;

        int dot = 0, e = 0;
        for (int i = 0; i < s.size(); i ++ )
        {
            if (s[i] >= '0' && s[i] <= '9');
            else if (s[i] == '.')
            {
                dot ++ ;
                if (e || dot > 1) return false;
            }
            else if (s[i] == 'e' || s[i] == 'E')
            {
                e ++ ;
                if (i + 1 == s.size() || !i || e > 1 || i == 1 && s[0] == '.') return false;
                if (s[i + 1] == '+' || s[i + 1] == '-')
                {
                    if (i + 2 == s.size()) return false;
                    i ++ ;
                }
            }
            else return false;
        }
        return true;
    }
};
```



#### ac自动机

为使用一个指针从前往后逐个检查字符串中的字符是否合法，如果合法，则指针后移，否则指针停止移动，显然如果字符串是合法的，这个指针应该移动到字符串的最末尾

同时在移动指针的过程中判断从字符串开始到当前位置的字符子串是否是合法的数值，并将它存储在isNum中，显然isNum记录了指针所指位置的字符子串是否能表示为数值的信息

```c++
class Solution {
public:
    bool isNumber(string s) {
           
        bool isNum = false; //该变量表示从0开始，到i位置的字符串是否构成合法数字，初始化为false
        int i = 0 , j = s.size() - 1; //检测指针初始化为0
        while(i < s.size() && s[i] == ' ') i ++;//滤除最前面的空格，指针后移
        while(j >= 0 && s[j] == ' ') j --; 
        if (i > j) return false;
        //指针移动过程中，最后会有一位的溢出，加上一位空字符防止字符串下标越界
        s += '\0';//s[j + 1] = '\0';
    
        if(s[i] == '+' || s[i] == '-') ++i; //一个‘-’或‘+’为合法输入，指针后移
        while(s[i] >= '0' && s[i] <= '9'){  //此处如果出现数字，为合法输入，指针后移，同时isNum置为true
            isNum = true;  //显然,在此处，前面的所有字符是可以构成合法数字的
            ++i;
        }
        if(s[i] == '.') ++i;    //按照前面的顺序，在此处出现小数点也是合法的，指针后移（此处能否构成合法字符取决于isNum）
        while(s[i] >= '0' && s[i] <= '9'){  //小数点后出现数字也是合法的，指针后移
            isNum = true;   //无论前面是什么，此处应当是合法数字
            ++i;
        }
        //上面的部分已经把所有只包含小数点和正负号以及数字的情况包括进去了，如果只判断不含E或e的合法数字，到此处就可以停止了
        if(isNum && (s[i] == 'e' || s[i] == 'E')){ //当前面的数字组成一个合法数字时（isNum = true），此处出现e或E也是合法的
            ++i;
            isNum = false; //但到此处，E后面还没有数字，根据isNum的定义，此处的isNum应等于false;

            if(s[i] == '-' || s[i] == '+') ++i; //E或e后面可以出现一个‘-’或‘+’，指针后移

            while(s[i] >= '0' & s[i] <= '9') {
                ++i;
                isNum = true; //E后面接上数字后也就成了合法数字
            }
        }
        //如果字符串为合法数字，指针应当移到最后，即是s[i] == '\0' 同时根据isNum判断数字是否合法
        //整个过程中只有当i位置处的输入合法时，指针才会移动
        return (s[i] == '\0' && isNum);
    }
};
```

### 18：[调整数组顺序使奇数位于偶数前面](https://www.acwing.com/problem/content/30/)

#### 思路

快排思路

#### 题解

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
         int i = 0, j = array.size()-1;
         while(i < j){         
             while (array[i] %2 == 1)i ++;    
             while (array[j] %2 == 0)j --;
             if(i < j) swap(array[i],array[j]);
         }
    }
};
```

###  19：[链表中倒数第k个节点](https://www.acwing.com/problem/content/32/)

#### 思路：

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* findKthToTail(ListNode* pListHead, int k) {
        int count = 0;
        ListNode* p = pListHead;
        //for(auto p = pListHead; p; p = p -> next) count ++;
        while(pListHead != nullptr){
            count ++;
            pListHead = pListHead -> next;
        }
        
        if(k > count) return nullptr;
        
        for(int i = 0; i < count - k; i ++) p = p -> next;
        return p;
    }
};

class Solution {
public:
    ListNode* findKthToTail(ListNode* pListHead, int k) {
        //快慢指针的思想就是两个指针之间距离是K 只要链表存在，相对于fast 最后slow一定是倒数的k
        auto slow = pListHead;
        while(k){
            k --;
            if(pListHead) pListHead = pListHead -> next;
            else return nullptr;
        }
        while(pListHead){
            slow = slow -> next;
            pListHead = pListHead -> next;
        }
        return slow;
    }
};


```

### 20：[链表中环的入口结点](https://www.acwing.com/problem/content/86/)            

#### 思路

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *entryNodeOfLoop(ListNode *head) {
        //if(head == nullptr) return NULL;
        unordered_map<ListNode*, int> unum;
        while(head){
            unum[head] ++;
            if(unum[head] == 2) return head;
            head = head -> next;
        }
        return NULL;
    }
};

class Solution {
public:
    ListNode *entryNodeOfLoop(ListNode *head) {
        
        if(head == nullptr || head -> next == nullptr) return nullptr;
        auto slow = head, fast = head;
        while(fast && fast -> next){
            fast = fast -> next -> next;
            slow = slow -> next;
            if(fast == slow) break;
        }
        
        if(fast == nullptr || fast -> next == nullptr) return nullptr;
        slow = head;
        while(fast != slow){
            fast = fast -> next;
            slow = slow -> next;
        }
        return slow;
    }
};
```

### 21.[反转链表](https://www.acwing.com/problem/content/33/)

#### 思路

![img](剑指offer.assets/RGIF2-16478848653801.gif)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 翻转即将所有节点的next指针指向前驱节点。
由于是单链表，我们在迭代时不能直接找到前驱节点，所以我们需要一个额外的指针保存前驱节点。同时在改变当前节点的next指针前，不要忘记保存它的后继节点。
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *prev = nullptr;
        ListNode *cur = head;
        while (cur)
        {
            auto next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }
};
 
```

```c++
/*
首先我们先考虑 reverseList 函数能做什么，它可以翻转一个链表，并返回新链表的头节点，也就是原链表的尾节点。
所以我们可以先递归处理 reverseList(head->next)，这样我们可以将以head->next为头节点的链表翻转，并得到原链表的尾节点tail，此时head->next是新链表的尾节点，我们令它的next指针指向head，并将head->next指向空即可将整个链表翻转，且新链表的头节点是tail。
*/
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *tail = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return tail;
    }
};
```

### 22.[合并链表](https://www.acwing.com/problem/content/34/)

思路：

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        auto dummy = new ListNode(-1);
        auto cur = dummy;
        while(l1 && l2){
            if(l1->val < l2->val){
                cur -> next = l1;
                cur = l1;
                l1 = l1 -> next;
            }
            else{
                cur -> next = l2;
                cur = l2;
                l2 = l2 -> next;
            }
        }
        if(l1) cur -> next = l1;
        else cur -> next = l2;
        return dummy->next;
    }
};
```

### 23.[树的子结构](https://www.acwing.com/problem/content/35/)

#### 思路：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 * 代码分为两个部分：
    遍历树A中的所有非空节点R；
    判断树A中以R为根节点的子树是不是包含和树B一样的结构，且我们从根节点开始匹配；
对于第一部分，我们直接递归遍历树A即可，遇到非空节点后，就进行第二部分的判断。
对于第二部分，我们同时从根节点开始遍历两棵子树：
    如果树B中的节点为空，则表示当前分支是匹配的，返回true；
    如果树A中的节点为空，但树B中的节点不为空，则说明不匹配，返回false；
    如果两个节点都不为空，但数值不同，则说明不匹配，返回false；
    否则说明当前这个点是匹配的，然后递归判断左子树和右子树是否分别匹配即可；
 */
class Solution {
public:
    bool hasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if(!pRoot1 || !pRoot2) return false;//是否空
        if(isPart(pRoot1, pRoot2)) return true;//开始A树看的都是 根结点，根结点不匹配了再看A的左子树或者A的右子树。
        return hasSubtree(pRoot1 -> left, pRoot2) ||hasSubtree(pRoot1 -> right, pRoot2);//历树A中的所有非空节点R
    }
    bool isPart(TreeNode* p1, TreeNode* p2){//递归判断p2是否都在p1
        if(!p2) return true;//如果树B中的节点为空，则表示当前分支是匹配的，返回true；
        if(!p1) return false;
        if(p1 -> val != p2 -> val) return false;
        //否则、B树的节点不为空， A树的节点也不为空， A树和B树的当前节点是匹配的
        return isPart(p1->left, p2->left) && isPart(p1->right, p2->right);//左右子树是否同时匹配
    }
};
```

### 24.[二叉树的镜像](https://www.acwing.com/problem/content/37/)

#### 思路：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 * 镜像后的树就是将原树的所有节点的左右儿子互换！
所以我们递归遍历原树的所有节点，将每个节点的左右儿子互换即可。
 */
class Solution {
public:
    void mirror(TreeNode* root) {
        if(!root) return;
        mirror(root->left);
        mirror(root->right);
        swap(root->left,root->right);//从底向上一次互换镜像
    }
};
```

### 25:[对称的二叉树](https://www.acwing.com/problem/content/38/)

#### 思路

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 * 两个子树互为镜像当且仅当：
    两个子树的根节点值相等；
    第一棵子树的左子树和第二棵子树的右子树互为镜像，且第一棵子树的右子树和第二棵子树的左子树互为镜像
 */
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        return dfs(root->left,root->right);
    }
    bool dfs(TreeNode* p, TreeNode* q){
        //如果左节点为空或者右节点为空，两者必须为空才可能相等
        // if(!p&&!q)return true;
        // if(!p||!q)return false;
        if(!p || !q) return p == q;//!p&&!q
        if(p->val != q->val) return false;
        return dfs(p->left,q->right) && dfs(p->right,q->left);
    }
};
```

### 26.[顺时针打印矩阵](https://www.acwing.com/problem/content/39/)

#### 思路：

![image-20220322154633933](剑指offer.assets/image-20220322154633933.png)

```c++
/*从左上角开始遍历，先往右走，走到不能走为止，然后更改到下个方向，再走到不能走为止，依次类推，遍历 n2 个格子后停止。*/
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> res;
        int n = matrix.size();
        if(!n) return res;
        int m = matrix[0].size();
        
        vector<vector<bool>> st(n, vector<bool>(m, false));
        int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
        int x = 0, y = 0, d = 1;
        
        for(int i = 0; i < n * m; i ++){
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[d], b = y + dy[d];
            if(a < 0 || a >= n || b < 0 || b >= m || st[a][b]){
                d = (d + 1) % 4;
                a = x + dx[d], b = y + dy[d];
            }
            x = a, y = b; 
        }
        return res;
    }
};
```

### 27.[包含main的栈](https://www.acwing.com/problem/content/90/)

#### 题解：

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <stack>
using namespace std;
class MinStack
{
public:
    void pop()
    {
        if(min_stk.top() == stk.top()) min_stk.pop();
        stk.pop();
    }
    void push(int value)
    {
        stk.push(value);
        if (min_stk.empty() || min_stk.top() >= value) min_stk.push(value);
    }
    int min()
    {
        return min_stk.top();
    }
    bool empty()
    {
        return stk.empty();
    }
private:
    stack<int> stk,min_stk;
};

int main()
{
    int data[] = { 5, 3, 1, 2, 1, 3 };
    int n = (sizeof data) / 4;
    MinStack minStack;
    for (int i = 0; i < n; i++)
    {
        minStack.push(data[i]);
        cout << minStack.min() << endl;
    }
    cout << "--------------------" << endl;
    for (int i = 0; i < n; i++)
    {
        cout << minStack.min() << endl;
        minStack.pop();
    }
    return 0;
}
```

### 28[.栈的压入、弹出序列](https://www.acwing.com/problem/content/40/)

#### 题解

```c++
class Solution {
public:
    bool isPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.empty() && popV.empty()) return true;
        if(popV.size() != pushV.size()) return false;
        int popIndex = 0;
        stack<int> stk;
        int len = pushV.size();
        for(int i = 0; i < len; i ++ ){
            stk.push(pushV[i]);//模拟每一次栈弹出操作   是不是和 popV一样。
            while(!stk.empty() && stk.top() == popV[popIndex]){
                stk.pop();
                popIndex ++;
            }
        }
        return stk.empty();
    }
};
```

### 29.[不分行从上往下打印二叉树](https://www.acwing.com/problem/content/41/)

#### 题目

![image-20220322185215836](剑指offer.assets/image-20220322185215836.png)

#### 题解

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> printFromTopToBottom(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while(q.size()){
            auto t = q.front();
            res.push_back(t->val);
            q.pop();
            if(t->left)  q.push(t->left);
            if(t->right) q.push(t->right);
        }
        return res;
    }
};
```

### 30.[分行从上往下打印二叉树](https://www.acwing.com/problem/content/42/)

#### 题解

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> printFromTopToBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while(q.size()){
            int level_len = q.size();//每层的个数
            vector<int> temp;
            for(int j = 0; j < level_len; j++){//while(level_len --){
                root = q.front();//auto node = q.front();
                temp.push_back(root->val);
                q.pop();
                if(root->left) q.push(root->left);
                if(root->right)q.push(root->right);
            }
            res.push_back(temp);
        }
        return res;
    }
};
```

### 31.[之字形打印二叉树](https://www.acwing.com/problem/content/43/)



#### 题解

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> printFromTopToBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        
        queue<TreeNode*> q;
        q.push(root);
        int i = 1;
        while(q.size()){
            
            int level_len = q.size();
            vector<int> temp;
            
            for(int i = 0; i < level_len; i++ ){
                root = q.front(); 
                temp.push_back(root ->val);
                q.pop();
                if(root->left) q.push(root->left);
                if(root->right)q.push(root->right);
            }
            if(i % 2 == 0) reverse(temp.begin(), temp.end());//偶数行直接数组倒排
            i ++;
            res.push_back(temp);
        }
        return res;
    }
};
```

### 32.[二叉搜索树的后序遍历序列](https://www.acwing.com/problem/content/44/)

#### 题解

```c++
class Solution {
public:
    bool verifySequenceOfBST(vector<int> sequence) {
        // if(sequence.empty()) return false;
        if(sequence.empty() || sequence.size() == 1) return true;
        return dfs(sequence, 0, sequence.size() - 1);
    }
    bool dfs(vector<int> seq, int l, int r){
        if(l >= r) return true;
        int root = seq[r];
        
        int k = l;
        while(k < r && seq[k] < root) k ++;
        
        for(int i = k; i < r; i ++){
            if(seq[i] < root) return false;
        }
        return dfs(seq, l, k - 1) && dfs(seq, k, r - 1);
    }
};
```

### 33.二[叉树中和为某一值的路径](https://www.acwing.com/problem/content/45/)

#### 题解

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */老牌方法  从子节点以上遍历 找出每条路径的
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;
    vector<vector<int>> findPath(TreeNode* root, int sum) {
        dfs(root, sum);
        return ans;
    }
    void dfs(TreeNode* root, int sum){
        if(!root) return;
        path.push_back(root->val);
        sum -= root->val;
        if(!root->left &&!root->right && !sum) ans.push_back(path);
        dfs(root->left, sum);
        dfs(root->right, sum);
        
        path.pop_back();
    }
};
```

### 34：[复杂链表的复刻](https://www.acwing.com/problem/content/89/)



#### 题解

```c++
/**
 * Definition for singly-linked list with a random pointer.
 * struct ListNode {
 *     int val;
 *     ListNode *next, *random;
 *     ListNode(int x) : val(x), next(NULL), random(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *copyRandomList(ListNode *head) {
        for(auto p = head; p;){
            auto temp = new ListNode(p -> val);
            // auto next = p -> next;
            // p -> next = temp;
            // temp -> next = next;
            // p = next;
            temp -> next = p -> next;
            p -> next = temp;
            p = temp -> next;
        }
        for(auto p = head; p; p = p -> next -> next)
            if(p -> random)
                p -> next -> random = p -> random -> next;//赋给复制的节点random
                
        auto dummy = new ListNode(-1);
        auto cur = dummy;//复制的链表
        for(auto p = head; p;){
            auto pre = p;//原来的链表
            cur -> next = p -> next;
            cur = cur -> next;
            p = p -> next -> next;//原来链的节点
            pre -> next = p;
        }
        return dummy -> next;
    }
};
```

### 35，[二叉搜索树与双向链表](https://www.acwing.com/problem/content/87/)

#### 题解

错误的  新建了节点

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* convert(TreeNode* root) {
        if(!root) return nullptr;
        root = dfs(root);
        while(root -> left) root = root -> left;
        return root;
    }
    TreeNode* dfs(TreeNode* root){
        if(!root->left && !root->right) return root;
        if(root->left){
            auto left = dfs(root->left);
            while(left->right) left = left->right;
            left->right = root;
            root->left = left;
        }
        if(root->right){
            auto right = dfs(root->right);
            while(right->left) right = right->left;
            right->left = root;
            root->right = right;
        }
        return root;
    }
};
```

### 36.[序列化二叉树](https://www.acwing.com/problem/content/46/)

#### 题解

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 * "[8, 12, 2, null, null, 6, 4, null, null, null, null]"
 */
class Solution {
private:
    void dfs_s(TreeNode* root, string &res){
        if(!root) {
            res += "# ";
            return;
        }
        res += to_string(root -> val) + ' ';
        dfs_s(root->left, res);
        dfs_s(root->right, res);
    }
    
    TreeNode* dfs_d(string &str, int &index){
        if(str[index] == '#'){
            index ++ ;
            return nullptr;
        }
        
        int num = 0, negativeFlag = 1;
        if(str[index] == '-'){
            negativeFlag = -1;
            index ++;
        }
        while(str[index] != ' '){
            num = num * 10 + str[index] - '0';
            index ++;
        }
        num = num * negativeFlag;
        
        TreeNode* root = new TreeNode(num);
        root->left = dfs_d(str, ++index);
        root->right = dfs_d(str, ++index);
        return root;
    }   
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string res;
        dfs_s(root, res);
        return res;
    }
    
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        int index = 0;
        return dfs_d(data, index);
    }

};
```

### 37：[数字排列](https://www.acwing.com/problem/content/47/)

#### 题解

```c++
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        // res.clear();
        // path.clear();
        vector<bool> used(nums.size(), false);
        sort(nums.begin(), nums.end());
        dfs(nums, used);
        return res;
    }
    void dfs(vector<int>& nums, vector<bool>& used){
        if(nums.size() == path.size()){
            res.push_back(path);
            return;
        }
        //
        for(int i = 0; i < nums.size(); i ++){
            // used[i - 1] == true，说明同⼀树⽀nums[i - 1]使⽤过
            // used[i - 1] == false，说明同⼀树层nums[i - 1]使⽤过
            // 如果同⼀树层nums[i - 1]使⽤过则直接跳过
            if(i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false)
               continue;
            if(used[i] == false){
                path.push_back(nums[i]);
                used[i] = true;
                dfs(nums, used);
                path.pop_back();
                used[i] = false;
            }
        }
    }
};
```

### 38.[数组中出现次数超过一半的数字](https://www.acwing.com/problem/content/48/)

```c++
class Solution {
public:
    int moreThanHalfNum_Solution(vector<int>& nums) {
        int cnt = 1, val = nums[0];
        for(int i = 1; i < nums.size(); i ++){
            if(nums[i] == val) cnt ++;
            else cnt --;
            if(!cnt) {
                cnt = 1;
                val = nums[i];
            }
        }
        return val;
    }
};

class Solution {
public:
    int moreThanHalfNum_Solution(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        return nums[nums.size()/2];
    }
};
```

### 39：[最小的k个数](https://www.acwing.com/problem/content/49/)

#### 题解

```c++
class Solution {
public:
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        priority_queue<int> heap;//默认大顶堆
        for(auto x : input){
            heap.push(x);
            if(heap.size() > k) heap.pop();//维护k大小的大顶堆
        }
        
        vector<int> res;
        while(heap.size()) res.push_back(heap.top()), heap.pop();
        
        reverse(res.begin(), res.end());//反转 升序
        return res;
    }
};
```

### 40.[数据流中的中位数](https://www.acwing.com/problem/content/88/)

#### 题解

```c++
class Solution {
public:
    priority_queue<int> max_heap;
    priority_queue<int, vector<int>, greater<int>> min_heap;
    void insert(int num){
        max_heap.push(num);
        //如果大堆top小于小堆top 交换
        //如果大堆尺寸 大于   小堆 要维护两个堆
        if(min_heap.size() && max_heap.top() > min_heap.top()){
            auto maxv = max_heap.top(), minv = min_heap.top();
            min_heap.push(maxv),max_heap.push(minv);
            min_heap.pop(),max_heap.pop();
        }
        if(max_heap.size() > min_heap.size() + 1) {
            min_heap.push(max_heap.top());
            max_heap.pop();
        }
    }

    double getMedian(){
        if(min_heap.size() + max_heap.size() & 1) return max_heap.top();
        return (min_heap.top() + max_heap.top()) / 2.0;
    }
};
```

### 41.[连续子数组的最大和](https://www.acwing.com/problem/content/50/)

#### 题解

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN, sum = 0;
        for(auto x : nums){
            
            if(sum < 0) sum = 0;
            sum += x;                //一边累加sum一边更新最大值
            res = max(res, sum);
            //若数组中全是负数，那么最大值就是最小的那一个负数，因为负数越累加越小。
        }
        return res;
    }
};

//还有动态规划
```

### 42.[从1到n整数中1出现的次数](https://www.acwing.com/problem/content/51/)

#### 题解

```c++
/*
example 设数字为abcde，当前位数为c位，
        c位1的个数即为高位个数+低位个数
        高位范围为 00 ~ ab-1 ：
            有 ab*100 个(因为c在百位)
        低位分为三种情况：
            c = 0 ,有 0 个
            c = 1 ，有 de + 1 个
            c > 1 , 有 100 个 (因为c在百位)
        依次遍历每一位数相加，即为总共1的个数
*/
class Solution {
public:
    int numberOf1Between1AndN_Solution(int n) {
        if(!n) return 0;
        vector<int> nums;
        int res = 0;
        while(n) nums.push_back(n % 10), n /= 10;
        for(int i = nums.size() - 1; i >= 0; i --){
            auto left = 0, right = 0, t = 1;
            for(int j = nums.size() - 1; j > i; j --) left = left * 10 + nums[j];   //高位left 低位right
            for(int j = i - 1; j >= 0; j --) right = right * 10 + nums[j], t *= 10; //t是当前位置
            res += left * t;
            cout<<t<<" "<<res<<" "<< right<<" "<<left<<" ";
            if(nums[i] == 1) res += right + 1;
            else if(nums[i] > 1) res += t;
        }
        return res;
    }
};

/*
// 分两种情况，例如：1234和2234，high为最高位，pow为最高位权重 // 在每种情况下都将数分段处理，即0-999，1000-1999，...，剩余部分 // case1：最高位是1，则最高位的1的次数为last+1（1000-1234） // 每阶段即0-999的1的个数1*countDigitOne(pow-1) // 剩余部分1的个数为countDigitOne(last)--最高位已单独计算了 // case2：最高位不是1，则最高位的1的次数为pow（1000-1999） // 每阶段除去最高位即0-999，1000-1999中1的次数为high*countDigitOne(pow-1) // 剩余部分1的个数为countDigitOne(last) // 发现两种情况仅差别在最高位的1的个数，因此单独计算最高位的1（cnt），合并处理两种情 况
*/
class Solution {
public:
    int countDigitOne(int n) {
        if(!n) return 0;
        if(n < 10) return 1;
        int left = n, val = 1;
        while(left >= 10) left /= 10, val *= 10;//取出最高位 以及 最高位的权重
        cout<< " " << left << " "<< val;
        int last = n - left * val;
        int cnt = left == 1 ? last + 1:val;// 最高位是否为1，最高位的1个数不同
        return cnt + left * countDigitOne(val - 1) + countDigitOne(last);
    }
};
```

### 43;[数字序列中某一位的数字](https://www.acwing.com/problem/content/52/)

#### 题解

```c++
class Solution {
public:
    int digitAtIndex(int n) {
        long long len = 1, cnum = 9, val = 1;//几位数，个数，位数的起始值
        //确定是几位数
        while(n > len * cnum){  // 1000 - 9 - 90 * 2 - 900 * 3 ,。
            n -=len * cnum;     //当i= 3 时不符合条件，说明是在三位数里面
            len ++;
            cnum *= 10;
            val *= 10;
        }
        //确定是这几位数的第几个数 
        int number = val + (n + len - 1) / len - 1;//base以后的第几个数 考虑0故减1 n/i 向上取整
        //属于这个数的第几位
        int index = n % len ? n % len : len;//余数  除不尽就是第几位，除尽就是最后一位。
        for(int i = 0; i < len - index; i ++) number /= 10;//数的第len - index位 339第二位
        
        return number % 10;//index=len 最后一位
    }
};
```

### 44.[把数组排成最小的数](https://www.acwing.com/problem/content/54/)

#### 题解

```c++
class Solution {
public:
    static bool cmp(int a, int b){
        auto sa = to_string(a), sb = to_string(b);
        return sa + sb < sb +sa;//说明a < b 即a应该排在b前面
    }
    string printMinNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), cmp);
        string res;
        for(auto x : nums) res += to_string(x);
        return res;
    }
};
```

### 45[把数字翻译成字符串](https://www.acwing.com/problem/content/55/)

#### 题解

```c++
class Solution {
public:
    int getTranslationCount(string nums) {
 
        int n = nums.size();

        vector<int> f(n + 1);
        f[0] = 1;

        for(int i = 1; i <= n; i ++){
            f[i] = f[i - 1];
            if(i > 1){
                int t = (nums[i - 2]-'0') * 10 + nums[i - 1]-'0';
                if(t >= 10 && t <= 25) f[i] += f[i - 2];
            }
        }
        return f[n];
    }
};
```

### 46.[礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/submissions/)

#### 题解

```c++
扩容不扩容
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        // for(int i = 0; i < m; i++) dp[i][0] = grid[i][0];
        // for(int j = 0; j < n; j++) dp[0][j] = grid[0][j];
		//dp[i][j] ：表示从（0 ，0）出发，到(i-1, j-1) 有最大价值。
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) 
            dp[0][i] = dp[0][i - 1] + grid[0][i];

        for (int i = 1; i < m; i++) 
            dp[i][0] = dp[i-1][0] + grid[i][0];

        for(int i = 1; i < m; i++)
            for(int j = 1; j < n; j ++){
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        return dp[m - 1][n - 1];
    }
};

class Solution {
public:
    int getMaxValue(vector<vector<int>>& grid) {
        //dp[i][j]表示从grid[0][0]到grid[i - 1][j - 1]时的最大价值
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        for(int i = 1; i <= n; i ++)
            for(int j = 1; j <= m; j ++)
                f[i][j] = max(f[i - 1][j], f[i][j - 1] )+ grid[i - 1][j - 1];
        //dp[i][j] ：表示从（1，1）出发，到(i, j) 有最大价值。
        for(auto &x : f)
            for(auto y : x)
                cout << y << " ";
            
        return f[n][m];
    }
};
```

**双指针前提是有单调性**

### [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210310102321909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1lvdU1pbmdfTGk=,size_16,color_FFFFFF,t_70)

```c++
class Solution {
public:
//i,j两个指针，当j指针大于1以后，i++，直到j指针所在的字符等于1
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> umap;
        int res = 0;
        for(int i = 0, j = 0; j < s.size(); j++){
            umap[s[j]]++;
            while(umap[s[j]] > 1){
                umap[s[i]]--;
                i++;
            }
            res = max(res, j - i + 1);
        }
        return res;
    }
};
```

### [263. 丑数](https://leetcode.cn/problems/ugly-number/)

```c++
class Solution {
    //使用数据结构set来存储每个元素是否出现过，没有就加入堆中
	//小根堆会自动排序，将整个输入变成一个升序排列的序列。
public:
    int nthUglyNumber(int n) {
        vector<int>factors = {2,3,5};
        unordered_set<long> umap;
        priority_queue<long, vector<long>, greater<long>> minheap;

        umap.insert(1L);
        minheap.push(1L);

        while(--n){// 
            long cur = minheap.top();
            minheap.pop();

            for(int factor:factors){
                long next = factor * cur;
                if(!umap.count(next)){
                    umap.insert(next);
                    minheap.push(next);
                }
            }
        }
        return minheap.top();

    }
};

/*
1.三指针的思想，所以定义3个指针i, j, k。
2。vector存储的是丑数数组，一开始只有1个1，后面 动态添加元素进vector。
t取出的是3个指针分别指向的3个子数组(2 3 5)中的最小值。如果最小值是3个子数组中的哪一个，就把里面的指针i j k 增1。因为可能同时出现在多个数组，所以用3个if来表示。
3.最后输出vector的最后一位，就是第n个丑数
*/
class Solution {
public:
    int getUglyNumber(int n) {
        vector<long>res(1,1);
        
        int i = 0, j = 0, k = 0;
        while(--n){
            long cur = min(res[i] * 2, min(res[j]*3, res[k]*5));
            res.emplace_back(cur);
            
            //去重
            if(res[i]*2 == cur) i++;
            if(res[j]*3 == cur) j++;
            if(res[k]*5 == cur) k++;
        }
        // for(auto x : res)
        //     cout << x << " ";
        return res.back();
    }
};
```

### [387. 字符串中的第一个唯一字符](https://leetcode.cn/problems/first-unique-character-in-a-string/)

```c++
class Solution {
public:
    int firstUniqChar(string s) {
        if(s.size()==0) return -1;
        unordered_map<char,int>umap;
        for(auto str:s)
            umap[str]++;
        for(int i = 0; i< s.size(); i++)
            if(umap[s[i]]==1)
                return i;
        return -1;
    }
};
```

### [64.字符流中第一个只出现一次的字符]([64. 字符流中第一个只出现一次的字符 - AcWing题库](https://www.acwing.com/problem/content/60/))

```c++
class Solution{
public:
    unordered_map<char,int>umap;
    queue<char>que;
    //Insert one char from stringstream
    void insert(char ch){
        //等于1才加入
        if(++umap[ch]>1){
            while(!que.empty() && umap[que.front()] > 1)que.pop();
        }
        else que.push(ch);
    }
    //return the first appearence once char in current stringstream
    char firstAppearingOnce(){
        if(que.empty()) return '#';
        return que.front();
    }
};
```

### [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```c++
 class Solution {
public:
    int merge(vector<int>& nums, int l, int r){
        if(l >= r) return 0;
        int mid = (l + r) >> 1;
        int res = merge(nums, l, mid) + merge(nums, mid + 1, r);
        int i = l, j = mid+1;
        vector<int>temp;
        while(i <= mid && j <= r){
            if(nums[i] <= nums[j]) temp.push_back(nums[i++]);
            else{
                temp.push_back(nums[j++]);//后面的元素小于前面的元素
                res += mid - i + 1;
            }
        }
        //无论是i或者是j，元素都比之前的数组要大，所以不可能存在新的逆序
        while(i <=mid) temp.push_back(nums[i++]);
        while(j <= r)  temp.push_back(nums[j++]);
        i = l;
        for(auto x:temp) nums[i++] = x;
        
        for(auto y : nums) cout << y << " ";
        cout <<endl;
        
        return res;
    }
    int inversePairs(vector<int>& nums) {
        return merge(nums, 0, nums.size()-1);
    } 
};
```

### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```c++
//有的是  nums[mid] < k 和 nums[mid] <= k 来划分  都可以
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.empty()) return 0;

        int l = 0, r = nums.size()-1;
        while(l < r){
            int mid = (l + r )>> 1;
            if(nums[mid] >= target) //大于key的第一个元素
                r = mid;
            else l = mid + 1;
        }

        if(nums[l] != target) return 0;
        int left = l;
        cout << left << endl;

        l = 0, r = nums.size()-1;
        while(l < r){
            int mid = (l + r + 1)>> 1;
            if(nums[mid] <= target) //小于key的最后一个元素
                l = mid;
            else r = mid -1;
        }
        cout << l << " " <<  r - left + 1 <<endl;
        return r - left + 1;
    }
};
```

### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

![缺失数字.png](https://cdn.acwing.com/media/article/image/2019/05/31/1_37a28f4683-%E7%BC%BA%E5%A4%B1%E6%95%B0%E5%AD%97.png)

```c++
class Solution {
public:
//注意特殊情况：当所有数都满足nums[i] == i时，表示缺失的是 nn
    int missingNumber(vector<int>& nums) { 
        if (nums.empty()) return 0;
        int l = 0, r = nums.size()-1;
        while(l < r){
            int mid = (l + r ) >> 1;
            if(nums[mid] == mid) l = mid + 1;
            else r = mid;
        }
        /*int l = 0, r = nums.size() - 1;
        while (l < r)
        {
            int mid = l + r >> 1;
            if (nums[mid] != mid) r = mid;
            else l = mid + 1;
        }*/
        if(nums[r] == r) r++;
        return r;
    }
};

class Solution {
public:
    int getMissingNumber(vector<int>& nums) {
        int n = nums.size() + 1;
        int res = n*(n -1) / 2;
        for(auto x:nums) res -= x;
        return res;
    }
};


```

69. ### [数组中数值和下标相等的元素]([69. 数组中数值和下标相等的元素 - AcWing题库](https://www.acwing.com/problem/content/65/))

```c++
//数组单调递增我们就可以利用单调性来找答案
class Solution {
public:
    int getNumberSameAsIndex(vector<int>& nums) {
        int l = 0, r = nums.size()-1;
        while(l < r){
            int mid = (l + r) << 1;
            if(nums[mid] - mid >= 0) r = mid;
            else l = mid + 1;
        }
        return nums[l] == l ? l : -1; 
    }
};
```

### [230. 二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

```c++
class Solution {
public:
    TreeNode* res;
    void travesal(TreeNode* cur, int &k){
        if(!cur) return;
        travesal(cur->left,k);
        k--;
        if(!k) res = cur;
        if(k>0) travesal(cur->right,k);
    }
    int kthSmallest(TreeNode* root, int k) {
        travesal(root, k);
        return res->val;
    }
};

class Solution {
public:
    TreeNode* kthNode(TreeNode* root, int k) {
        stack<TreeNode*>st;
        auto cur = root;
        while(!st.empty()||cur){
            if(cur){
                st.push(cur);
                cur=cur->left;
            }
            else{
                cur = st.top();
                st.pop();
                k--;
                if(!k) root = cur;
                cur=cur->right;
            }
        }
        return root;
    }
};
```

### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*>que;
        int depth = 0;
        if(root) que.push(root);
        while(!que.empty()){
            int size = que.size();
            depth++;
            for(int i = 0; i< size; i++){
                auto node = que.front();
                que.pop();
                if(node->left) que.push(node->left);
                if(node->right)que.push(node->right);
            }
        }
        return depth;
    }
};
```

### [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

```c++
class Solution {
public:
    bool res = true;
    int traversal(TreeNode* cur){
        if(!cur) return 0;
        int left = traversal(cur->left), right = traversal(cur->right);
        if(abs(left-right) > 1) 
            res = false;
        return max(left, right) + 1;
    }
    bool isBalanced(TreeNode* root) {
        traversal(root);
        return res;
    }
};
```

### [73. 数组中只出现一次的两个数字 - AcWing题库](https://www.acwing.com/problem/content/69/)

[可以了解](https://leetcode.cn/problems/single-number-iii/solution/dong-hua-tu-jie-yi-ding-neng-hui-by-yuan-gqg8/)

```c++
/*
任何一个数字异或他自己都等于0，从头到尾异或所有数字之后，出现两次的数字相互异或会变成0，所以最终的结果会是x异或y，也即两个仅出现一次的数字的异或。x^y这个异或结果一定不为0，因为x不等于y，所以肯定至少有一位为1。我们找到这个结果中第一个为1的位，在这个位上x和y的二进制表示肯定相反。所以我们可以对于原先的数组（分类），根据该位是否为1，将其划分成两个数组，这样两个数组中就分别存在一个仅出现一次的数字。我们再分别对两个数组进行异或运算，最后分别剩下的结果，就是x和y
*/

class Solution {
public:
    vector<int> findNumsAppearOnce(vector<int>& nums) {
        int sum = 0;
        for(int num : nums) sum ^= num; //循环结束后 xy = x^y
        int k = 0;
        while(!(sum >> k & 1)) k++; //循环结束之后k就是xy二进制表示中第一个为1的位
        int first = 0;
        for(int num : nums) //从xy中分离x和y
            if(num >> k & 1)
                first ^= num;
        int y = first ^ sum;
        return {first, y};
    }
};
```

### [[137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)](https://leetcode.cn/problems/single-number-ii/solution/zhi-chu-xian-yi-ci-de-shu-zi-ii-by-leetc-23t6/)

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for(int i = 0; i < 32; i++){
            int cnt = 0;
            for(int num : nums)
                if(num >> i & 1) 
                    cnt ++;

            if(cnt % 3) 
                res = res | 1 << i;
        }
        return res;
    }
};
```

### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

```c++
class Solution {
public:
    vector<vector<int> > findContinuousSequence(int sum) {
        vector<vector<int>> res;
        for(int i = 1, j = 1, s = 1; i <= sum; i ++){
            while(s < sum) s += ++j;
            if(s == sum && j - i + 1 > 1){
                vector<int>line;
                for(int k = i; k <= j; k ++) line.emplace_back(k);
                res.emplace_back(line);
            }
            //维护s
            s -= i;
        }
        return res;
    }
};
```

### [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

```c++
class Solution {
public:
    string reverseWords(string s) {
        string str;
        for(int i = s.size()-1; i >= 0; i--){
            if(s[i] != ' '){
                str += " ";
                string temp;
                while(i >= 0 && s[i] != ' '){
                    temp = s[i] + temp;
                    i--;
                }
                str += temp;
            }
        }
        str.erase(str.begin());
        return str;

    }
};
```

### [剑指 Offer 58 - II. 左旋转字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```c++
class Solution {
public:
    string reverseLeftWords(string str, int k) {
        int n = str.size();
        reverse(str.begin(),str.end());
        reverse(str.begin(), str.begin()+n-k);
        reverse(str.begin() +n-k, str.end());
        return str;
    }
};
```

### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode.cn/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

单调队列

```c++
class Solution {
public:
    vector<int> maxInWindows(vector<int>& nums, int k) {
        deque<int>que;//队列里存的是下标
        vector<int>res;
        for(int i = 0; i < nums.size(); i++){
            while(que.size() && i - que.front() >= k) que.pop_front();//维持k个
            while(que.size() && nums[i] >= nums[que.back()]) que.pop_back();//维持单调
            que.push_back(i);
            for(auto x : que)
                cout<<x<<" ";
            cout<<endl;
            if(i + 1 >= k)//前面k-1个数不是答案
                res.push_back(nums[que.front()]);
        }

        return res;
    }
};
```

### [80. 骰子的点数 - AcWing题库](https://www.acwing.com/problem/content/76/)



```c++
(DFS、暴力枚举)
1.DFS跟动规一样，注重3点：
	状态的表示，也就是状态的定义。
	按照什么样的顺序来计算这个状态
	边界情况，最后一次掷色子的情况，分为6类
2.dfs(n, s)表示一共投了n次色子，总和是s的情况下，方案数是多少。
3.dfs()表示的就是答案，也就是要求的东西
    
class Solution {
public:
    vector<int> numberOfDice(int n) {
        vector<int> res;
        for(int i = n; i <= 6*n; i++) res.emplace_back(dfs(n, i));
        return res;
    }
    //掷色子n次 和为sum 返回可能掷的方案
    int dfs(int n, int sum){
        if(n == 0) return !sum;//n =0时sum=0就是一个方案
        if(sum < 0) return 0;//wujie
        //热狗分类
        //最后一次掷的色子是i点的方案
        int res = 0;
        for(int i = 1; i <= 6; i++){
            res += dfs(n - 1, sum - i);
        }
        return res;
     }
};


(线性DP)
1.DP也是，考虑状态表示和状态计算，和最后的边界情况。
2.f[i][j]表示——前i次掷色子，总和是j的方案数。
3.边界就是最后一次的情况，分6类，不同的类对应不同的结果。
4.三重for循环中，投掷1次，就有6种可能的点数，投掷2次，就有12种可能的点数，所以投掷n次，就有6n种可能的点数。所以在第2重循环中，j <= i * 6，因为可能还没到第n次，总数j也不会到6n个。
5.最后一重循环，因为最后的模型是f[i][j] += f[i - 1][j - k]。也就是i次中，投出1次后，变成n - 1次，总和从j变到j - k后剩余的方案数。因为是j - k，不能越界，所以j >= k，所以k = min(j, 6)，可能前期j还没有枚举到6，那么k就不能取到6。
6.最后我们将计算好的答案放在res中，f[n][i]，i = n ~ 6n，表示投掷n次后，总和分别为n~6n的所有方案数。
 class Solution {
public:
    vector<int> numberOfDice(int n) {
        //掷前n次点数和时sum（n~6n）的方案数
        vector<vector<int>>f(n+1, vector<int>(6*n + 1));//掷n次n步
        // f[0][0] = 1;
        for(int i = 1; i <= 6; i++) f[1][i] = 1;//掷一次的点数方案
        for(int i = 2; i <= n; i ++)//掷n次
            for(int j = i; j <= 6 * i; j++)//枚举sum
                // 每一次方案是前i - 1个骰子6种点数的总和
                for(int k = 1; k <= 6; k++) 
                    if(j - k > 0)
                        f[i][j] += f[i-1][j-k];
        vector<int>res;
        for(int i = n; i<= 6*n; i++) res.emplace_back(f[n][i]);
        return res;
    }
};

class Solution {
public:
    bool isContinuous( vector<int> numbers ) {
        if(!numbers.size()) return false;
        //while  用来过滤
        sort(numbers.begin(), numbers.end());
        int k = 0;
        while(!numbers[k]) k++;
        for(int i = k + 1; i < numbers.size(); i++){
            if(numbers[i] == numbers[i - 1])
                return false;
        }
        if(numbers[k] == 1 && numbers.back() > 10) 
            return numbers.back() - numbers[k] <= 12 && numbers[k+1] > 9;
            
        return numbers.back() - numbers[k] <= 4; 
    }
};
```

### [约瑟夫环问题（递归+数学+迭代优化空间） - 圆圈中最后剩下的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/yue-se-fu-huan-wen-ti-di-gui-shu-xue-die-nxdx/)

<img src="https://img-blog.csdnimg.cn/20210313172428537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1lvdU1pbmdfTGk=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:67%;" />

![未命名文件 (1).jpg](https://pic.leetcode-cn.com/1651627951-UbNqoJ-%E6%9C%AA%E5%91%BD%E5%90%8D%E6%96%87%E4%BB%B6%20(1).jpg)

![约瑟夫环2.png](https://pic.leetcode-cn.com/68509352d82d4a19678ed67a5bde338f86c7d0da730e3a69546f6fa61fb0063c-%E7%BA%A6%E7%91%9F%E5%A4%AB%E7%8E%AF2.png)

```c++
class Solution {
public:
    int lastRemaining(int n, int m){
        if(n == 0) return 0;
        return (lastRemaining(n - 1, m) + m) % n;
    }
};

class Solution {
public:
    int lastRemaining(int n, int m){
        int f = 0;//n个数字
        for(int i = 2; i <= n; i++){
            f =(f + m) % i;
        }
        return f;
    }
};
```

### [剑指 Offer 63. 股票的最大利润](https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof/)

```c++
class Solution {
public:
    int maxDiff(vector<int>& nums) {
        int res = 0;
        for(int i = 0; i < nums.size(); i++){
            for(int j = i + 1; j < nums.size(); j++)
            {
                res = max(res, nums[j] - nums[i]);
            }
        }
        return res;
    }
};
|
v
class Solution {
public:
    int maxDiff(vector<int>& nums) {
        int res = 0;
        if(nums.empty()) return 0;
        for(int i = 1, minv = nums[0]; i < nums.size(); i++){
            res = max(res, nums[i] - minv);
            minv = min(minv, nums[i]);
        }
        return res;
    }
};
```

