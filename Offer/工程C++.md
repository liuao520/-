

*通过父类指针去析构子类对象，分三种情况：*

*1**、父类如**A**的析构函数不是虚函数，这种情况下，将只会调用**A**的析构函数而不会调用子类的析构函数，前面的文章中有提到过，*

*非虚函数是通过类型来寻址的，这样的析构将会导致析构畸形*

virtual成员函数通过虚函数表保留一份跟踪记录,不接受对象的类型指针的迷惑,能正确找到对应函数

但是,当使用基类指针指向派生类时候,通过指针渠道的对象类型是错误的,要通过虚函数表机制找到继承栈正确的析构



*2**、父类如**A**的析构函数是普通的虚函数，这种情况下，会很正常，从子类一直析构到基类，最后完成析构*

*3**、父类如**A**的析构函数是纯析构函数，如本文所提，正是重点，在这种情况之下，由于析构函数首先是虚函数，所以会按**2**的方法从子类一直析构到父类，但是，又由于父类的析构函数是纯虚函数，没有实现体，所以，当析构到父类时，由于没有实现体，所以导致父类无法析构，最终也导致了析构畸形，因此，特殊的地方就在于这里，纯虚析构函数需要提供一个实现体，以完成对象的析构*





/*

1. 基于对象风格：具体类加全局函数的设计风格。
2. 面向对象风格：使用继承和多态的设计风格。

C编程风格,注册三个全局函数到网络库,网络库通过函数指针来回调

面向对象风格,用一个EchoServer继承TcpSever(抽象类),实现三个接口onConnection,onMessage,onClose

基于对象风格,用一个EchoServer包含一个TcpServer(具体类),在构造函数中用boost::bind来注册三个成员函数onConnection,onMessage,onClose

*/

数据的生产者消费者队列  

![image-20220510222613487](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220510222613487.png)

## 智能指针

![image-20220519130229565](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220519130229565.png)



![image-20220519130539789](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220519130539789.png)

![image-20220522184137242](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522184137242.png)

![image-20220519130717801](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220519130717801.png)

![image-20220519130403670](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220519130403670.png)

![image-20220519130337490](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220519130337490.png)

## 基本底层

### 进程虚拟地址空间区域划分

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522125516913.png" alt="image-20220522125516913" style="zoom: 67%;" />![image-20220522130834797](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522130834797.png)

![image-20220522131030785](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522131030785.png)

### 从指令角度分析函数调用堆栈过程

![image-20220522134334434](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522134334434.png)

![image-20220522134235947](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522134235947.png)

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522134257580.png" alt="image-20220522134257580" style="zoom: 50%;" />

### 从编译器角度分析C++的编译和链接原理

![image-20220522144443603](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522144443603.png)

![image-20220522144632934](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522144632934.png)

![image-20220522144606172](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522144606172.png)

## 对象的应用优化 右值引用的优化

#### 1.对象使用过程中背后调用方法

![image-20220522145942369](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522145942369.png)

![image-20220522150044300](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522150044300.png)

##### **不能用一个指针去指向一个临时变量**

![image-20220522150356171](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522150356171.png)

​							**常引用**

##### 总结

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522151434658.png" alt="image-20220522151434658" style="zoom:67%;" />

7.逗号表达式 最后一个表达式的值?????

#### 2.函数调用过程中对象背后调用

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522152524797.png" alt="image-20220522152524797" style="zoom: 67%;" />

实参->形参  拷贝构造函数  引用是别名,就不需要

返回tmp:在main栈上拷贝构造一个临时对象 赋值

**对象初始化是调用构造函数的,赋值是两个对象都存在了,左边的赋值**

#### 2.1 三个对象优化规则

![image-20220522161735418](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522161735418.png)

备注:用临时对象拷贝构造一个新对象时会优化 直接构成一个临时变量  不会额外构造加赋值

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522161132413.png" alt="image-20220522161132413" style="zoom: 67%;" />

以初始化的方式接受临时变量

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522161631884.png" alt="image-20220522161631884" style="zoom:67%;" />

#### 3.右值优化

![image-20220522154154650](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522154154650.png)

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522155342306.png" alt="image-20220522155342306" style="zoom:67%;" />

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522155533281.png" alt="image-20220522155533281" style="zoom: 80%;" />



#### 右值应用之后

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522160146326.png" alt="image-20220522160146326" style="zoom:50%;" />

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522155943750.png" alt="image-20220522155943750" style="zoom:67%;" />

## 32



## C11引入

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522164159102.png" alt="image-20220522164159102" style="zoom:67%;" />

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522164233640.png" alt="image-20220522164233640" style="zoom: 80%;" />



#### C++语言级别支持的多线程编程

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522173044701.png" alt="image-20220522173044701" style="zoom: 67%;" />

##### 通过thread类编写C++多线程程序

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522175141791.png" alt="image-20220522175141791" style="zoom: 67%;" />

**所有自线程自动结束**

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522175316907.png" alt="image-20220522175316907" style="zoom:50%;" />

##### 线程间同步通信-生产者消费者模型

![image-20220522183715070](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522183715070.png)

![image-20220522200341723](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522200341723.png)

![image-20220522183527336](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522183527336.png)

##### 线程间互斥-mutex互斥锁和lock_guard





![image-20220522185135809](G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522185135809.png)

##### 基于CAS操作的atomic原子类型

**相当于给总线加锁，以原子操作做了一个寄存器和内存之间的数据交换，并不会改变线程之间的状态**

**而互斥锁，是一个比较重的锁，会改变线程的状态**

<img src="G:\desktop\work\learning\study_point\Offer\工程C++.assets\image-20220522172708348.png" alt="image-20220522172708348" style="zoom:80%;" />

##### volatile 

防止多线程对于共享对象进行缓存,一个线程对于共享对象的改变立马反应到另一个线程上

## 模板编程



## 设计模式



## 面向对象



## 继承与多态



## 运算符重载

