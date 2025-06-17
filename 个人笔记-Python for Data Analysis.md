

**<font size=8>[Python for Data Analysis, 3E](https://wesmckinney.com/book/)</font>**

只与DL相关。

# 1 Preliminaries

## 1.1 What is this book about?

介绍一些用python进行数据分析的package。

### What kinds of data?

结构化数据，如：

- Tabular or spreadsheet-like data in which each column may be a different type (string, numeric, date, or otherwise). This includes most kinds of data commonly stored in relational databases or tab- or comma-delimited text files.
- Multidimensional arrays (matrices).
- Multiple tables of data interrelated by key columns (what would be primary or foreign keys for a SQL user).
- Evenly or unevenly spaced time series.

## 1.2 Why python for data analysis?

## 1.3 Essential python libraries

### NumPy

Numerical Python的缩写。提供数据结构、算法和library glue。它包含：

- Array类：A fast and efficient multidimensional array object *ndarray*
- array的计算：Functions for performing element-wise computations with arrays or mathematical operations between arrays
- 读写array的工具：Tools for reading and writing array-based datasets to disk
- 线性代数操作：Linear algebra operations, Fourier transform, and random number generation
- 用其它语言的接口：A mature C API to enable Python extensions and native C or C++ code to access NumPy's data structures and computational facilities

用它来存数字，在python中操作是最快的。其它语言也可以直接用NumPy array。

python中很多数字的计算工具都是基于NumPy的。

### pandas

Penel data的缩写。

The primary objects in pandas that will be used in this book are the DataFrame, a tabular, column-oriented data structure with both row and column labels, and the Series, a one-dimensional labeled array object.它将NumPy的阵列计算想法与表格和关系数据库中的数据操作功能相结合，提供了方便的索引功能，可以reshape、聚合、切片数据。

- Data structures with labeled axes supporting automatic or explicit data alignment—this prevents common errors resulting from misaligned data and working with differently indexed data coming from different sources
- Integrated time series functionality
- The same data structures handle both time series data and non-time series data
- Arithmetic operations and reductions that preserve metadata
- Flexible handling of missing data
- Merge and other relational operations found in popular databases (SQL-based, for example)

| 特点             | NumPy (array)                                    | pandas                                                |
| ---------------- | ------------------------------------------------ | ----------------------------------------------------- |
| **数据结构**     | `ndarray`（多维数组）                            | `DataFrame`（二维表格数据结构），`Series`（一维数据） |
| **数据类型限制** | 必须是同一数据类型（例如全是数值）               | 支持多种数据类型（每列可以不同）                      |
| **灵活性**       | 主要是数值计算，更适合进行矩阵运算和线性代数操作 | 灵活，支持复杂的数据操作和标签功能                    |
| **操作功能**     | 提供基本的数学和数值计算功能                     | 提供高级的分析功能，适合数据清洗、过滤、分组等        |
| **标签支持**     | 不支持标签，只支持基于整数的索引                 | 支持行列标签（标签化数据）                            |
| **主要用途**     | 数值计算、科学计算、机器学习等                   | 数据清洗、探索性数据分析、表格数据操作等              |

### matplotlib

画图、二维数据可视化。

### 略

# 4 NumPy basics: array and vectorized computation

Here are some of the things you'll find in NumPy:

- ndarray, an efficient multidimensional array providing fast array-oriented arithmetic operations and flexible *broadcasting* capabilities
- Mathematical functions for fast operations on entire arrays of data without having to write loops
- Tools for reading/writing array data to disk and working with memory-mapped files
- Linear algebra, random number generation, and Fourier transform capabilities
- A C API for connecting NumPy with libraries written in C, C++, or FORTRAN

For most data analysis applications, the main areas of functionality I’ll focus on are:

- Fast array-based operations for data munging and cleaning, subsetting and filtering, transformation, and any other kind of computation
- Common array algorithms like sorting, unique, and set operations
- Efficient descriptive statistics and aggregating/summarizing data
- Data alignment and relational data manipulations for merging and joining heterogeneous datasets
- Expressing conditional logic as array expressions instead of loops with `if-elif-else` branches
- Group-wise data manipulations (aggregation, transformation, and function application)

没有时间序列，pandas有。

Numpy相比于loop的运算，要高效得多。

## 4.1 The NumPy ndarray: a multidimensional array object

在一个ndarray中，所有的元素类型必须一样。

### Creating ndarrays

```python
In [22]: data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
In [23]: arr2 = np.array(data2)⭐
In [24]: arr2
Out[24]: 
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

In [25]: arr2.ndim⭐
Out[25]: 2
In [26]: arr2.shape⭐
Out[26]: (2, 4)
In [28]: arr2.dtype⭐
Out[28]: dtype('int64')    

In [32]: np.arange(15)⭐
Out[32]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
    
In [30]: np.zeros((3, 6))⭐
Out[30]: 
array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])

In [31]: np.empty((2, 3, 2))⭐
Out[31]: 
array([[[0., 0.],
        [0., 0.],
        [0., 0.]],
       [[0., 0.],
        [0., 0.],
        [0., 0.]]])
```

❗注意：不要认为 `numpy.empty` 会生成一个全是0的array，有时候它会有一些乱七八糟的数值。只有当你只是想生成一个壳子的时候，才用这个function，其余时候要用`np.zeros`。

#### Table 4.1: Some important NumPy array creation functions

| Function            | Description                                                  |
| :------------------ | :----------------------------------------------------------- |
| `array`             | Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a data type or explicitly specifying a data type; copies the input data by default |
| `asarray`           | Convert input to ndarray, but do not copy if the input is already an ndarray |
| `arange`            | Like the built-in `range` but returns an ndarray instead of a list |
| `ones, ones_like`   | Produce an array of all 1s with the given shape and data type; `ones_like` takes another array and produces a `ones` array of the same shape and data type |
| `zeros, zeros_like` | Like `ones` and `ones_like` but producing arrays of 0s instead |
| `empty, empty_like` | Create new arrays by allocating new memory, but do not populate with any values like `ones` and `zeros` |
| `full, full_like`   | Produce an array of the given shape and data type with all values set to the indicated "fill value"; `full_like` takes another array and produces a filled array of the same shape and data type |
| `eye, identity`     | Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere) |

### Data types for ndarrays

```python
In [34]: arr2 = np.array([1, 2, 3], dtype=np.int32) ⭐
In [36]: arr2.dtype
Out[36]: dtype('int32')
    
In [39]: float_arr = arr2.astype(np.float64) ⭐
In [41]: float_arr.dtype
Out[41]: dtype('float64')
    
In [42]: arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
In [43]: arr
Out[43]: array([ 3.7, -1.2, -2.6,  0.5, 12.9, 10.1])
In [44]: arr.astype(np.int32)⭐
Out[44]: array([ 3, -1, -2,  0, 12, 10], dtype=int32) #还是不要乱转换

In [45]: numeric_strings = np.array(["1.25", "-9.6", "42"], dtype=np.string_)
In [46]: numeric_strings.astype(float)⭐
Out[46]: array([ 1.25, -9.6 , 42.  ]) #还是不要这样做，有时候会减掉小数，不如用pandas

#借用别人的type
In [47]: int_array = np.arange(10)
In [48]: calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
In [49]: int_array.astype(calibers.dtype)⭐
Out[49]: array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

#用type code
In [50]: zeros_uint32 = np.zeros(8, dtype="u4")⭐
```

#### Table 4.2: NumPy data types

| Type                                    | Type code      | Description                                                  |
| :-------------------------------------- | :------------- | :----------------------------------------------------------- |
| `int8, uint8`                           | `i1, u1`       | Signed and unsigned 8-bit (1 byte) integer types             |
| `int16, uint16`                         | `i2, u2`       | Signed and unsigned 16-bit integer types                     |
| `int32, uint32`                         | `i4, u4`       | Signed and unsigned 32-bit integer types                     |
| `int64, uint64`                         | `i8, u8`       | Signed and unsigned 64-bit integer types                     |
| `float16`                               | `f2`           | Half-precision floating point                                |
| `float32`                               | `f4 or f`      | Standard single-precision floating point; compatible with C float |
| `float64`                               | `f8 or d`      | Standard double-precision floating point; compatible with C double and Python `float` object |
| `float128`                              | `f16 or g`     | Extended-precision floating point                            |
| `complex64`, `complex128`, `complex256` | `c8, c16, c32` | Complex numbers represented by two 32, 64, or 128 floats, respectively |
| `bool`                                  | ?              | Boolean type storing `True` and `False` values               |
| `object`                                | O              | Python object type; a value can be any Python object         |
| `string_`                               | S              | Fixed-length ASCII string type (1 byte per character); for example, to create a string data type with length 10, use `'S10'` |
| `unicode_`                              | U              | Fixed-length Unicode type (number of bytes platform specific); same specification semantics as `string_` (e.g., `'U10'`) |

### Arithmetic with NumPy arrays

`+ - * / > < == **`

### Basic indexing and slicing

```python
In [61]: arr = np.arange(10)
In [62]: arr
Out[62]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
In [63]: arr[5]
Out[63]: 5
In [65]: arr[5:8] = 12
In [66]: arr
Out[66]: array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])
    
In [67]: arr_slice = arr[5:8] #这个slice只是arr的view，改了它，arr也会改
In [69]: arr_slice[1] = 12345
In [70]: arr
Out[70]: array([    0,     1,     2,     3,     4,    12, 12345,    12,     8,    9])
    
In [71]: arr_slice[:] = 64
In [72]: arr
Out[72]: array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])
    
In [73]: arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
In [75]: arr2d[0][2]
Out[75]: 3
In [76]: arr2d[0, 2]
Out[76]: 3
    
In [77]: arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
In [81]: arr3d[0] = 42
In [82]: arr3d
Out[82]: array([[[42, 42, 42],
        [42, 42, 42]],
       [[ 7,  8,  9],
        [10, 11, 12]]])
```

❗注意：slicing会让原arr跟着改；如果确实想要一个独立的数据，改了它原arr不会跟着改，可以用`arr[5:8].copy()`

​                 多维array索引的方法只用于array，如果将它转换成list，情况就不一样咯。

* 对2d-array来说，arr2d[2,3]是指3row行4column列

### Boolean indexing

```python
In [104]: names == "Bob"
Out[104]: array([ True, False, False,  True, False, False, False])
In [105]: data[names == "Bob"]
Out[105]: array([[4, 7],
                 [0, 0]])
In [106]: data[names == "Bob", 1:]
Out[106]: array([[7],
                 [0]])
In [107]: data[names == "Bob", 1]
Out[107]: array([7, 0])
    
In [113]: mask = (names == "Bob") | (names == "Will")
In [115]: data[mask]
```

❗注意：用Boolean这种方式选择array是复制，改了它原arr不会跟着改

```python
array([[  4,   7],
       [  0,   2],
       [ -5,   6],
       [  0,   0],
       [  1,   2],
       [-12,  -4],
       [  3,   4]])
In [116]: data[data < 0] = 0⭐
In [117]: data
Out[117]: 
array([[4, 7],
       [0, 2],
       [0, 6],
       [0, 0],
       [1, 2],
       [0, 0],
       [3, 4]])

In [118]: data[names != "Joe"] = 7⭐
In [119]: data
Out[119]: 
array([[7, 7],
       [0, 2],
       [7, 7],
       [7, 7],
       [7, 7],
       [0, 0],
       [3, 4]])
```

### Fancy indexing

```python
array([[0., 0., 0., 0.],
       [1., 1., 1., 1.],
       [2., 2., 2., 2.],
       [3., 3., 3., 3.],
       [4., 4., 4., 4.],
       [5., 5., 5., 5.],
       [6., 6., 6., 6.],
       [7., 7., 7., 7.]])
#用list选择rows
In [123]: arr[[4, 3, 0, 6]]⭐
In [124]: arr[[-3, -5, -7]]⭐
Out[124]: 
array([[5., 5., 5., 5.],
       [3., 3., 3., 3.],
       [1., 1., 1., 1.]])

In [125]: arr = np.arange(32).reshape((8, 4))
In [126]: arr
Out[126]: 
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
#用两个list选择元素组成array
In [127]: arr[[1, 5, 7, 2], [0, 3, 1, 2]]⭐
Out[127]: array([ 4, 23, 29, 10])
#选择又排序
In [128]: arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]⭐
Out[128]: 
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])
##arr[[1, 5, 7, 2], [0, 3, 1, 2]] = 0这样arr的数据会跟着改
##arr_fancy=arr[[1, 5, 7, 2], [0, 3, 1, 2]], arr_fancy = 0这样arr的数据不会跟着改
```

### Transposing arrays and swapping axes

```python
# 转置矩阵Transpose-.T，这是object的属性attribute
In [132]: arr = np.arange(15).reshape((3, 5))
In [133]: arr
Out[133]: 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
In [134]: arr.T⭐
Out[134]: 
array([[ 0,  5, 10],
       [ 1,  6, 11],
       [ 2,  7, 12],
       [ 3,  8, 13],
       [ 4,  9, 14]])
```

```python
#矩阵计算np.dot / @
In [135]: arr = np.array([[0, 1, 0], [1, 2, -2], [6, 3, 2], [-1, 0, -1], [1, 0, 1
]])
In [136]: arr
Out[136]: 
array([[ 0,  1,  0],
       [ 1,  2, -2],
       [ 6,  3,  2],
       [-1,  0, -1],
       [ 1,  0,  1]])
#方式一：
In [137]: np.dot(arr.T, arr)⭐
Out[137]: 
array([[39, 20, 12],
       [20, 14,  2],
       [12,  2, 10]])
#方式二：
In [138]: arr.T @ arr⭐
Out[138]: 
array([[39, 20, 12],
       [20, 14,  2],
       [12,  2, 10]])
```

```python
#交换任意两个轴，也可以用于转置矩阵，这是method
In [139]: arr
Out[139]: 
array([[ 0,  1,  0],
       [ 1,  2, -2],
       [ 6,  3,  2],
       [-1,  0, -1],
       [ 1,  0,  1]])
In [140]: arr.swapaxes(0, 1)⭐
Out[140]: 
array([[ 0,  1,  6, -1,  1],
       [ 1,  2,  3,  0,  0],
       [ 0, -2,  2, -1,  1]])
```

## 4.2 Pseudorandom number generation

```python
#随机生成标准正态分布，生成得快呀~~~
In [141]: samples = np.random.standard_normal(size=(4, 4))⭐
In [142]: samples
Out[142]: 
array([[-0.2047,  0.4789, -0.5194, -0.5557],
       [ 1.9658,  1.3934,  0.0929,  0.2817],
       [ 0.769 ,  1.2464,  1.0072, -1.2962],
       [ 0.275 ,  0.2289,  1.3529,  0.8864]])

#除了用这个正态生成器，还可以用其它的👇
In [147]: rng = np.random.default_rng(seed=12345) #seed是初始状态，用这个rng生成随机数，每次都一样⭐
In [148]: data = rng.standard_normal((2, 3)) #可以复现⭐

In [149]: type(rng)
Out[149]: numpy.random._generator.Generator
```

#### Table 4.3: NumPy random number generator methods

| Method            | Description                                                  |
| :---------------- | :----------------------------------------------------------- |
| `permutation`     | Return a random permutation排列 of a sequence, or return a permuted range |
| `shuffle`         | Randomly permute a sequence in place只能打乱行第一维         |
| `uniform`         | Draw samples from a uniform均匀 distribution                 |
| `integers`        | Draw random integers from a given low-to-high range          |
| `standard_normal` | Draw samples from a normal distribution with mean 0 and standard deviation 1 |
| `binomial`        | Draw samples from a binomial distribution                    |
| `normal`          | Draw samples from a normal (Gaussian) distribution           |
| `beta`            | Draw samples from a beta distribution                        |
| `chisquare`       | Draw samples from a chi-square distribution                  |
| `gamma`           | Draw samples from a gamma distribution                       |
| `uniform`         | Draw samples from a uniform [0, 1) distribution              |

np.random.uniform(0,10,size=(2,3))

| 更常用后者         | `shuffle`            | `permutation`          |
| :----------------- | :------------------- | :--------------------- |
| 是否**原地修改**   | ✅（直接改原数据）    | ❌（返回新的打乱结果）  |
| 是否**保留原数据** | ❌（被改了）          | ✅（原数据不动）        |
| 返回结果类型       | 没返回，直接改       | 返回一个新的副本       |
| 安全性             | 有风险（破坏原数据） | 安全（原数据保持不变） |

## 4.3 Universal functions: fast element-wise array functions

Universal functions是指能够作用于narray对象的每一个元素上，而不是针对ndarray的操作。

```python
In [150]: arr = np.arange(10)
In [151]: arr
Out[151]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#开根
In [152]: np.sqrt(arr)⭐
Out[152]: array([0.    , 1.    , 1.4142, 1.7321, 2.    , 2.2361, 2.4495, 2.6458,       2.8284, 3.    ])
#指数函数
In [153]: np.exp(arr)⭐
Out[153]: array([   1.,2.7183,7.3891, 20.0855,54.5982,148.4132, 403.4288, 1096.6332, 2980.958 , 8103.0839])
#俩array对位位置较大的
In [158]: np.maximum(x, y)⭐
#对位相加
np.add(x,y)⭐
#拆分成整数和小数
In [159]: arr = rng.standard_normal(7) * 5
In [160]: arr
Out[160]: array([ 4.5146, -8.1079, -0.7909,  2.2474, -6.718 , -0.4084,  8.6237])
In [161]: remainder, whole_part = np.modf(arr) ⭐
In [162]: remainder
Out[162]: array([ 0.5146, -0.1079, -0.7909,  0.2474, -0.718 , -0.4084,  0.6237])
In [163]: whole_part
Out[163]: array([ 4., -8., -0.,  2., -6., -0.,  8.])
```

#### Table 4.4: Some unary universal functions一元

| Function                                            | Description                                                  |
| :-------------------------------------------------- | :----------------------------------------------------------- |
| `abs, fabs`                                         | Compute the absolute value element-wise for integer, floating-point, or complex values |
| `sqrt`             square root                      | Compute the square root of each element (equivalent to `arr ** 0.5`) |
| `square`                                            | Compute the square of each element (equivalent to `arr ** 2`) |
| `exp`               e to the x                      | Compute the exponent ex of each element                      |
| `log, log10, log2, log1p`                           | Natural logarithm (base *e*), log base 10, log base 2, and log(1 + x), respectively |
| `sign`                                              | Compute the sign of each element: 1 (positive), 0 (zero), or –1 (negative) |
| `ceil`                                              | Compute the ceiling of each element (i.e., the smallest integer greater than or equal to that number) |
| `floor`                                             | Compute the floor of each element (i.e., the largest integer less than or equal to each element) |
| `rint`                                              | Round elements to the nearest integer, preserving the `dtype` |
| `modf`                                              | Return fractional and integral parts of array as separate arrays |
| `isnan`                                             | Return Boolean array indicating whether each value is `NaN` (Not a Number) |
| `isfinite, isinf`                                   | Return Boolean array indicating whether each element is finite (non-`inf`, non-`NaN`) or infinite, respectively |
| `cos, cosh, sin, sinh, tan, tanh`                   | Regular and hyperbolic trigonometric functions               |
| `arccos, arccosh, arcsin, arcsinh, arctan, arctanh` | Inverse trigonometric functions                              |
| `logical_not`                                       | Compute truth value of `not` `x` element-wise (equivalent to `~arr`) |

#### Table 4.5: Some binary universal functions二元

| Function                                                     | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| `add`              +                                         | Add corresponding elements in arrays                         |
| `subtract`             -                                     | Subtract elements in second array from first array           |
| `multiply`           *                                       | Multiply array elements                                      |
| `divide, floor_divide`         /                             | Divide or floor divide (truncating the remainder)            |
| `power`                  ^                                   | Raise elements in first array to powers indicated in second array |
| `maximum, fmax`                                              | Element-wise maximum; `fmax` ignores `NaN`                   |
| `minimum, fmin`                                              | Element-wise minimum; `fmin` ignores `NaN`                   |
| `mod`            %                                           | Element-wise modulus模量 (remainder of division)             |
| `copysign`                                                   | Copy sign of values in second argument to values in first argument |
| `greater, greater_equal, less, less_equal, equal, not_equal` | Perform element-wise comparison, yielding Boolean array (equivalent to infix operators `>, >=, <, <=, ==, !=`) |
| `logical_and`                                                | Compute element-wise truth value of AND (`&`) logical operation |
| `logical_or`                                                 | Compute element-wise truth value of OR (`|`) logical operation |
| `logical_xor`                                                | Compute element-wise truth value of XOR (`^`) logical operation异或，两个输入不相同时为1，否则为0 |

## 4.4 Array-oriented programming with arrays

```python
#用两个1d-array生产两个2d-array，对应(x,y)对
In [169]: points = np.arange(-5, 5, 0.01) # 100 equally spaced points
In [170]: xs, ys = np.meshgrid(points, points) ⭐
Out[171]: 
array([[-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
        ...,
        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99]]),
 array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
        [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
        [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
        ...,
        [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
        [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
        [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])
In [172]: z = np.sqrt(xs ** 2 + ys ** 2) ⭐ 
    
In [174]: import matplotlib.pyplot as plt
In [175]: plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
In [176]: plt.colorbar()
In [177]: plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
```

### Expressing conditional logic as array operations

```python
# np.where表达条件逻辑
In [165]: xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
In [166]: yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
In [167]: cond = np.array([True, False, True, True, False])
In [170]: result = np.where(cond, xarr, yarr)
In [171]: result
Out[171]: array([ 1.1,  2.2,  1.3,  1.4,  2.5])
    
In [175]: np.where(arr > 0, 2, -2) #大于0的替换为2，小于等于0的替换为-2
In [176]: np.where(arr > 0, 2, arr) # set only positive values to 2
```

### Mathematical and statistical methods

```python
# 最外面的那层axis=0，越往里面axis越大
# mean/sum/std
array([[-1.1082,  0.136 ,  1.3471,  0.0611],
       [ 0.0709,  0.4337,  0.2775,  0.5303],
       [ 0.5367,  0.6184, -0.795 ,  0.3   ],
       [-1.6027,  0.2668, -1.2616, -0.0713],
       [ 0.474 , -0.4149,  0.0977, -1.6404]])

In [194]: arr.mean()
Out[194]: -0.08719744457434529
In [195]: np.mean(arr)
Out[195]: -0.08719744457434529
In [196]: arr.sum()
Out[196]: -1.743948891486906

In [197]: arr.mean(axis=1)
Out[197]: array([ 0.109 ,  0.3281,  0.165 , -0.6672, -0.3709])
In [198]: arr.sum(axis=0)
Out[198]: array([-1.6292,  1.0399, -0.3344, -0.8203])
    
# cumsum累积和/cumprod累积乘积
import numpy as np
a = np.array([1, 2, 3, 4])
np.cumsum(a)  # 或a.cumsum()这是静态方法或者类方法；输出：[ 1  3  6 10]
np.cumprod(a)  # 或a.cumprod()输出：[ 1  2  6 24]
# 多维的
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
In [188]: arr.cumsum(axis=0)
Out[188]: 
array([[ 0,  1,  2],
       [ 3,  5,  7],
       [ 9, 12, 15]])
In [189]: arr.cumprod(axis=1)
Out[189]: 
array([[  0,   0,   0],
       [  3,  12,  60],
       [  6,  42, 336]])
```

#### Table 4.6: Basic array statistical methods

| Method           | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| `sum`            | Sum of all the elements in the array or along an axis; zero-length arrays have sum 0 |
| `mean`           | Arithmetic mean; invalid (returns `NaN`) on zero-length arrays |
| `std, var`       | Standard deviation标准差 and variance方差, respectively      |
| `min, max`       | Minimum and maximum                                          |
| `argmin, argmax` | Indices of minimum and maximum elements, respectively        |
| `cumsum`         | Cumulative sum of elements starting from 0                   |
| `cumprod`        | Cumulative product of elements starting from 1               |

### Methods for boolean arrays

针对布尔值能用的方法

```python
# 带条件语法的
In [190]: arr = np.random.randn(100)
In [191]: (arr > 0).sum() # Number of positive values
Out[191]: 42

# any：是否有True；all：是否都是True
In [192]: bools = np.array([False, False, True, False])
In [193]: bools.any()
Out[193]: True
In [194]: bools.all()
Out[194]: False
```

### Sorting

```python
# 一维
In [195]: arr = np.random.randn(6)
In [196]: arr
Out[196]: array([ 0.6095, -0.4938,  1.24  , -0.1357,  1.43  , -0.8469])
In [197]: arr.sort() #从小到大
In [198]: arr
Out[198]: array([-0.8469, -0.4938, -0.1357,  0.6095,  1.24  ,  1.43  ])

#二维
In [199]: arr = np.random.randn(5, 3)
Out[200]: 
array([[ 0.6033,  1.2636, -0.2555],
       [-0.4457,  0.4684, -0.9616],
       [-1.8245,  0.6254,  1.0229],
       [ 1.1074,  0.0909, -0.3501],
       [ 0.218 , -0.8948, -1.7415]])
In [201]: arr.sort(1) #arr.sort(axis=1)
Out[202]: 
array([[-0.2555,  0.6033,  1.2636],
       [-0.9616, -0.4457,  0.4684],
       [-1.8245,  0.6254,  1.0229],
       [-0.3501,  0.0909,  1.1074],
       [-1.7415, -0.8948,  0.218 ]])

# np.sort()
In [221]: arr2 = np.array([5, -10, 7, 1, 0, -3])
In [222]: sorted_arr2 = np.sort(arr2)
Out[223]: array([-10,  -3,   0,   1,   5,   7])
In [203]: large_arr = np.random.randn(1000)
In [204]: large_arr.sort()
In [205]: large_arr[int(0.05 * len(large_arr))] # 5% quantile
Out[205]: -1.5311513550102103
```

### Unique and other set logic

```python
# np.unique 针对一维数组，找到数组中的唯一值并返回已经排序的结果
In [206]: names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
In [207]: np.unique(names)
Out[207]: 
array(['Bob', 'Joe', 'Will'],
      dtype='<U4')
In [208]: ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
In [209]: np.unique(ints)
Out[209]: array([1, 2, 3, 4])

# np.in1d 测试一个数组中是值是否包含另一个数组中相同的值，返回bool
In [211]: values = np.array([6, 0, 0, 3, 2, 5, 6])
In [212]: np.in1d(values, [2, 3, 6])
Out[212]: array([ True, False, False,  True,  True, False,  True], dtype=bool)
```

#### Table 4.7: Array set operations

| Method              | Description                                                  |
| :------------------ | :----------------------------------------------------------- |
| `unique(x)`         | Compute the sorted, unique elements in `x`                   |
| `intersect1d(x, y)` | Compute the sorted, common elements 交集 in `x` and `y`      |
| `union1d(x, y)`     | Compute the sorted union 并集 of elements                    |
| `in1d(x, y)`        | Compute a Boolean array indicating whether each element of `x` is contained in `y` |
| `setdiff1d(x, y)`   | Set difference, elements in `x` that are not in `y`          |
| `setxor1d(x, y)`    | Set symmetric differences对称差集; elements that are in either of the arrays, but not both |

## 4.5 File input and output with arrays

```python
# np.save()/np.load()保存文件和加载文件
In [213]: arr = np.arange(10)
In [214]: np.save('some_array', arr)
In [215]: np.load('some_array.npy')
Out[215]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
# np.savez()将多个数组保存在一个未压缩文件中/np.load()
In [216]: np.savez('array_archive.npz', a=arr, b=arr)
In [217]: arch = np.load('array_archive.npz')
In [218]: arch['b']
Out[218]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
# np.savez_compressed()压缩文件
In [219]: np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)
In [220]: arch = np.load('array_compressed.npz')
```

## 4.6 Linear algebra

点乘 （*是对位的乘法，不一样哦~）

```python
# 点乘np.dot()
In [223]: x = np.array([[1., 2., 3.], [4., 5., 6.]])
In [224]: y = np.array([[6., 23.], [-1, 7], [8, 9]])
In [227]: x.dot(y)
Out[227]: array([[  28.,   64.], [  67.,  181.]])
In [228]: np.dot(x, y)
Out[228]: array([[  28.,   64.], [  67.,  181.]])
    
# 点乘@
In [230]: x @ np.ones(3)
Out[230]: array([  6.,  15.])
```

如果想做其它的线性代数操作，需要numpy.linalg

```python
In [231]: from numpy.linalg import inv, qr

In [232]: X = np.random.randn(5, 5)
In [233]: mat = X.T.dot(X) #X转置点乘X，这样mat是一个方阵，可能有逆矩阵

In [234]: inv(mat) #逆矩阵
Out[234]: array([[  933.1189,   871.8258, -1417.6902, -1460.4005,  1782.1391],
                [  871.8258,   815.3929, -1325.9965, -1365.9242,  1666.9347],
                [-1417.6902, -1325.9965,  2158.4424,  2222.0191, -2711.6822],
                [-1460.4005, -1365.9242,  2222.0191,  2289.0575, -2793.422 ],
                [ 1782.1391,  1666.9347, -2711.6822, -2793.422 ,  3409.5128]])

In [235]: mat.dot(inv(mat)) #方阵点乘它的逆矩阵得到下面这个，验证了确实是求逆矩阵
Out[235]: array([[ 1.,  0., -0., -0., -0.],
                [-0.,  1.,  0.,  0.,  0.],
                [ 0.,  0.,  1.,  0.,  0.],
                [-0.,  0.,  0.,  1., -0.],
                [-0.,  0.,  0.,  0.,  1.]])

In [236]: q, r = qr(mat)
In [237]: r
Out[237]: 
array([[-1.6914,  4.38  ,  0.1757,  0.4075, -0.7838],
       [ 0.    , -2.6436,  0.1939, -3.072 , -1.0702],
       [ 0.    ,  0.    , -0.8138,  1.5414,  0.6155],
       [ 0.    ,  0.    ,  0.    , -2.6445, -2.1669],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.0002]])
```

#### Table 4.8: Commonly used numpy.linalg functions

| Function | Description                                                  |
| :------- | :----------------------------------------------------------- |
| `diag`   | Return the diagonal (or off-diagonal) elements of a square matrix as a 1D array, or convert a 1D array into a square matrix with zeros on the off-diagonal 提取方阵对角线为数列，或数列转为对角矩阵 |
| `dot`    | Matrix multiplication                                        |
| `trace`  | Compute the sum of the diagonal elements 计算迹（对角线之和） |
| `det`    | Compute the matrix determinant 计算行列式                    |
| `eig`    | Compute the eigenvalues and eigenvectors of a square matrix 计算特征值和特征向量 |
| `inv`    | Compute the inverse of a square matrix 逆矩阵                |
| `pinv`   | Compute the Moore-Penrose pseudoinverse of a matrix 伪逆矩阵 |
| `qr`     | Compute the QR decomposition QR分解                          |
| `svd`    | Compute the singular value decomposition (SVD) 奇异值分解    |
| `solve`  | Solve the linear system Ax = b for x, where A is a square matrix |
| `lstsq`  | Compute the least-squares solution to `Ax = b`               |

## 4.7 Example: Random Walks

随机漫步，可能+1可能-1

```python
#直接漫步5000次，每次1000步
nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis=1)
```

然后计算穿越30或者-30的最小穿越时间

```python
#不是每次漫步都达到了30，所以先看看达到过30的漫步次数
In [266]: hits30 = (np.abs(walks) >= 30).any(1) #axis=1的轴上是否有True
In [267]: hits30
Out[267]: array([False,  True, False, ..., False,  True, False], dtype=bool)
In [268]: hits30.sum() # Number that hit 30 or -30
Out[268]: 3410
```

```python
In [269]: crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
In [270]: crossing_times.mean()
Out[270]: 498.88973607038122
```

# 5 Getting Started with pandas

pandas是本书后续内容的首选库。它含有使数据清洗和分析工作变得更快更简单的数据结构和操作工具。

pandas经常和其它工具一同使用，如数值计算工具NumPy和SciPy，分析库statsmodels和scikit-learn，和数据可视化库matplotlib。pandas是基于NumPy数组构建的，特别是基于数组的函数和不使用for循环的数据处理。

虽然pandas采用了大量的NumPy编码风格，但二者最大的不同是pandas是专门为处理表格和混杂数据设计的。而NumPy更适合处理统一的数值数组数据。

```python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
```

## 5.1 Introduction to pandas Data Structures

要使用pandas，你首先就得熟悉它的两个主要数据结构：Series和DataFrame。

### 5.1.1 Series

Series是一种类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成。

```python
#一组最简单的Series
In [11]: obj = pd.Series([4, 7, -5, 3])
In [12]: obj
Out[12]: 
0    4
1    7
2   -5
3    3
dtype: int64
```

这个看似没有索引，但其实已经自动给指定了0~N-1的索引

```python
In [13]: obj.values
Out[13]: array([ 4,  7, -5,  3])

In [14]: obj.index  # like range(4)
Out[14]: RangeIndex(start=0, stop=4, step=1)
```

也可以明确指定索引：

```python
In [15]: obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
In [16]: obj2
Out[16]: 
d    4
b    7
a   -5
c    3
dtype: int64
In [17]: obj2.index
Out[17]: Index(['d', 'b', 'a', 'c'], dtype='object')
In [18]: obj2['a']
Out[18]: -5
In [19]: obj2['d'] = 6
In [20]: obj2[['c', 'a', 'd']]
Out[20]: 
c    3
a   -5
d    6
dtype: int64
```

这样就特别符合平时用于深度学习的数据形式了。

* 一些运算：使用NumPy函数或类似NumPy的运算（如根据布尔型数组进行过滤、标量乘法、应用数学函数等）都会保留索引值的链接：

```python
In [21]: obj2[obj2 > 0]
Out[21]: 
d    6
b    7
c    3
dtype: int64

In [22]: obj2 * 2
Out[22]:
d    12
b    14
a   -10
c     6
dtype: int64

In [23]: np.exp(obj2)
Out[23]: 
d     403.428793
b    1096.633158
a       0.006738
c      20.085537
dtype: float64
```

有数据有索引，其实是很像字典的，实际上也确实可以通过字典创建Series：

```python
In [26]: sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
In [27]: obj3 = pd.Series(sdata)
In [28]: obj3
Out[28]: 
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64
    
#除了这种一一对应的关系，实际上也可以基于这个字典创建不太一样的Series
In [29]: states = ['California', 'Ohio', 'Oregon', 'Texas']
In [30]: obj4 = pd.Series(sdata, index=states)
In [31]: obj4
Out[31]: 
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64
```

在任何算法的预处理中，数据缺失的处理都是必要的，Series有查找缺失的方法isnull/isna和notnull/notna

```python
In [32]: pd.isna(obj4)
Out[32]: 
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool

In [33]: pd.notna(obj4)
Out[33]: 
California    False
Ohio           True
Oregon         True
Texas          True
dtype: bool
    
#实例方法，两个都可以
In [34]: obj4.isna()
Out[34]: 
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool
```

对于许多应用而言，Series最重要的一个功能是，它会根据运算的索引标签自动对齐数据：

```python
In [35]: obj3
Out[35]: 
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64

In [36]: obj4
Out[36]: 
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64

In [37]: obj3 + obj4
Out[37]: 
California         NaN
Ohio           70000.0
Oregon         32000.0
Texas         142000.0
Utah               NaN
dtype: float64
```

以下略

### 5.1.2 DataFrame

#### Table 5.1: Possible data inputs to the DataFrame constructor

| Type                                   | Notes                                                        |
| :------------------------------------- | :----------------------------------------------------------- |
| 2D ndarray                             | A matrix of data, passing optional row and column labels     |
| Dictionary of arrays, lists, or tuples | Each sequence becomes a column in the DataFrame; all sequences must be the same length |
| NumPy structured/record array          | Treated as the “dictionary of arrays” case                   |
| Dictionary of Series                   | Each value becomes a column; indexes from each Series are unioned together to form the result’s row index if no explicit index is passed |
| Dictionary of dictionaries             | Each inner dictionary becomes a column; keys are unioned to form the row index as in the “dictionary of Series” case |
| List of dictionaries or Series         | Each item becomes a row in the DataFrame; unions of dictionary keys or Series indexes become the DataFrame’s column labels |
| List of lists or tuples                | Treated as the “2D ndarray” case                             |
| Another DataFrame                      | The DataFrame’s indexes are used unless different ones are passed |
| NumPy MaskedArray                      | Like the “2D ndarray” case except masked values are missing in the DataFrame result |

### 5.1.3 Index Objects

#### Table 5.2: Some Index methods and properties

| Method/Property  | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| `append()`       | Concatenate with additional Index objects, producing a new Index |
| `difference()`   | Compute set difference as an Index                           |
| `intersection()` | Compute set intersection                                     |
| `union()`        | Compute set union                                            |
| `isin()`         | Compute Boolean array indicating whether each value is contained in the passed collection |
| `delete()`       | Compute new Index with element at Index `i` deleted          |
| `drop()`         | Compute new Index by deleting passed values                  |
| `insert()`       | Compute new Index by inserting element at Index `i`          |
| `is_monotonic`   | Returns `True` if each element is greater than or equal to the previous element |
| `is_unique`      | Returns `True` if the Index has no duplicate values          |
| `unique()`       | Compute the array of unique values in the Index              |

| 方法名             | 含义                                                         |
| ------------------ | ------------------------------------------------------------ |
| **append()**       | 将一个或多个 Index 对象连接起来，返回一个新的 Index。例如：`idx1.append(idx2)`。注意，这不会去重。 |
| **difference()**   | 计算两个 Index 的差集，返回在当前 Index 中但不在传入的 Index 中的元素。例如：`idx1.difference(idx2)` |
| **intersection()** | 计算两个 Index 的交集，即两个都有的值。                      |
| **union()**        | 计算两个 Index 的并集，即所有唯一的值。                      |
| **isin()**         | 判断当前 Index 中的每个元素是否包含在另一个集合中，返回布尔数组。 |
| **delete()**       | 删除指定位置（索引）的元素，返回新的 Index。例如：`idx.delete(2)` 删除第3个元素。 |
| **drop()**         | 删除指定的值（不是位置），返回新的 Index。例如：`idx.drop(['a', 'b'])`。 |
| **insert()**       | 在指定位置插入一个新元素，返回新的 Index。例如：`idx.insert(1, 'new')`。 |
| **unique()**       | 返回去重后的唯一值数组。相当于 NumPy 中的 `np.unique()`。    |

## 5.2 Essential Functionality

### 5.2.1 Reindexing

#### Table 5.3: `reindex` function arguments

| Argument     | Description                                                  |
| :----------- | :----------------------------------------------------------- |
| `labels`     | New sequence to use as an index. Can be Index instance or any other sequence-like Python data structure. An Index will be used exactly as is without any copying. |
| `index`      | Use the passed sequence as the new index labels.             |
| `columns`    | Use the passed sequence as the new column labels.            |
| `axis`       | The axis to reindex, whether `"index"` (rows) or `"columns"`. The default is `"index"`. You can alternately do `reindex(index=new_labels)` or `reindex(columns=new_labels)`. |
| `method`     | Interpolation (fill) method; `"ffill"` fills forward, while `"bfill"` fills backward. |
| `fill_value` | Substitute value to use when introducing missing data by reindexing. Use `fill_value="missing"` (the default behavior) when you want absent labels to have null values in the result. |
| `limit`      | When forward filling or backfilling, the maximum size gap (in number of elements) to fill. |
| `tolerance`  | When forward filling or backfilling, the maximum size gap (in absolute numeric distance) to fill for inexact matches. |
| `level`      | Match simple Index on level of MultiIndex; otherwise select subset of. |
| `copy`       | If `True`, always copy underlying data even if the new index is equivalent to the old index; if `False`, do not copy the data when the indexes are equivalent. |

| 参数        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| **labels**  | 新的标签序列，用作索引。可以是一个 `Index` 对象，也可以是 Python 中任何“类似序列”的数据结构（如 list、tuple、array）。如果是 `Index`，将原样使用（不会复制）。⚠️ 在 DataFrame 中通常不直接用这个参数，而是用 `index` 或 `columns`。 |
| **index**   | 用作新行索引的序列。例如 `df.reindex(index=[0, 2, 4])` 会按给定顺序重新排列行，或引入缺失行。 |
| **columns** | 用作新列索引的序列。你可以重排、选择或引入新列。             |
| **axis**    | 指定要重建的轴：`"index"` 表示行，`"columns"` 表示列。默认是 `"index"`。通常推荐用 `index=` 或 `columns=` 显式指定，`axis` 主要用于 Series。 |
| **method**  | 用于填补缺失值的插值方式（对 `NaN` 有效）：`"ffill"`：前向填充（用上一个有效值）；`"bfill"`：后向填充（用下一个有效值），用于时间序列或对齐数据。 |
| **fill_value** | 当新引入标签导致缺失值时，填充用的默认值。例如：`fill_value=0`。如果不指定，默认是 `NaN`。 |
 | **limit** | 控制前向/后向填充时**最多填充的缺失数量**。防止一口气填太多。 |
 | **tolerance** | 在进行前向或后向填充时，允许的“距离差异”范围（主要用于数值型 index 或时间索引）。 |
 | **level** | 如果你使用的是 **MultiIndex**（多级索引），可以通过这个参数指定在哪一级进行匹配。 |
 | **copy** | 是否复制数据。`True` 表示即使新旧索引一样也复制，`False` 表示如果索引未变就复用原数据（节省内存）。 |

### 5.2.2 Dropping Entries from an Axis

### 5.2.3 Indexing, Selection, and Filtering

#### Table 5.4: Indexing options with DataFrame

| Type                  | Notes                                                        |
| :-------------------- | :----------------------------------------------------------- |
| `df[column]`          | Select single column or sequence of columns from the DataFrame; special case conveniences: Boolean array (filter rows), slice (slice rows), or Boolean DataFrame (set values based on some criterion) |
| `df.loc[rows]`        | Select single row or subset of rows from the DataFrame by label |
| `df.loc[:, cols]`     | Select single column or subset of columns by label           |
| `df.loc[rows, cols]`  | Select both row(s) and column(s) by label                    |
| `df.iloc[rows]`       | Select single row or subset of rows from the DataFrame by integer position |
| `df.iloc[:, cols]`    | Select single column or subset of columns by integer position |
| `df.iloc[rows, cols]` | Select both row(s) and column(s) by integer position         |
| `df.at[row, col]`     | Select a single scalar value by row and column label         |
| `df.iat[row, col]`    | Select a single scalar value by row and column position (integers) |
| `reindex` method      | Select either rows or columns by labels                      |

| **表达式**            | **作用**                 | **说明**                                                     |
| --------------------- | ------------------------ | ------------------------------------------------------------ |
| `df[column]`          | 选取列                   | 如果 `column` 是列名，返回一个 **Series**；如果是列名列表，返回 **DataFrame**。特殊情况：👉 如果是布尔数组 `df[bool_array]`，表示按行筛选（filter rows）。👉 如果是切片 `df[1:4]`，表示按行切片（等同于 `df.iloc[1:4]`）。 |
| `df.loc[rows]`        | 按标签选取行             | `rows` 是行标签（如索引名），也可以是列表或切片。例如：`df.loc["a"]`, `df.loc[["a", "b"]]` |
| `df.loc[:, cols]`     | 按标签选取列             | `cols` 是列标签。例如 `df.loc[:, "A"]` 返回 `A` 列；`df.loc[:, ["A", "B"]]` 选取多个列 |
| `df.loc[rows, cols]`  | 按标签选取行和列         | 常见用法。标签必须真实存在。例如：`df.loc["a", "A"]` 取单个值`df.loc[["a", "b"], ["A", "B"]]` 选子集 |
| `df.iloc[rows]`       | 按位置选取行             | 用整数索引。例如：`df.iloc[0]`, `df.iloc[1:3]`               |
| `df.iloc[:, cols]`    | 按位置选取列             | 用整数索引。例如：`df.iloc[:, 0]` 表示第一列                 |
| `df.iloc[rows, cols]` | 按位置选取行和列         | 比如 `df.iloc[0, 1]` 表示第0行第1列的值                      |
| `df.at[row, col]`     | 按标签选取**单个标量值** | 高速方法，不能用于多个值。例如：`df.at["a", "A"]`            |
| `df.iat[row, col]`    | 按位置选取**单个标量值** | 类似于 `.at`，但用整数坐标：`df.iat[0, 1]`                   |
| `df.reindex(...)`     | 用标签重排或对齐         | 创建一个**新 DataFrame**，行/列可以是重排的、缺失的（NaN）或子集 |

### 5.2.4 Arithmetic and Data Alignment

#### Table 5.5: Flexible arithmetic methods

| Method                | Description                     |
| :-------------------- | :------------------------------ |
| `add, radd`           | Methods for addition (+)        |
| `sub, rsub`           | Methods for subtraction (-)     |
| `div, rdiv`           | Methods for division (/)        |
| `floordiv, rfloordiv` | Methods for floor division (//) |
| `mul, rmul`           | Methods for multiplication (*)  |
| `pow, rpow`           | Methods for exponentiation (**) |

| **方法**                     | **说明**         | **等效运算符** | **示例**                                          |
| ---------------------------- | ---------------- | -------------- | ------------------------------------------------- |
| `add()` / `radd()`           | 加法             | `+`            | `s1.add(s2)` ⇔ `s1 + s2`                          |
| `sub()` / `rsub()`           | 减法             | `-`            | `s1.sub(s2)` ⇔ `s1 - s2``s1.rsub(s2)` ⇔ `s2 - s1` |
| `mul()` / `rmul()`           | 乘法             | `*`            | `s1.mul(s2)` ⇔ `s1 * s2`                          |
| `div()` / `rdiv()`           | 除法（普通除法） | `/`            | `s1.div(s2)` ⇔ `s1 / s2``s1.rdiv(s2)` ⇔ `s2 / s1` |
| `floordiv()` / `rfloordiv()` | 向下整除         | `//`           | `s1.floordiv(s2)` ⇔ `s1 // s2`                    |
| `pow()` / `rpow()`           | 幂运算           | `**`           | `s1.pow(s2)` ⇔ `s1 ** s2`                         |

```python
import pandas as pd

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5], index=['b', 'd'])

# 不加 fill_value：缺失位置运算结果为 NaN
print(s1.add(s2))
# 加上 fill_value=0：缺失位置用 0 替代
print(s1.add(s2, fill_value=0))
```

默认是从column，axis=1加减

### 5.2.5 Function Application and Mapping

### 5.2.6 Sorting and Ranking

排名ranking有啥用啊？

#### Table 5.6: Tie-breaking methods with rank

| Method      | Description                                                  |
| :---------- | :----------------------------------------------------------- |
| `"average"` | Default: assign the average rank to each entry in the equal group |
| `"min"`     | Use the minimum rank for the whole group                     |
| `"max"`     | Use the maximum rank for the whole group                     |
| `"first"`   | Assign ranks in the order the values appear in the data      |
| `"dense"`   | Like `method="min"`, but ranks always increase by 1 between groups rather than the number of equal elements in a group |

### 5.2.7 Axis Indexes with Duplicate Labels

## 5.3 Summarizing and Computing Descriptive Statistics

#### Table 5.7: Options for reduction methods

| Method   | Description                                                  |
| :------- | :----------------------------------------------------------- |
| `axis`   | Axis to reduce over; "index" for DataFrame’s rows and "columns" for columns |
| `skipna` | Exclude missing values; `True` by default                    |
| `level`  | Reduce grouped by level if the axis is hierarchically indexed (MultiIndex) |

| 参数     | 说明                                                         | 示例代码                                                     |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `axis`   | 指定在哪个维度上进行聚合： `0` 或 `"index"` 为按列聚合（对每列求值），`1` 或 `"columns"` 为按行聚合 | `df.sum(axis=0)` → 对每一列求和 `df.sum(axis=1)` → 对每一行求和 |
| `skipna` | 是否跳过缺失值 `NaN`，默认 `True`，若设为 `False` 会导致结果为 `NaN` | `df.mean(skipna=False)` → 如果某列有 NaN，该列的均值就是 NaN |
| `level`  | 用于 `MultiIndex` 分组聚合时指定按哪一级索引聚合（用于分层索引的 DataFrame 或 Series） | `df.sum(level=0)` → 在多层索引中，按第一级索引进行聚合（需配合 MultiIndex 使用） |

#### Table 5.8: Descriptive and summary statistics

| Method           | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| `count`          | Number of non-NA values                                      |
| `describe`       | Compute set of summary statistics                            |
| `min, max`       | Compute minimum and maximum values                           |
| `argmin, argmax` | Compute index locations (integers) at which minimum or maximum value is obtained, respectively; not available on DataFrame objects |
| `idxmin, idxmax` | Compute index labels at which minimum or maximum value is obtained, respectively |
| `quantile`       | Compute sample quantile ranging from 0 to 1 (default: 0.5)   |
| `sum`            | Sum of values                                                |
| `mean`           | Mean of values                                               |
| `median`         | Arithmetic median (50% quantile) of values                   |
| `mad`            | Mean absolute deviation from mean value                      |
| `prod`           | Product of all values                                        |
| `var`            | Sample variance of values                                    |
| `std`            | Sample standard deviation of values                          |
| `skew`           | Sample skewness (third moment) of values                     |
| `kurt`           | Sample kurtosis (fourth moment) of values                    |
| `cumsum`         | Cumulative sum of values                                     |
| `cummin, cummax` | Cumulative minimum or maximum of values, respectively        |
| `cumprod`        | Cumulative product of values                                 |
| `diff`           | Compute first arithmetic difference (useful for time series) |
| `pct_change`     | Compute percent changes                                      |

| 方法                    | 描述                                      | 示例（假设 `df = pd.DataFrame({'A': [1, 2, np.nan, 4]})`） |
| ----------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| `count()`               | 非 NaN 元素数量                           | `df['A'].count()` → `3`                                    |
| `describe()`            | 一组汇总统计数据（均值、标准差等）        | `df['A'].describe()`                                       |
| `min()` / `max()`       | 最小值 / 最大值                           | `df['A'].min()` → `1`                                      |
| `argmin()` / `argmax()` | 最小/最大值的**整数位置**（仅Series可用） | `df['A'].argmax()` → `3`                                   |
| `idxmin()` / `idxmax()` | 最小/最大值的**索引标签**                 | `df['A'].idxmax()` → `3`                                   |
| `quantile(q)`           | 分位数（默认0.5为中位数）                 | `df['A'].quantile(0.25)`                                   |
| `sum()`                 | 总和                                      | `df['A'].sum()` → `7.0`                                    |
| `mean()`                | 平均值                                    | `df['A'].mean()` → `2.333`                                 |
| `median()`              | 中位数（50%分位）                         | `df['A'].median()` → `2.0`                                 |
| `mad()`                 | 与均值的**平均绝对差**                    | `df['A'].mad()` → `0.888...`                               |
| `prod()`                | 所有值的乘积                              | `df['A'].prod()` → `8.0`                                   |
| `var()`                 | 方差                                      | `df['A'].var()` → `2.333`                                  |
| `std()`                 | 标准差                                    | `df['A'].std()` → `1.527...`                               |
| `skew()`                | 偏度（第三阶矩）                          | `df['A'].skew()`                                           |
| `kurt()`                | 峰度（第四阶矩）                          | `df['A'].kurt()`                                           |
| `cumsum()`              | 累加和                                    | `df['A'].cumsum()` → `[1.0, 3.0, NaN, 7.0]`                |
| `cummin()` / `cummax()` | 累计最小/最大值                           | `df['A'].cummax()` → `[1.0, 2.0, NaN, 4.0]`                |
| `cumprod()`             | 累积乘积                                  | `df['A'].cumprod()` → `[1.0, 2.0, NaN, 8.0]`               |
| `diff()`                | 一阶差分（常用于时间序列）                | `df['A'].diff()` → `[NaN, 1.0, NaN, 2.0]`                  |
| `pct_change()`          | 百分比变化（时间序列常用）                | `df['A'].pct_change()` → `[NaN, 1.0, NaN, 1.0]`            |

### 5.3.1 Correlation and Covariance相关系数和协方差

### 5.3.2 Unique Values, Value Counts, and Membership

#### Table 5.9: Unique, value counts, and set membership methods

| Method         | Description                                                  |
| :------------- | :----------------------------------------------------------- |
| `isin`         | Compute a Boolean array indicating whether each Series or DataFrame value is contained in the passed sequence of values |
| `get_indexer`  | Compute integer indices for each value in an array into another array of distinct values; helpful for data alignment and join-type operations |
| `unique`       | Compute an array of unique values in a Series, returned in the order observed |
| `value_counts` | Return a Series containing unique values as its index and frequencies as its values, ordered count in descending order |

| 方法名                | 描述                                                         | 示例（假设：`s = pd.Series([1, 2, 2, 3, 3, 3])`）          |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| `isin(values)`        | 判断每个元素是否包含在给定的序列中，返回布尔数组             | `s.isin([2, 3])` → `[False, True, True, True, True, True]` |
| `get_indexer(target)` | 找出目标序列中每个值在当前序列中的位置（整数索引），常用于对齐/连接 | `pd.Index([1, 2, 3]).get_indexer([3, 2, 1])` → `[2, 1, 0]` |
| `unique()`            | 返回唯一值数组，顺序与原序列中首次出现时一致                 | `s.unique()` → `[1, 2, 3]`                                 |
| `value_counts()`      | 返回值的频数统计，结果是以值为索引、频数为值的 `Series`，按频率降序排列 | `s.value_counts()` → `3: 3, 2: 2, 1: 1`                    |

# 6 Data Loading, Storage, and File Formats

## 6.1 Reading and Writing Data in Text Format

#### Table 6.1: Text and binary data loading functions in pandas

| Function         | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| `read_csv`       | Load delimited data from a file, URL, or file-like object; use comma as default delimiter |
| `read_fwf`       | Read data in fixed-width column format (i.e., no delimiters) |
| `read_clipboard` | Variation of `read_csv` that reads data from the clipboard; useful for converting tables from web pages |
| `read_excel`     | Read tabular data from an Excel XLS or XLSX file             |
| `read_hdf`       | Read HDF5 files written by pandas                            |
| `read_html`      | Read all tables found in the given HTML document             |
| `read_json`      | Read data from a JSON (JavaScript Object Notation) string representation, file, URL, or file-like object |
| `read_feather`   | Read the Feather binary file format                          |
| `read_orc`       | Read the Apache ORC binary file format                       |
| `read_parquet`   | Read the Apache Parquet binary file format                   |
| `read_pickle`    | Read an object stored by pandas using the Python pickle format |
| `read_sas`       | Read a SAS dataset stored in one of the SAS system's custom storage formats |
| `read_spss`      | Read a data file created by SPSS                             |
| `read_sql`       | Read the results of a SQL query (using SQLAlchemy)           |
| `read_sql_table` | Read a whole SQL table (using SQLAlchemy); equivalent to using a query that selects everything in that table using `read_sql` |
| `read_stata`     | Read a dataset from Stata file format                        |
| `read_xml`       | Read a table of data from an XML file                        |

