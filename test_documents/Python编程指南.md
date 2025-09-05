# Python编程指南

## Python简介

Python是一种高级、解释型、交互式和面向对象的脚本语言。Python由Guido van Rossum于1989年底发明，第一个公开发行版发行于1991年。

### Python的特点

- **易于学习**：Python有相对较少的关键字，结构简单，语法清晰
- **易于阅读**：Python代码定义更清晰
- **易于维护**：Python的成功在于它的源代码相当容易维护
- **广泛的标准库**：Python的最大的优势之一是丰富的库
- **互动模式**：互动模式的支持，您可以从终端输入执行代码并获得结果的语言
- **可移植**：基于其开放源代码的特性，Python已经被移植到许多平台
- **可扩展**：如果你需要一段运行很快的关键代码，或者是想要编写一些不愿开放的算法，你可以使用C或C++完成那部分程序
- **数据库**：Python提供所有主要的商业数据库的接口
- **GUI编程**：Python支持GUI可以创建和移植到许多系统调用
- **可嵌入**：你可以将Python嵌入到C/C++程序，让你的程序的用户获得"脚本化"的能力

## 基础语法

### 变量和数据类型

```python
# 数字类型
integer_var = 42
float_var = 3.14
complex_var = 3 + 4j

# 字符串类型
string_var = "Hello, World!"
multiline_string = """这是一个
多行字符串"""

# 布尔类型
bool_var = True

# 列表类型
list_var = [1, 2, 3, "hello", True]

# 元组类型
tuple_var = (1, 2, 3, "hello")

# 字典类型
dict_var = {"name": "Alice", "age": 30, "city": "Beijing"}

# 集合类型
set_var = {1, 2, 3, 4, 5}
```

### 控制结构

#### 条件语句

```python
# if-elif-else语句
age = 18
if age < 18:
    print("未成年")
elif age == 18:
    print("刚成年")
else:
    print("成年人")

# 三元运算符
result = "成年" if age >= 18 else "未成年"
```

#### 循环语句

```python
# for循环
for i in range(5):
    print(f"数字: {i}")

# 遍历列表
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"水果: {fruit}")

# while循环
count = 0
while count < 5:
    print(f"计数: {count}")
    count += 1

# 循环控制
for i in range(10):
    if i == 3:
        continue  # 跳过当前迭代
    if i == 7:
        break     # 跳出循环
    print(i)
```

## 函数

### 函数定义和调用

```python
# 基本函数定义
def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

# 函数调用
message = greet("Alice")
print(message)

# 带默认参数的函数
def greet_with_title(name, title="先生"):
    return f"Hello, {title} {name}!"

# 可变参数函数
def sum_numbers(*args):
    return sum(args)

result = sum_numbers(1, 2, 3, 4, 5)

# 关键字参数函数
def create_profile(**kwargs):
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

user_profile = create_profile(name="Alice", age=30, city="Beijing")
```

### Lambda函数

```python
# Lambda函数（匿名函数）
square = lambda x: x ** 2
print(square(5))  # 输出: 25

# 在高阶函数中使用lambda
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
```

## 面向对象编程

### 类和对象

```python
class Person:
    """人员类"""
    
    # 类变量
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        """构造函数"""
        self.name = name  # 实例变量
        self.age = age
    
    def introduce(self):
        """自我介绍方法"""
        return f"我是{self.name}，今年{self.age}岁"
    
    def have_birthday(self):
        """过生日方法"""
        self.age += 1
        return f"{self.name}现在{self.age}岁了"
    
    @classmethod
    def get_species(cls):
        """类方法"""
        return cls.species
    
    @staticmethod
    def is_adult(age):
        """静态方法"""
        return age >= 18

# 创建对象
person1 = Person("Alice", 25)
person2 = Person("Bob", 17)

# 调用方法
print(person1.introduce())
print(person1.have_birthday())
print(Person.is_adult(person2.age))
```

### 继承

```python
class Student(Person):
    """学生类，继承自Person类"""
    
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # 调用父类构造函数
        self.student_id = student_id
        self.courses = []
    
    def enroll_course(self, course):
        """选课方法"""
        self.courses.append(course)
        return f"{self.name}已选择课程：{course}"
    
    def introduce(self):
        """重写父类方法"""
        base_intro = super().introduce()
        return f"{base_intro}，学号是{self.student_id}"

# 使用继承
student = Student("Charlie", 20, "S001")
print(student.introduce())
print(student.enroll_course("Python编程"))
```

## 异常处理

```python
# 基本异常处理
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"除零错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
else:
    print("没有异常发生")
finally:
    print("无论是否有异常都会执行")

# 抛出自定义异常
class CustomError(Exception):
    """自定义异常类"""
    pass

def validate_age(age):
    if age < 0:
        raise CustomError("年龄不能为负数")
    if age > 150:
        raise CustomError("年龄不能超过150岁")
    return True

try:
    validate_age(-5)
except CustomError as e:
    print(f"验证错误: {e}")
```

## 文件操作

```python
# 文件读写
# 写入文件
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("Hello, World!\n")
    file.write("这是第二行\n")

# 读取文件
with open("example.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)

# 逐行读取
with open("example.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())

# JSON文件操作
import json

# 写入JSON
data = {"name": "Alice", "age": 30, "city": "Beijing"}
with open("data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

# 读取JSON
with open("data.json", "r", encoding="utf-8") as file:
    loaded_data = json.load(file)
    print(loaded_data)
```

## 模块和包

### 导入模块

```python
# 导入整个模块
import math
print(math.pi)
print(math.sqrt(16))

# 导入特定函数
from math import pi, sqrt
print(pi)
print(sqrt(16))

# 使用别名
import numpy as np
import pandas as pd

# 导入所有（不推荐）
from math import *
```

### 创建自定义模块

```python
# mymodule.py
def add(a, b):
    """加法函数"""
    return a + b

def multiply(a, b):
    """乘法函数"""
    return a * b

PI = 3.14159

# 在其他文件中使用
# from mymodule import add, multiply, PI
```

## 常用内置函数

```python
# 数学函数
print(abs(-5))        # 绝对值
print(round(3.7))     # 四舍五入
print(max([1,2,3]))   # 最大值
print(min([1,2,3]))   # 最小值
print(sum([1,2,3]))   # 求和

# 类型转换
print(int("123"))     # 字符串转整数
print(str(123))       # 整数转字符串
print(float("3.14"))  # 字符串转浮点数
print(list("hello"))  # 字符串转列表

# 序列操作
print(len("hello"))   # 长度
print(sorted([3,1,2])) # 排序
print(reversed([1,2,3])) # 反转

# 高阶函数
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
even = list(filter(lambda x: x%2==0, numbers))
```

## 列表推导式和生成器

```python
# 列表推导式
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x%2 == 0]

# 字典推导式
square_dict = {x: x**2 for x in range(5)}

# 集合推导式
unique_squares = {x**2 for x in range(-5, 6)}

# 生成器表达式
square_gen = (x**2 for x in range(10))

# 生成器函数
def fibonacci(n):
    """斐波那契数列生成器"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用生成器
for num in fibonacci(10):
    print(num)
```

## 装饰器

```python
# 简单装饰器
def timer(func):
    """计时装饰器"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "完成"

# 带参数的装饰器
def repeat(times):
    """重复执行装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")
```

## 常用标准库

### datetime模块

```python
from datetime import datetime, date, timedelta

# 当前时间
now = datetime.now()
today = date.today()

# 格式化时间
formatted = now.strftime("%Y-%m-%d %H:%M:%S")

# 时间计算
tomorrow = today + timedelta(days=1)
last_week = today - timedelta(weeks=1)
```

### os模块

```python
import os

# 文件和目录操作
print(os.getcwd())  # 当前工作目录
os.listdir('.')    # 列出目录内容
os.path.exists('file.txt')  # 检查文件是否存在
os.path.join('path', 'to', 'file')  # 路径拼接
```

### random模块

```python
import random

# 随机数生成
print(random.random())      # 0-1之间的随机浮点数
print(random.randint(1, 10)) # 1-10之间的随机整数
print(random.choice(['a', 'b', 'c']))  # 随机选择

# 列表随机操作
my_list = [1, 2, 3, 4, 5]
random.shuffle(my_list)     # 随机打乱
print(random.sample(my_list, 3))  # 随机采样
```

## 最佳实践

### 代码风格（PEP 8）

1. **缩进**：使用4个空格进行缩进
2. **行长度**：每行不超过79个字符
3. **命名规范**：
   - 变量和函数：小写字母，用下划线分隔
   - 类名：首字母大写的驼峰命名
   - 常量：全大写字母，用下划线分隔
4. **导入**：标准库、第三方库、本地模块分组导入
5. **空行**：类和函数定义前后使用空行分隔

### 性能优化

1. **使用内置函数**：内置函数通常比自定义函数更快
2. **列表推导式**：比传统循环更高效
3. **生成器**：处理大数据时节省内存
4. **避免全局变量**：局部变量访问更快
5. **使用适当的数据结构**：字典查找比列表查找快

### 调试技巧

```python
# 使用print调试
def debug_function(x):
    print(f"输入值: {x}")
    result = x * 2
    print(f"结果: {result}")
    return result

# 使用assert断言
def divide(a, b):
    assert b != 0, "除数不能为零"
    return a / b

# 使用logging模块
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def logged_function(x):
    logger.debug(f"处理输入: {x}")
    result = x ** 2
    logger.info(f"计算完成: {result}")
    return result
```

## 总结

Python是一门功能强大且易于学习的编程语言。掌握以上基础知识后，你可以：

1. 编写基本的Python程序
2. 使用面向对象编程
3. 处理文件和异常
4. 使用标准库
5. 遵循最佳实践

继续学习的方向包括：
- Web开发（Django、Flask）
- 数据科学（NumPy、Pandas、Matplotlib）
- 机器学习（Scikit-learn、TensorFlow、PyTorch）
- 自动化脚本
- 网络爬虫