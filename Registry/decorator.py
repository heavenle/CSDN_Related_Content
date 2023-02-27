from functools import wraps

def decorator(func):
    @wraps(func)
    #@wraps(func) 等同于 wrapper = wraps(func)(wrapper)
    def wrapper(arr1, arr2):
        print("实现新的功能,例如数据加1的功能")
        arr1 += 1
        arr2 += 1
        return func(arr1, arr2)
    return wrapper

def add(arr1, arr2):
    return arr1 + arr2

print(add.__name__)

add = decorator(add)
print(add.__name__)
