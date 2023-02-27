
def outside():
    # 此处obj为自由变量
    obj = []
    # inside就被称为闭包函数
    def inside(name):
        obj.append(name)
        print(obj)
    return inside

Lis = outside()
Lis("1")
Lis("2")
Lis("3")
print("-"*20)
# 闭包函数和闭包函数之间的自由变量不会相互影响。
Lis2 = outside()
Lis2("4")
Lis2("5")
Lis2("6")
