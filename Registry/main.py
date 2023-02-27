
class Registry:
    def __init__(self, name=None):
        # 生成注册列表的名字, 如果没有给出，则默认是Registry。
        if name == None:
            self._name = "Registry"
        self._name = name
        #创建注册字典。
        self._obj_list = {}

    def __registry(self, obj):
        """
        内部注册函数
        :param obj:函数或者类的地址。
        :return:
        """
        #判断是否目标函数或者类已经注册，如果已经注册过则标错，如果没有则进行注册。
        assert(obj.__name__ not in self._obj_list.keys()), "{} 已经在 {} 中完成注册".format(obj.__name__, self._name)
        self._obj_list[obj.__name__] = obj

    def registry(self, obj=None):
        """
        # 外部注册函数。注册方法分为两种。
        # 1.通过装饰器调用
        # 2.通过函数的方式进行调用

        :param obj: 函数或者类的地址。
        :return:
        """
        # 1.通过装饰器调用
        if obj == None:
            def _no_obj_registry(func__or__class, *args, **kwargs):
                self.__registry(func__or__class)
                # 此时被装饰的函数会被修改为该函数的返回值。

                return func__or__class


            return _no_obj_registry
        #2.通过函数的方式进行调用
        self.__registry(obj)

    def get(self, name):
        """
        通过字符串name获取对应的函数或者类。
        :param name: 函数或者类的名称
        :return: 对应的函数或者类
        """
        assert (name in self._obj_list.keys()), "{}  没有注册".format(name)
        return self._obj_list[name]


# 生成注册表
REGISTRY_LIST = Registry("REGISTRY_LIST")

# 通过装饰器调用
@REGISTRY_LIST.registry()
# @REGISTRY_LIST.registry() 可以转化为下面语句
# test_by_decorator = REGISTRY_LIST.registry()(test_by_decorator) = _no_obj_registry(test_by_decorator)
def create_by_decorator():
    print("通过装饰器完成注册的函数")


def create_by_function():
    print("直接通过registry函数进行注册")
REGISTRY_LIST.registry(create_by_function)


test1 = REGISTRY_LIST.get("create_by_decorator")
test1()
test2 = REGISTRY_LIST.get("create_by_function")
test2()