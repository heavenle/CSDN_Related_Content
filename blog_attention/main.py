import numpy as np
import matplotlib.pyplot as plt
from learn_attention_pool import CE, AttentionPoolWithParameter
import torch
import copy


def show_heapmap(query_x, x_train):
    """
    计算注意力机制图。
    :param query_x: 查询, 类型numpy(dim,)
    :param x_train: 键， 类型numpy(dim,)
    :return:注意力机制图，类型numpy(dim, dim)
    """
    heapmap = []
    for i in range(len(x_train)):
        heapmap.append(softmax(-(query_x[i] - x_train)**2/2))
    heapmap = np.array(heapmap)
    return heapmap


def f(x):
    """
    通过 y=sin(x)^2+x^0.8+\epsilon 获取所需的训练数据。
    :param x:训练数据x, 类型numpy(dim,)
    :return: 训练数据y, 类型numpy(dim,)
    """
    return np.sin(x)*2 + x**0.8


def softmax(x):
    """
    计算softmax值
    :param x: 输入数据x, 类型numpy(dim,)
    :return: softmax值， 类型numpy(dim,)
    """
    return np.exp(x)/np.sum(np.exp(x))


def average_pool(y_train):
    """
    平均汇聚层，实现 f(x)=\frac{1}{n}\sum^n_{i=1}y_i
    :param y_train:输入数据y_train，类型numpy(dim,)
    :return:numpy(公式的输出值 * dim)，类型numpy(dim,)
    """
    return np.array([np.sum(y_train)/len(y_train)]*len(y_train))


def attention_pool(query_x, key, value):
    """
    非参数的注意力汇聚层的实现方法。
    $f(x)=\sum^n_{i=1}softmax(exp(-\frac{1}{2}(x-x_i)^2))y_i$

    :param query_x:查询， 类型numpy(dim,)
    :param key:键，类型numpy(dim,)
    :param value:值，类型numpy(dim,)
    :return: 注意力汇聚的加权和，类型numpy(dim)。query_x中的元素，都是该元素通过该计算key的权重，和value的加权和。
    """
    for i in range(len(value)):
        query_x[i] = np.sum(np.dot(softmax(-(query_x[i] - key)**2/2), value))
    return query_x


def main():
    # ----------------------------------------------------------------------
    # 生成数据
    # ----------------------------------------------------------------------
    # 生成训练数据和标签。
    x_train = np.sort(np.random.rand(50)) * 6
    y_train = f(x_train) + np.random.normal(0, 0.5, 50)
    # 生成测试数据和真实标签。
    x_test = np.arange(0, 6.28, 0.12566)
    y_true = f(x_test)
    # 绘制图像
    plt.figure(1)
    l1 = plt.scatter(x_train, y_train, color="r")
    l2, = plt.plot(x_test, y_true, color="b")
    plt.legend(handles=[l1, l2], labels=["train_data", "sin_function"], loc="best")
    plt.savefig("data.png")

    # ----------------------------------------------------------------------
    # 平均汇聚方法
    # ----------------------------------------------------------------------
    average_function = average_pool(y_train)
    l3, = plt.plot(x_train, average_function, color="g")
    plt.legend(handles=[l1, l2, l3], labels=["train_data", "sin_function", "average_function"], loc="best")
    plt.savefig("average_function.png")

    # ----------------------------------------------------------------------
    # 非参数的注意力汇聚方法
    # ----------------------------------------------------------------------
    query_x = copy.deepcopy(x_test)
    sf_attebtiob_function = attention_pool(query_x, x_train, y_train)
    l4, = plt.plot(x_train, sf_attebtiob_function, color="black")
    plt.legend(handles=[l1, l2, l3, l4], labels=["train_data", "sin_function", "average_function", "sf_attention_function"], loc="best")
    plt.savefig("sf_average_function.png")
    # plt.show()
    # 生成注意力机制图。
    heap_map = show_heapmap(x_test, x_train)
    plt.figure(2)
    plt.imshow(heap_map)
    plt.xlabel("x_train")
    plt.ylabel("query_x")
    plt.savefig("heapmap_no_param.png")

    # ----------------------------------------------------------------------
    # 带参数的注意力汇聚方法
    # ----------------------------------------------------------------------
    net = AttentionPoolWithParameter()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss = CE()

    x_test = torch.tensor(x_test.astype(np.float32)).reshape(1, -1)
    y_true = torch.tensor(y_true.astype(np.float32)).reshape(1, -1)
    x_train = torch.tensor(x_train.astype(np.float32)).reshape(1, -1)
    y_train = torch.tensor(y_train.astype(np.float32)).reshape(1, -1)

    net.train()
    for epoch in range(50):
        optimizer.zero_grad()
        y_pred = net(x_test, x_train, y_train)
        l = loss(y_pred, y_true)
        l.sum().backward()
        optimizer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    print(net.w)
    net.eval()
    with torch.no_grad():
        y_pred = net(x_test, x_train, y_train)
        plt.figure(1)
        l5, = plt.plot(x_test.squeeze(), y_pred.squeeze(), color="pink")
        plt.legend(handles=[l1, l2, l3, l4, l5],
               labels=["train_data", "sin_function", "average_function",
                       "sf_attention_function", "sf_attention_with_params_function"], loc="best")
        plt.savefig("sf_average_with_params_function.png")

    heap_map = show_heapmap(x_test.squeeze().numpy(), y_pred.squeeze().numpy())
    plt.figure(3)
    plt.imshow(heap_map)
    plt.xlabel("x_train")
    plt.ylabel("query_x")
    plt.savefig("heapmap_param.png")
    plt.show()

if __name__ == "__main__":
    main()
