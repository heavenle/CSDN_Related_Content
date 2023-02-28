import torch
import torch.nn as nn


class AttentionPoolWithParameter(nn.Module):
    """
    带参数的注意力汇聚实现方法
    """
    def __init__(self):
        super(AttentionPoolWithParameter, self).__init__()
        # 可学习的参数w。
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, q:"(1, 50)", k:"(1, 50)", v:"(1, 50)"):
        """
        实现方法。
        :param q: 查询, tensor(1,dim)
        :param k: 键, tensor(1,dim)
        :param v: 值, tensor(1,dim)
        :return: 注意力权重和值的加权和, tensor(1,dim)
        """
        #通过复制将q的维度，扩展为(dim,dim)，方便计算
        q = q.repeat_interleave(k.shape[1]).reshape(-1, k.shape[1])
        attention = torch.softmax(-((q - k) * self.w)**2/2, dim=1)
        return torch.bmm(attention.unsqueeze(0), v.unsqueeze(-1)).reshape(1, -1)


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, pred, y_true):
        return self.loss(pred, y_true)