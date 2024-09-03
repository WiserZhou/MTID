import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ComplexLinearActivationNet(nn.Module):
    def __init__(self, horizon, action_dim, hidden_dim=128, activation_fn=nn.ReLU()):
        super(ComplexLinearActivationNet, self).__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.activation_fn = activation_fn

        # 更复杂的网络结构：增加隐藏层维度，增加层数
        self.a_layer_1 = nn.Linear(1, hidden_dim)  # 第一层
        self.a_layer_2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层
        self.a_layer_3 = nn.Linear(hidden_dim, horizon * action_dim)  # 第三层

        self.b_layer_1 = nn.Linear(1, hidden_dim)  # 第一层
        self.b_layer_2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层
        self.b_layer_3 = nn.Linear(hidden_dim, horizon * action_dim)  # 第三层

        # 初始化权重：使用Kaiming初始化
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.a_layer_1.weight)
        nn.init.constant_(self.a_layer_1.bias, 0)
        nn.init.kaiming_normal_(self.a_layer_2.weight)
        nn.init.constant_(self.a_layer_2.bias, 0)
        nn.init.kaiming_normal_(self.a_layer_3.weight)
        nn.init.constant_(self.a_layer_3.bias, 0)
        
        nn.init.kaiming_normal_(self.b_layer_1.weight)
        nn.init.constant_(self.b_layer_1.bias, 0)
        nn.init.kaiming_normal_(self.b_layer_2.weight)
        nn.init.constant_(self.b_layer_2.bias, 0)
        nn.init.kaiming_normal_(self.b_layer_3.weight)
        nn.init.constant_(self.b_layer_3.bias, 1)

    def forward(self, k):
        k = k.view(-1, 1)  # 将k转换成列向量

        # 生成a的参数
        a = self.activation_fn(self.a_layer_1(k))
        a = self.activation_fn(self.a_layer_2(a))
        a = self.a_layer_3(a)

        # 生成b的参数
        b = self.activation_fn(self.b_layer_1(k))
        b = self.activation_fn(self.b_layer_2(b))
        b = self.b_layer_3(b)

        # 计算更复杂的a*k + b
        out = a * (k**2) + b  # 使用k的平方

        # 通过激活函数
        out = self.activation_fn(out)
        
        # 将输出重塑为(horizon, action_dim)
        return out.view(-1, self.horizon, self.action_dim)



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 初始化网络
horizon = 3
action_dim = 105
net = ComplexLinearActivationNet(horizon, action_dim, activation_fn=nn.ReLU()).to(device)


SIZE = 18
# 生成示例数据
k = torch.arange(1, SIZE+1).float().to(device)  # 输入的自然数k

# 网络的前向传播
output = net(k)
print(output)

# 定义损失函数和优化器
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 示例训练代码
# 假设我们有目标矩阵target，形状也是(horizon, action_dim)
# target = torch.ones(1, horizon, action_dim)  # 目标矩阵
target = torch.rand(SIZE,horizon,action_dim).to(device)
print(target)
# 训练循环
for epoch in tqdm(range(100000)):  # 训练1000个epoch
    optimizer.zero_grad()  # 清空梯度
    output = net(k)  # 前向传播
    print(output)
    loss = criterion(output, target)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 优化

    # if epoch % 100 == 0:
        # print(f'Epoch {epoch}, Loss: {loss.item()}')

output = net(k)
print(output)

# import math
print(output-target)

# print(math.fabs(output-target))