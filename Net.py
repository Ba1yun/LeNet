import torch
from torch import nn
#定义一个网络模型
class MyLeNet5(nn.Module):    #Module目的是继承Module当中的东西
#初始化网络
    def __init__(self):    #构造函数，放置在这里面的都是该模型的固有属性
        super (MyLeNet5,self).__init__()  #调用父类的构造函数
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.Sigmoid=nn.Sigmoid()
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten=nn.Flatten()  #没有输入参数那么就是默认从1展开到最后一维，1实际上是第二维开始，因为都是从0开始算起
        self.f6=nn.Linear(120,84)
        self.output=nn.Linear(84,10)

    def forward(self,x):    #定义自己的网络模型必须有forward,这是建立网络模型各个层之间的联系，就是所谓的前向传播
        x=self.Sigmoid(self.c1(x))
        x=self.s2(x)
        x=self.Sigmoid(self.c3(x))
        x=self.s4(x)
        x=self.c5(x)
        x=self.flatten(x)    #展平，就是强行将多维展平为多少维，这里为全连接，展开成1*1*x
        x=self.f6(x)
        x=self.output(x)
        return x
if __name__=="__main__":    #可写可不写，注意格式（顶行不能在class类下边不然会认为没有MyLeNet5这个类）
    x=torch.randn(1,1,28,28)
    Module=MyLeNet5()
    Y=Module(x)

#Conv2d 是大写C  nn.的时候没有出可能是没写对,nn库下边首字母都是大写