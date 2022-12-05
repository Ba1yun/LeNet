'''
model:
导入模块基本上是：torch  nn
class MyLeNet5(nn.Module):
    def __init__(self):
        super (MyLeNet5,self).__init__() 这三句是定义一个网络模型必有的，继承基模块的属性
        ……这里是定义的神经网络层
        def forward(self,x):定义前向传播

'''

'''
train:
导入模块：torch  nn MyLeNet5 lr_scheduler datasets transforms DataLoader

数据转换
train和val数据集操作[下载数据集、定义数据集加载器]
定义device：device="cuda" if torch.cuda.is_available() else "cpu"
将模型传到device上：model=MyLeNet5().to(device)

定义损失函数[CrossEntropyLoss]
定义优化器[SGD（也就是参数跟新的方法）]
定义学习率[StepLR]

创建（定义）训练（train）函数，里面参数[数据加载器、网络模型、损失函数、优化器]
    损失值、精准度=0
    遍历数据集：enumerate(dataloader)获得图片的数据、标签
    送去device：X,y=X.to(device),y.to(device)
    输出：output=model(X)
    计算损失值
    获得该图片被认为最大概率是什么类别
    计算准确值[没一个batch_size中预测和实际相等的数量/batch_size]
    
    #更新参数
    梯度清零[没迭代一次就要清零一次]
    梯度清零之后开始链式反向传播
    优化器优化[到了指定位置就会跟新一次参数]
    然后是计算损失和精确值
    
val:
    跟train类似但是没有跟新参数的步骤，上一步不是训练好了一轮有一组网络参数已经保存好了，这一次就是再拿出一些图片放到模型当中用保存好的
    参数进行一次测试得到损失和精准
    
迭代：
就是迭代多少次，每一次都有哪些操作，保存最好的一次模型[按照精准度最高的来判断最好，先保存第一次的模型，看看下一次的精度是不是比这个高如果高则保存
不高则替换以保存的网络参数]

测试：
前面的跟训练一样：
导入库函数
数据转换
检测数据集
设备
把模型送到设备[model = MyLeNet5().to(device)]
加载权重
分类名称
循环导入图片
获得图片的两个属性（数据tensor，真实标签）
不设置计算梯度[with torch.no_grad():]
把数据送入模型预测得到output（batch、概率）
找到最大概率并返回索引
输出结果

'''