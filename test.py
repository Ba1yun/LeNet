import torch
from Net import MyLeNet5
from torch.autograd import Variable  #autograd下的子库，是用来储存时刻变化的变量，最适合反向传播时候不断跟新的权重
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 将数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载训练数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

#  如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，如果GPU可用则将模型转到GPU
model = MyLeNet5().to(device)

# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load("save_model.pth"))

# 获取预测结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把tensor转成Image， 方便可视化
#show = ToPILImage()

# 进入验证阶段
model.eval()
# 对test_dataset里10000张手写数字图片进行推理
# for i in range(len(test_dataloader)):
for i in range(5):
    x, y = test_dataset[i][0], test_dataset[i][1]
    #print("x:",x,"y:",y) x:张量数据，y:数字0-9之间的一个数
    #print("大小：",x.size())  #:[1 28 28]
    # tensor格式数据可视化
    #show(x).show()  #show = ToPILImage()和这一句都运行的话就是可以把图片显示出来
    # 扩展张量维度为4维.float()必须要不然没用，暂时不知道原因
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)#torch.unsqueeze(x, dim=0)加维度，看54行代码是三维这里加成4维
    #print(x.size()) [1,1,28,28]
    with torch.no_grad():
        pred = model(x)
        #print(pred)
        #print(pred[0][0])#这里得到的是一个一维数据1是每次的图片数所以这里的结果就只有一个信息：概率值
        #print(pred.size())  #torch.Size([1, 10])，因为每次都是拿1个每一个都有10个概率值所以是一维张量因此这里的10可以确定是概率
        # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的哪一类标签
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        #predicted, actual = torch.argmax(pred[0]),y,这个和上一个语句一样的作用，因为索引值就是这个数
        # 最终输出的预测值与真实值
        print(f'predicted: "{predicted}", actual:"{actual}"')


'''
1.torch.unsqueeze(x, dim=0),展开在尺寸dim=0的地方加一个维度
2.torch.argmax(pred[0])，torch.argmax（）只返回最大值的索引值，torch.max返回最大值和索引值
'''