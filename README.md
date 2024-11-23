# VIT-
 致谢@yxhuang7538，在他的代码基础上提供了大量解析，方便初学者参考
v0.1
补充更新：已完成训练，超参数为img_size = 224 patch_size = 16 embed_dim = 768  depth = 12  num_heads = 12  num_classes = 10  batch_size = 64  num_epochs = 6 learning_rate = 0.01
在google colab的t4 gpu训练约两个半小时，在ciafy—10测试集上表现在可接受范围内，约为55%以上，证明了复现模型结构的正确性，欢迎有能力者在更大的训练集上复现

v1.0
实现了在训练集上99%以上的正确率，测试集表现不佳，现已发行pth文件，超参数更新为
img_size = 224
patch_size = 16
embed_dim = 512
depth = 8
num_heads = 16
num_classes = 100  # CIFAR-100 有100个类
batch_size = 128
num_epochs = 150 # 增加训练轮数
learning_rate = 0.01

