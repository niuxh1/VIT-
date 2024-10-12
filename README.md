# VIT-
借鉴参考https://github.com/yxhuang7538/vit-pytorch ，致谢@yxhuang7538，在他的代码基础上提供了大量解析，方便初学者参考，暂时只有模型，将补充训练结果

补充更新：已完成训练，超参数为img_size = 224 patch_size = 16 embed_dim = 768  depth = 12  num_heads = 12  num_classes = 10  batch_size = 64  num_epochs = 6 learning_rate = 0.01
在google colab的t4 gpu训练约两个半小时，在ciafy—10测试集上表现在可接受范围内，约为55%以上，证明了复现模型结构的正确性，欢迎有能力者在更大的训练集上复现


