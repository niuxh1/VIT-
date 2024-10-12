from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    '''
    图像处理层，将图像分割为小块，方便特征提取
    '''

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        '''
        当前图片大小
        '''
        patch_size = (patch_size, patch_size)
        '''
        图像分割为小块的大小
        '''
        self.image_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        '''
        图像分割为小块的数量，即网格大小
        '''
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        '''
        分割后的patch数量
        '''
        self.embed_dim = self.patch_size[0] * self.patch_size[1]
        '''
        块的嵌入维度，即展平后的维度
        '''
        self.cnn = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        '''
        卷积层，提取特征，缩小维度，将3rgb通道转换为embed_dim，一个网格提取一个特征
        '''
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        '''
        归一化层，防止梯度爆炸
        '''

    def forward(self, x):
        '''
        CNN: (x,3,224,224) -> (x,768,14,14)
        flatten: (x,768,14,14) -> (x,768,196)
        transpose: (x,768,196) -> (x,196,768)
        '''
        x=self.cnn(x).flatten(2).transpose(1, 2)
        '''
        将3通道特征转换为embed_dim个特征，每个特征提取一个网格即14*13个像素
        展平，方便后续处理，
        转置，方便后续处理
        '''
        return x


def _init_vit_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        '''
        初始化全连接层权重
        '''
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            '''
            初始化全连接层偏置
            '''
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        '''
        初始化卷积层权重
        '''
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            '''
            初始化卷积层偏置
            '''
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        '''
        设置归一化层权重为1与偏置为0
        '''


class Block(nn.Module):
    '''
    最重要的Block层，包含自注意力层与全连接层，实现自注意力机制和特征提取
    '''

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qkv_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0,
                 drop_out_ratio=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        '''
        批归一化，防止梯度爆炸
        '''
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qkv_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        ''''
        自注意力层，提取特征
        '''
        self.drop_out = nn.Dropout(p=drop_out_ratio)
        '''
        随机丢失神经元，防止过拟合
        '''
        self.norm2 = norm_layer(dim)
        '''
        批归一化，防止梯度爆炸
        '''
        mlp_hidden_dim =int( dim * mlp_ratio)
        '''
        全连接层隐藏层维度
        '''
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop_out=drop_ratio)
        '''
        全连接层，输出特征
        '''

    def forward(self, x):
        x = x + self.drop_out(self.attn(self.norm1(x)))
        '''
        形成自注意力层后实现随即丢失
        通过残差（跳跃链接），防止梯度消失，模型退化
        '''
        x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    '''
    transformer(本文中为VIT)的核心，自注意力层
    '''

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0,
                 proj_drop_ratio=0):
        super().__init__()
        self.num_heads = num_heads
        '''
        自注意力头个数，在不同维度学习信息，便于学习不同特征
        '''
        self.head_num = dim // num_heads
        '''
        每个头的维度，dim除以num_heads,平局分配维度
        '''
        self.scale = qk_scale or self.head_num ** -0.5
        '''
        没有特殊设置，使用默认设置sqrt(head_num)^-1
        '''
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        '''
        每个qkv含dim个维度，共3个qkv
        '''
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        '''
        注意力层丢失
        '''
        self.proj = nn.Linear(dim, dim)
        '''
        输出层
        '''
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        '''
        输出层丢失
        '''

    def forward(self, x):
        B, N, C = x.shape
        '''
        x 大小 (B,14*14,768)
        b=Batch_size
        n=patch_size
        c=embed_dim
        '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_num).permute(2, 0, 3, 1, 4)
        '''
        qkv * x=(B,14*14,768*3)
        qkv.reshape: (B,14*14,768*3) -> (B,14*14,3,8,96)
        qkv.permute: (B,14*14,3,8,96) -> (3 (3个维度，代表qkv),B(对应每个x批次)，8(8个头，8个维度的特征),14*14(每个patch),96(每个头的维度))

        '''
        q, k, v = qkv[0], qkv[1], qkv[2]
        '''
        q,k,v=(B,num_heads,N,head_num)
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        '''
        qk点积，计算q,k的相似度，得到注意力分布
        '''
        '''
        attn=(B,num_heads,N,N)
        '''
        attn = attn.softmax(dim=-1)
        '''
        softmax归一化，得到注意力的概率分布
        '''

        attn = self.attn_drop(attn)
        '''
        随机丢失神经元，防止过拟合
        '''
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        '''
        attn 与 v点积，得到注意力的输出结果
        attn=(B,num_heads,N,N)，v=(B,num_heads,N,head_num)
        attn @ v=(B,num_heads,N,head_num)
        transpose(1,2)：(B,num_heads,N,head_num) -> (B,N,num_heads,head_num)
        reshape: (B,N,num_heads,head_num) -> (B,N,C)
        '''
        '''
        x=(B,N,C)
        '''
        x = self.proj(x)
        '''
        输出层
        '''
        x = self.proj_drop(x)
        '''
        随机丢失神经元，防止过拟合
        '''
        return x


class Mlp(nn.Module):
    '''
    多层感知机，提取特征，输出结果
    '''

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop_out=0.,
                 act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        '''
        输出特征默认为输入特征
        '''
        hidden_features = hidden_features or in_features
        '''
        全连接层隐藏层默认为输入特征
        '''
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        '''
        非线性激活函数，默认为GELU，效果优于RELU
        '''
        self.fc2 = nn.Linear(hidden_features, out_features)
        '''
        输出
        '''
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=6,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qkv_scale=None,
                 representation_size=None,
                 drop_ratio=0,
                 attn_drop_ratio=0,
                 embed_layer=PatchEmbed):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        '''
        记录特征维度
        '''

        self.num_tokens = 1
        '''
        特殊token
        '''
        norm_layer = partial(nn.LayerNorm)
        """
        默认使用LayerNorm
        """
        act_layer = nn.GELU
        '''
        默认使用GELU，效果优于RELU
        '''
        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_c,
                                       embed_dim=embed_dim,
                                       norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        '''
        cls_token=(1,1,768)
        得到全局 特征
        '''
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        '''
        定义嵌入位置，保证时间与逻辑有序
        pos_embed=(1,14*14+1,768)
        '''
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qkv_scale=qkv_scale,
                  drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio,
                  act_layer=act_layer,
                  norm_layer=norm_layer) for _ in range(depth)
        ])
        '''
        堆叠多个encoder，保证提取特征
        '''
        self.norm = norm_layer(embed_dim)
        '''
        归一化，防止梯度爆炸
        '''
        if representation_size:
            self.is_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', act_layer())
            ]))

        else:
            self.is_logits = False
            self.pre_logits = nn.Identity()
        '''
        预表达层
        降低维度
        方便进一步分类
        如不进行降维
        标记为false，不进行操作
        '''
        self.head = nn.Linear(self.num_features, num_classes)
        '''
        分类层，
        将特征映射到类别
        '''
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        '''
        初始化位置嵌入，类别嵌入
        '''
        self.apply(_init_vit_weight)
        '''
        递归遍历模型，
        引入自定义函数初始化权重
        '''

    def forward(self, x):
        '''
        开始前向传播
        '''
        x = self.patch_embed(x)
        '''
        数据预处理为patch
        方便后续处理
        '''
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        '''
        记录全局特征
        '''
        x = torch.cat((cls_tokens, x), dim=1)
        '''
        拼接全局特征与输入特征，同时考虑类别与图像块
        '''
        x = self.pos_drop(x + self.pos_embed)
        '''
        拼接位置信息保证时间与逻辑有序，
        随机丢失神经元，防止过拟合
        '''
        x = self.blocks(x)
        '''
        进入自注意力层，开始提取特征
        '''
        x = self.norm(x)
        '''
        防止梯度爆炸
        '''
        x = self.pre_logits(x[:, 0])
        '''
        开始预表达
        从每个x提取第一个token,即类别token,
        方便后续分类
        '''
        x = self.head(x)
        '''
        分类完成
        '''
        return x


if __name__ == '__main__':
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=5)
    print(model)
    '''
    打印模型
    '''