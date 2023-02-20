from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F


class HyperNet(nn.Module):
    """
    Hyper Network for Content Understanding

    Args:
        目前所有的参数都是复制原来的算法，没有进行自己的改动。
        lda_out_channels: 16
    """

    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size,
                 target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet, self).__init__()

        # 需要修改
        self.lda_out_channels = lda_out_channels  # 16, 16
        self.hyperInChn = hyper_in_channels  # 112 , 96
        self.target_in_size = target_in_size  # 224, 192
        self.f1 = target_fc1_size  # 112, 96
        self.f2 = target_fc2_size  # 56, 48
        self.f3 = target_fc3_size  # 28, 24
        self.f4 = target_fc4_size  # 14, 12
        self.feature_size = feature_size  # 7 -- > 6

        self.vit = vit_base_patch16_224_in21k(num_classes=2, has_logits=False)  # .to(device)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for vit output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(197 * 3, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        )

        # Hyper network
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

    def forward(self, img):
        feature_size = self.feature_size
        vit_out = self.vit(img)

        # input vector for target net
        target_in_vec = vit_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)  # B, 224 --> B, 192

        # input features for hyper net, vit模型的输出，将结果传递给超网络，从而生成目标网络的参数
        # 经过卷积后hyper_in_feat: B, 112, 7, 7
        # [B, 197, 768] --conv-> [B, hyperInChn, 768] --view-> [B, hyperInChn_96, 6, 6]
        hyper_in_feat = self.conv1(vit_out['hyper_in_feat'].view(-1, 197 * 3, 16, 16)).view(-1, self.hyperInChn,
                                                                                            feature_size,
                                                                                            feature_size)
        # 生成目标网络的weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        # 最后的一层
        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)  # 1

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.paras = paras
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )

        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )
        # self.num_classes = 2
        # self.l5_head = nn.Linear(in_features=12, out_features=2)
        # # self.l5_head.weight.data = paras['target_fc5w']
        # # self.l5_head.bias.data = paras['target_fc5b']
        # self.l5w = paras['target_fc5w'].squeeze()
        # self.l5b = paras['target_fc5b'].squeeze()
        # self.l5o = torch.zeros((len(paras['target_fc5b']), 2)).cuda()
        # self.l5bb = torch.zeros(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return self.sigmoid(q)
        # q = self.l4(q).squeeze()  # 8, 12
        # # print("q shape:", q.shape, q.squeeze().shape, type(q), self.l5_head.weight.data.shape)  # 8, 12, 1, 1
        # for i in range(q.shape[0]):  # 8
        #     self.l5_head.weight.data = self.l5w[i]
        #     if len(self.l5b.shape) == 0:
        #         continue
        #     self.l5bb = self.l5b[i].item()
        #     self.l5_head.bias.data = torch.tensor(self.l5bb)
        #     self.l5o[i] = self.l5_head(q[i])
        # return self.l5o


class TargetFC(nn.Module):
    """
    目标网络的全连接层
    在一个batch中的权重和偏置不同，故使用组卷积
    """

    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        # print('input_re:', input_.shape, input_re.shape)
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2],
                                     self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        # print(weight_re.shape, bias_re.shape)
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])
        # print("out:", out.shape)

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # nn.Identity() 占位符
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels    通道
            num_classes (int): number of classes for classification head    类 2
            embed_dim (int): embedding dimension    嵌入维度 120
            depth (int): depth of transformer   层数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU  # 激活层

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        # print('shape:', self.blocks)
        # start here
        self.blocks_1 = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(3)
        ])
        self.blocks_2 = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(3)
        ])
        self.blocks_3 = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(3)
        ])
        self.blocks_4 = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(3)
        ])

        # local distortion aware module
        # 按照hyperIQA，将一个特征图转化为1个值（不严谨）
        in_channels = 197 * 3  # 还要转化为特征图，16*16*3
        out_channels = 32
        lda_out_channels = 16
        in_chn = 192
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        )
        # feature_size ^ 2 * out_channels --> 6*6*32 = 1152
        self.lda1_fc = nn.Linear(1152, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        )
        self.lda2_fc = nn.Linear(1152, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
            # nn.AvgPool2d(kernel_size=5, stride=3, padding=1)
        )
        self.lda3_fc = nn.Linear(1152, lda_out_channels)

        self.lda4_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        )
        self.lda4_fc = nn.Linear(1152, in_chn - lda_out_channels * 3)  # 1152, 176 --> 512 144

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        # x = self.blocks(x)
        # 将每一阶段的结果先做处理，Local Distortion Aware Module, 然后去cat
        B, H, _ = x.shape
        x = self.blocks_1(x)
        lda_1 = self.lda1_fc(self.lda1_pool(x.view(B, H * 3, 16, 16)).view(x.size(0), -1))  # B, 16
        x = self.blocks_2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x.view(B, H * 3, 16, 16)).view(x.size(0), -1))  # B, 16
        x = self.blocks_3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x.view(B, H * 3, 16, 16)).view(x.size(0), -1))  # B, 16
        x = self.blocks_4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x.view(B, H * 3, 16, 16)).view(x.size(0), -1))  # B, 48
        # x = self.norm(x_4)  # norm(x)
        # print(x_1.shape, x_2.shape, x_3.shape, x_4.shape, x.shape)

        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)  # B, 96
        # 由于不使用分类头，故先去掉
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]
        out = {}
        out['hyper_in_feat'] = x  # B,197*768
        out['target_in_vec'] = vec
        return out

    def forward(self, x):
        x = self.forward_features(x)
        # 没有使用transformer的分类头，所以去掉了这里
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = vit_base_patch16_224_in21k(num_classes=2, has_logits=False).to(device)
    from torchsummary import summary

    ckpt = torch.load('vit_base_patch16_224.pth')
    # del_keys = ['head.weight', 'head.bias', 'pre_logits', 'pre_logits.weight', 'pre_logits.bias']
    # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #    else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    # for k in del_keys:
    #    ckpt.pop(k)
    if 'head.bias' in list(ckpt.keys()):
        ckpt.pop('head.bias')
    ckpt.pop('head.weight')
    model.load_state_dict(ckpt, strict=False)
    k = torch.rand(8)
    print(k)
    print(nn.Sigmoid()(k))
