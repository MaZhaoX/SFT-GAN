import torch
import torch.nn as nn
from torch.autograd import Variable

import Probabilistic_Weight as pw
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from dysample import DySample
import torch.fft

num_pw = 0
NUM_BANDS = 6
PATCH_SIZE = 256
fc1 = 8 * NUM_BANDS
fc2 = 4 * NUM_BANDS
fc3 = 2 * NUM_BANDS
fc4 = NUM_BANDS

def softsign(x):
    return x / (1 + torch.abs(x))
def addweight(x):
    s = 1/(1+torch.exp(-x))
    return 0.5 * s


def butterworth_filter(image, cutoff, order):
    batch_size, channels, height, width = image.shape
    u = torch.arange(0, height, dtype=torch.float32).reshape(-1, 1).repeat(1, width)
    v = torch.arange(0, width, dtype=torch.float32).repeat(height, 1)
    u = u - height // 2
    v = v - width // 2
    D = torch.sqrt(u ** 2 + v ** 2)
    H = 1 / (1 + (D / cutoff) ** (2 * order))
    H = H.to(image.device)
    filtered_image = torch.zeros_like(image)
    for i in range(batch_size):
        for c in range(channels):
            image_fft = torch.fft.fftshift(torch.fft.fft2(image[i, 4]))
            filtered_fft = image_fft * H
            filtered_image[i, c] = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)).real
    return filtered_image

def gaussian_filter(image, sigma):
    batch_size, channels, height, width = image.shape
    u = torch.arange(0, height, dtype=torch.float32).reshape(-1, 1).repeat(1, width)
    v = torch.arange(0, width, dtype=torch.float32).repeat(height, 1)
    u = u - height // 2
    v = v - width // 2
    D = torch.sqrt(u ** 2 + v ** 2)
    H = torch.exp(-(D ** 2) / (2 * (sigma ** 2)))
    H = H.to(image.device)
    filtered_image = torch.zeros_like(image)
    for i in range(batch_size):
        for c in range(channels):
            image_fft = torch.fft.fftshift(torch.fft.fft2(image[i, c]))
            filtered_fft = image_fft * H
            filtered_image[i, c] = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)).real
    return filtered_image

def ideal_low_pass_filter(image, cutoff):
    batch_size, channels, height, width = image.shape
    u = torch.arange(0, height, dtype=torch.float32).reshape(-1, 1).repeat(1, width)
    v = torch.arange(0, width, dtype=torch.float32).repeat(height, 1)
    u = u - height // 2
    v = v - width // 2
    D = torch.sqrt(u ** 2 + v ** 2)
    H = (D <= cutoff).float()
    H = H.to(image.device)
    filtered_image = torch.zeros_like(image)
    for i in range(batch_size):
        for c in range(channels):
            image_fft = torch.fft.fftshift(torch.fft.fft2(image[i, c]))
            filtered_fft = image_fft * H
            filtered_image[i, c] = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)).real
    return filtered_image




class MLP(nn.Module):
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
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads = 8,
                 qkv_bias = False,
                 qk_scale = None,
                 attn_drop = 0.,
                 proj_drop = 0.,
                 sr_rate = 1,
                 apply_transform = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_pw = 0
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.kv = nn.Linear(dim,2*dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_rate
        if sr_rate > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_rate+1, stride=sr_rate, padding=sr_rate//2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads >1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k_ = pw.random_sampling(k, math.floor(math.log(N)))
        s_ = q @ k_.transpose(-2, -1)
        m_ = pw.S_weight(s_)
        _, _, _, w = q.shape
        out, q_, q_index = pw.Top_u(q, m_, math.floor(math.log(N)), w)
        attn1 = torch.matmul(q_, k.transpose(-2, -1))
        attn1 = attn1 * self.scale
        attn = torch.zeros_like(q)
        attn.scatter_(2, q_index, attn1)
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_rate=sr_ratio, apply_transform=apply_transform)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x2 = self.norm1(x)
        x2 = self.attn(x2, H, W)
        x2 = self.drop_path(x2)
        x2 = x + x2

        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        x3 = self.drop_path(x3)
        x4 = x2 + x3

        return x4


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=4, out_ch=768, with_pos=False):
        super().__init__()
        self.path_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size//2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.path_size[0], W // self.path_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=6, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)

        return x


class Stem(nn.Module):
    def __init__(self, in_ch=6, out_ch=64, with_pos=False):
        super(Stem,self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self.with_pos:
            x = self.pos(x)
        return x


class pwT(nn.Module):
    def __init__(self,
                 in_chans=6,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scaale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[2, 2, 2, 2],
                 sr_rations=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm,
                 apply_transform=False):
        super().__init__()
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scaale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_rations[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])
        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scaale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_rations[1],
                  apply_transform=apply_transform)
            for i in range(self.depths[1])])
        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scaale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_rations[2],
                  apply_transform=apply_transform)
            for i in range(self.depths[2])])
        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scaale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_rations[3],
                  apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        f1 = x

        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        f2 = x

        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        f3 = x

        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        f4 = x

        return f1, f2, f3, f4


class lFusion(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.linear = nn.Linear(in_feature, in_feature)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        h = x + y
        B, C, H, W = h.shape
        m = h.view(B, C, -1).permute(0, 2, 1)
        w = self.linear(m)
        w = self.softmax(w)
        w = w.permute(0, 2, 1).view(B, C, H, W)

        h = h * w
        return h


class Fusion(nn.Module):
    def __init__(self, fc1, fc2, fc3, fc4, H = 256, W = 256):
        super().__init__()
        self.cf1 = lFusion(int(H * W / 128))
        self.cf2 = lFusion(int(H * W / 256))
        self.cf3 = lFusion(int(H * W / 512))
        self.cf4 = lFusion(int(H * W / 1024))
        self.up1 = DySample(256)
        self.up2 = DySample(128)
        self.up3 = DySample(64)
        self.up4 = DySample(NUM_BANDS)
        self.up5 = DySample(NUM_BANDS)
        self.cov1 = nn.Conv2d(512, 256, kernel_size=1)
        self.cov2 = nn.Conv2d(512, 128, kernel_size=1)
        self.cov3 = nn.Conv2d(256, 64, kernel_size=1)
        self.cov4 = nn.Conv2d(128, NUM_BANDS, kernel_size=1)

    def forward(self, c1, c2, c3, c4, f1, f2, f3, f4):
        h1 = self.cf1(c4, f4)
        h2 = self.cf2(c3, f3)
        h3 = self.cf3(c2, f2)
        h4 = self.cf4(c1, f1)
        #stage 1
        h1 = self.cov1(h1)
        h1 = self.up1(h1)
        #stage2
        h2 = torch.cat((h1, h2), dim=1)
        h2 = self.cov2(h2)
        h2 = self.up2(h2)
        #stage3
        h3 = torch.cat((h2, h3), dim=1)
        h3 = self.cov3(h3)
        h3 = self.up3(h3)
        #stage4
        h4 = torch.cat((h3, h4), dim=1)
        h4 = self.cov4(h4)
        out = self.up4(h4)
        out = self.up5(out)

        return out


class sAttention(nn.Module):

    def __init__(self, in_chans):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_chans, in_chans * 2, kernel_size=1, bias=False)
        self.relu = nn.SiLU()
        self.fc2 = nn.Conv2d(in_chans * 2, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        x = self.pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * 2
        x = x.expand_as(out)

        return x


class Generator(nn.Module):
    def __init__(self, num_bands = NUM_BANDS, H = 256, W = 256):

        super(Generator, self).__init__()
        self.time_feature = pwT(in_chans=num_bands)
        self.space_feature = pwT(in_chans=num_bands)
        self.fusion = Fusion(fc1, fc2, fc3, fc4, H, W)
        self.spectral_attn = sAttention(NUM_BANDS)

    def forward(self, cc1, cc2, ff1):
        # cc1, cc2, ff1 =torch.split(input, 6, dim=1)
        c = cc1-cc2
        c1, c2, c3, c4 = self.time_feature(c)
        f1, f2, f3, f4 = self.space_feature(ff1)
        ff2 = self.fusion(c1, c2, c3, c4, f1, f2, f3, f4)
        spectral_weight = self.spectral_attn(cc2)
        out = ff2

        b_image1 = butterworth_filter(out, 50, 2)
        b_image2 = butterworth_filter(out, 150, 4)
        b_image3 = butterworth_filter(out, 250, 4)

        # b_image1 = gaussian_filter(out, 1)
        # b_image2 = gaussian_filter(out, 2)
        # b_image3 = gaussian_filter(out, 3)

        # b_image1 = ideal_low_pass_filter(out, 50)
        # b_image2 = ideal_low_pass_filter(out, 150)
        # b_image3 = ideal_low_pass_filter(out, 200)

        d1 = ff1 - b_image3
        d2 = out - b_image2
        d21 = b_image1 - b_image2
        d3 = out - b_image1
        d31 = b_image2 - b_image1
        w1 = 1-0.5 * torch.sign(d1)
        w2 = addweight(d2)
        w3 = addweight(d3)

        out = out + w1 * d1 + w2 * d2 * 0.5 + w3 * d3 * 0.5 + w2 * d21 * 0.5 + w3 * d31 * 0.5
        out = out * spectral_weight

        return out


class Discriminator(nn.Module):
    def __init__(self, C = NUM_BANDS, H = 256, W = 256, conv_dim = 64):
        super().__init__()
        self.channels = C
        curr_H = H // 2
        curr_W = W // 2
        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.LayerNorm([conv_dim, curr_H, curr_W]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim
        curr_H = curr_H // 2
        curr_W = curr_W // 2

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, 2 * curr_dim, 4, 2, 1, bias=False),
            nn.LayerNorm([curr_dim * 2, curr_H, curr_W]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        curr_dim = curr_dim * 2
        curr_H = curr_H // 2
        curr_W = curr_W // 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, 2 * curr_dim, 4, 2, 1),
            nn.LayerNorm([curr_dim * 2, curr_H, curr_W]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        curr_dim = curr_dim * 2
        curr_H = curr_H // 2
        curr_W = curr_W // 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, 2 * curr_dim, 4, 2, 1),
            nn.LayerNorm([curr_dim * 2, curr_H, curr_W]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        curr_dim = curr_dim * 2
        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.last_adv(out)

        return out.squeeze()



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
