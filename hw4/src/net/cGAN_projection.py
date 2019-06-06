import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
from src.net.conditional_batchnorm2d import ConditionalBatchNorm2d
from src.net.utils import get_label_id


class Generator(nn.Module):
    def __init__(self, noise_dim, base_acti_maps, dropout_rate=0.5):
        super(Generator, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Sequential(
            # input size: [B x noise_dim]
            (nn.ConvTranspose2d(noise_dim, base_acti_maps * 2 ** 5, 4, 1, 0, bias=False)),
            # nn.BatchNorm2d(base_acti_maps * 2 ** 5),
            # nn.ReLU(inplace=True),
            # state size: [B x base * 2^5 x 4 x 4]
        )
        self.conv2 = nn.Sequential(
            (nn.ConvTranspose2d(base_acti_maps * 2 ** 5 + 4, base_acti_maps * 2 ** 4, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(base_acti_maps * 2 ** 4),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_rate),
            # state size: [B x base * 2^4 x 8 x 8]
        )
        self.conv3 = nn.Sequential(
            (nn.ConvTranspose2d(base_acti_maps * 2 ** 4 + 4, base_acti_maps * 2 ** 3, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(base_acti_maps * 2 ** 3),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_rate),
            # state size: [B x base * 2^3 x 16 x 16]
        )
        self.conv4 = nn.Sequential(
            (nn.ConvTranspose2d(base_acti_maps * 2 ** 3 + 4, base_acti_maps * 2 ** 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(base_acti_maps * 2 ** 2),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_rate),
            # state size: [B x base * 2^2 x 32 x 32]
        )
        self.conv5 = nn.Sequential(
            (nn.ConvTranspose2d(base_acti_maps * 2 ** 2 + 4, base_acti_maps * 2 ** 1, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(base_acti_maps * 2 ** 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_rate),
            # state size: [B x base * 2^1 x 64 x 64]
        )
        self.conv6 = nn.Sequential(
            (nn.ConvTranspose2d(base_acti_maps * 2 ** 1, 3, 4, 2, 1, bias=False)),
            # state size: [B x 3 x 128 x 128]
            nn.Tanh()
        )
        self.h_embed = torch.randn(6, 16, requires_grad=False).cuda()
        self.e_embed = torch.randn(4, 16, requires_grad=False).cuda()
        self.f_embed = torch.randn(3, 16, requires_grad=False).cuda()
        self.g_embed = torch.randn(2, 16, requires_grad=False).cuda()

        self.upsample_mode = 'bilinear'

        # num_classes = 144
        # self.bn1 = ConditionalBatchNorm2d(base_acti_maps * 2 ** 5 + 4, num_classes)
        # self.bn2 = ConditionalBatchNorm2d(base_acti_maps * 2 ** 4 + 4, num_classes)
        # self.bn3 = ConditionalBatchNorm2d(base_acti_maps * 2 ** 3 + 4, num_classes)
        # self.bn4 = ConditionalBatchNorm2d(base_acti_maps * 2 ** 2 + 4, num_classes)
        # self.bn5 = ConditionalBatchNorm2d(base_acti_maps * 2 ** 1, num_classes)
        self.bn1 = nn.BatchNorm2d(base_acti_maps * 2 ** 5 + 4)
        self.bn2 = nn.BatchNorm2d(base_acti_maps * 2 ** 4 + 4)
        self.bn3 = nn.BatchNorm2d(base_acti_maps * 2 ** 3 + 4)
        self.bn4 = nn.BatchNorm2d(base_acti_maps * 2 ** 2 + 4)
        self.bn5 = nn.BatchNorm2d(base_acti_maps * 2 ** 1)

    def forward(self, noises, hair, eye, face, glass):
        label_id = get_label_id(hair, eye, face, glass)
        hair = torch.mm(hair, self.h_embed)
        eye = torch.mm(eye, self.e_embed)
        face = torch.mm(face, self.f_embed)
        glass = torch.mm(glass, self.g_embed)

        state = torch.cat((noises,hair, eye, face, glass), 1)
        state = state.view(state.size(0), -1, 1, 1)

        hair = hair.view(hair.size(0), 1, 4, 4)
        eye = eye.view(hair.size(0), 1, 4, 4)
        face = face.view(hair.size(0), 1, 4, 4)
        glass = glass.view(hair.size(0), 1, 4, 4)

        state = self.conv1(state)
        state = torch.cat((state, hair, eye, face, glass), 1)
        # state = self.bn1(state, label_id)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.dropout2d(state, p=self.dropout_rate)

        state = self.conv2(state)
        state = torch.cat((state,
                           F.interpolate(hair, size=8, mode=self.upsample_mode),
                           F.interpolate(eye, size=8, mode=self.upsample_mode),
                           F.interpolate(face, size=8, mode=self.upsample_mode),
                           F.interpolate(glass, size=8, mode=self.upsample_mode),
                           ), 1)
        # state = self.bn2(state, label_id)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.dropout2d(state, p=self.dropout_rate)

        state = self.conv3(state)
        state = torch.cat((state,
                           F.interpolate(hair, size=16, mode=self.upsample_mode),
                           F.interpolate(eye, size=16, mode=self.upsample_mode),
                           F.interpolate(face, size=16, mode=self.upsample_mode),
                           F.interpolate(glass, size=16, mode=self.upsample_mode),
                           ), 1)
        # state = self.bn3(state, label_id)
        state = self.bn3(state)
        state = F.relu(state)
        # state = F.dropout2d(state, p=self.dropout_rate)

        state = self.conv4(state)
        state = torch.cat((state,
                           F.interpolate(hair, size=32, mode=self.upsample_mode),
                           F.interpolate(eye, size=32, mode=self.upsample_mode),
                           F.interpolate(face, size=32, mode=self.upsample_mode),
                           F.interpolate(glass, size=32, mode=self.upsample_mode),
                           ), 1)
        # state = self.bn4(state, label_id)
        state = self.bn4(state)
        state = F.relu(state)
        state = F.dropout2d(state, p=self.dropout_rate)

        state = self.conv5(state)
        # state = self.bn5(state, label_id)
        state = self.bn5(state)
        state = F.relu(state)
        # state = F.dropout2d(state, p=self.dropout_rate)

        state = self.conv6(state)
        return state


class Discriminator(nn.Module):
    def __init__(self, base_acti_maps):
        super(Discriminator, self).__init__()
        self.lrelu_slope = 0.2
        self.conv = nn.Sequential(
            # input size: [B x 3 x 128 x 128]
            spectral_norm(nn.Conv2d(3, base_acti_maps * 2 ** 0, 4, 2, 1, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 0 x 64 x 64]
            spectral_norm(nn.Conv2d(base_acti_maps * 2 ** 0, base_acti_maps * 2 ** 1, 4, 2, 1, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 1 x 32 x 32]
            spectral_norm(nn.Conv2d(base_acti_maps * 2 ** 1, base_acti_maps * 2 ** 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 2 x 16 x 16]
            spectral_norm(nn.Conv2d(base_acti_maps * 2 ** 2, base_acti_maps * 2 ** 3, 4, 2, 1, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 3 x 8 x 8]
            spectral_norm(nn.Conv2d(base_acti_maps * 2 ** 3, base_acti_maps * 2 ** 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 4 x 4 x 4]
            spectral_norm(nn.Conv2d(base_acti_maps * 2 ** 4, base_acti_maps * 2 ** 4, 4, 1, 0, bias=False)),
            nn.LeakyReLU(self.lrelu_slope, inplace=True),
            # state size: [B x base * 2 ** 4 x 1 x 1]
            # nn.AdaptiveAvgPool2d(1)
        )
        self.img_head = nn.Sequential(spectral_norm(nn.Linear(base_acti_maps * 2 ** 4 + 15, 1)))

        self.hair_head = nn.Sequential(spectral_norm(nn.Linear(base_acti_maps * 2 ** 4, 6)), nn.Softmax(1))
        self.eye_head = nn.Sequential(spectral_norm(nn.Linear(base_acti_maps * 2 ** 4, 4)), nn.Softmax(1))
        self.face_head = nn.Sequential(spectral_norm(nn.Linear(base_acti_maps * 2 ** 4, 3)), nn.Softmax(1))
        self.glass_head = nn.Sequential(spectral_norm(nn.Linear(base_acti_maps * 2 ** 4, 2)), nn.Softmax(1))

    def forward(self, img, hairs, eyes, faces, glasses):
        # img = img + torch.randn_like(img) / 5
        img = self.conv(img)
        img = img.view(img.size(0), -1)
        # img_vector size: [B x 16]
        img_act = self.img_head(torch.cat((img, torch.cat((hairs, eyes, faces, glasses), 1)), 1))

        hair_act = self.hair_head(img)
        eye_act = self.eye_head(img)
        face_act = self.face_head(img)
        glass_act = self.glass_head(img)
        return img_act, hair_act, eye_act, face_act, glass_act
