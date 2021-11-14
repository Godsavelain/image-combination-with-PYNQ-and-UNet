import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, socket, time
from PIL import Image

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans


def predict_img(net, full_img, device, scale_factor=1.0, out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


net = UNet(n_channels=3, n_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
net.load_state_dict(torch.load('MODEL.pth', map_location=device))


def from_imagepath_to_result(image_path):
    img = Image.open(image_path)
    mask = predict_img(net=net, full_img=img, scale_factor=0.5, out_threshold=0.5, device=device)
    result = Image.fromarray((mask * 255).astype(np.uint8)).resize((img.size[0], img.size[1])).convert('RGB')
    return result


def send_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    len_data = len(data)
    space = len_data
    UDP_server.sendall(space.to_bytes(16, 'big'))
    UDP_server.sendall(data)
    # for i in range(0, space):
    #     start_idx = i * 10000
    #     end_idx = min(len_data, (i+1) * 10000)
    #     UDP_server.sendall(data[start_idx: end_idx])
    #     print('Send {}'.format(end_idx - start_idx))
    return 'Send Image {} OK'.format(image_path)


def recv_image(image_path):
    space = UDP_server.recv(16)
    space = int.from_bytes(space, 'big')
    print('space {}'.format(space))

    if space == 99999 or space == 0:
        return 'close'

    with open(image_path, 'wb') as f:
        while space > 0:
            data = UDP_server.recv(1024)
            # print('data len {}'.format(len(data)))
            space -= len(data)
            f.write(data)
    return 'Recv Image {} OK'.format(image_path)


TCP_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_server.bind(('192.168.31.56', 13579))
TCP_server.listen(20)
print('Socket bind over!')

while True:
    UDP_server, src_addr = TCP_server.accept()
    print('Addr ', src_addr)

    while True:
        image_count = time.strftime('%Y_%m_%d_%H_%M_%S') + '_' + str(time.time() * 100 - int(time.time())*100)
        print(image_count)
        image_recv_path = './UDP_Files/Recv/{}.jpg'.format(image_count)
        image_send_path = './UDP_Files/Send/{}.jpg'.format(image_count)
        message = recv_image(image_recv_path)
        print(message)
        if message == 'close':
            break
        result = from_imagepath_to_result(image_recv_path)
        result.save(image_send_path)
        print('Local process ok!')
        message = send_image(image_send_path)
        print(message)

    UDP_server.close()

TCP_serverc.close()