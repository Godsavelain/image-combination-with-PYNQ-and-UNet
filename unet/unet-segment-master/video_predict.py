import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
import torchsummary as summary
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
    #                     help='Filenames of ouput images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help="Visualize the images as they are processed",
    #                     default=False)
    # parser.add_argument('--no-save', '-n', action='store_true',
    #                     help="Do not save the output masks",
    #                     default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.6)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def mask_to_BGR(mask):
    return cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    cap = cv2.VideoCapture(0)
    while True:
        ret_flag, img = cap.read()
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        mask = predict_img(net=net, full_img=img_PIL, scale_factor=args.scale,
                           out_threshold=args.mask_threshold, device=device)
        mask_BGR = mask_to_BGR(mask)
        cv2.imshow('Unet', np.hstack([img, mask_BGR]))

        k = cv2.waitKey(100)
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    #
    # for i, fn in enumerate(in_files):
    #     logging.info("\nPredicting image {} ...".format(fn))
    #
    #     img = Image.open(fn)
    #
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
    #
    #     if not args.no_save:
    #         out_fn = out_files[i]
    #         result = mask_to_image(mask)
    #         result.save(out_files[i])
    #
    #         logging.info("Mask saved to {}".format(out_files[i]))
    #
    #     if args.viz:
    #         logging.info("Visualizing results for image {}, close to continue ...".format(fn))
    #         plot_img_and_mask(img, mask)