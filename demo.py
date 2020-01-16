import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
# imported by dorsa
import cv2
from torchvision import transforms, utils
from demo_transform import *
import PIL
import time

plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model.load_state_dict(torch.load('./final_model.pth'))
    model.eval()

    # nyu2_loader = loaddata.readNyu2('data/demo/1.png')
    nyu2_loader = loaddata.readNyu2('examples/5.jpg')
    image = list(nyu2_loader)[0]
    print(image.shape)

    # test(nyu2_loader, model)
    webcam(model)


def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transform = transforms.Compose([
        Scale([320, 240]),
        CenterCrop([304, 228]),
        ToTensor(),
        Normalize(__imagenet_stats['mean'],
                  __imagenet_stats['std'])
    ])

    image = transform(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor


def webcam(model):
    capture = cv2.VideoCapture(0)
    # frame_width = 1216
    # frame_height = 912
    # video_writer = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    fps = 0
    while (True):
        start = time.time()
        for i in range(3):
            capture.grab()
        ret, frame = capture.read()

        image = preprocess(frame)
        out = model(image)
        out = out.view(out.size(2), out.size(3)).data.cpu().numpy()

        end = time.time()
        print("Time: " + str(end - start))

        out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        img = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        img = cv2.resize(img, None, fx=8, fy=8)
        cv2.imshow('Deptf', img)
        print(img.shape)
        # video_writer.write(img)
        fps = 0
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    # video_writer.release()
    cv2.destroyAllWindows()


def test(nyu2_loader, model):

    for i, image in enumerate(nyu2_loader):
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)

        # matplotlib.image.imsave('data/demo/out2.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
        plt.imshow(out.view(out.size(2), out.size(3)).data.cpu().numpy())
        plt.show()


if __name__ == '__main__':
    main()
