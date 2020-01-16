import argparse

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import loaddata
import util
import numpy as np
import sobel
from models import modules, net, resnet, densenet, senet, ssim_torch, dbe
from statistics import mean
from util import ReplayBuffer

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=5, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--lrD', '--learning-rate_D', default=0.00001, type=float,
                    help='initial Dis learning rate')
parser.add_argument('--alpha', '--alpha', default=0.6, type=float,
                    help='initial alpha for l1ssim')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


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
    global args
    args = parser.parse_args()
    print(args)
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
 
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = model.cuda()
        batch_size = 4

    netD = net.NLDis(1)
    netD = netD.cuda()

    netDBE = dbe.DBELoss()

    print('  + Number of params in model: {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    print('  + Number of params in netD: {}'.format(
        sum(p.numel() for p in netD.parameters() if p.requires_grad)))

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer_D = torch.optim.Adam(netD.parameters(), args.lrD, weight_decay=0)

    train_loader = loaddata.getTrainingData(batch_size)

    min_loss = 1000


    fake_buffer = ReplayBuffer()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train_epoch_loss,train_epoch_loss_depth,train_epoch_loss_dx,train_epoch_loss_dy,train_epoch_loss_normal,train_epoch_loss_l1,train_epoch_loss_ssim,train_epoch_loss_l1SSIM,train_epoch_loss_adv,train_epoch_loss_D_real,train_epoch_loss_D_fake,train_epoch_loss_D = train(train_loader, model, netD, optimizer, optimizer_D, netDBE, epoch, fake_buffer)

        if train_epoch_loss < min_loss:
            min_loss = train_epoch_loss
            torch.save(model.state_dict(), 'epoch_{:1}_model.pth'.format(epoch))

        with open('epoch_losses.txt', 'a') as f:
            f.write("Epoch : {:1} | Loss : {:.4f} | loss_depth : {:.4f} | loss_dx : {:.4f} | loss_dy : {:.4f} | loss_normal : {:.4f} | loss_l1 : {:.4f} | loss_ssim : {:.4f} | loss_l1SSIM : {:.4f} | loss_adv : {:.4f} | loss_D_real : {:.4f} | loss_D_fake : {:.4f} | loss_D : {:.4f} | \n"
                    .format(epoch, train_epoch_loss,train_epoch_loss_depth,train_epoch_loss_dx,train_epoch_loss_dy,train_epoch_loss_normal,train_epoch_loss_l1,train_epoch_loss_ssim,train_epoch_loss_l1SSIM,train_epoch_loss_adv,train_epoch_loss_D_real,train_epoch_loss_D_fake,train_epoch_loss_D))

    # save_checkpoint({'state_dict': model.state_dict()})
    torch.save(model.state_dict(), 'final_model.pth')


def train(train_loader, model, netD, optimizer, optimizer_D, netDBE, epoch, fake_buffer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_D = AverageMeter()

    model.train()
    netD.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    l1 = nn.L1Loss()
    ssim = ssim_torch.SSIM()
    mse = nn.MSELoss()

    end = time.time()

    train_epoch_losses = []
    train_epoch_loss_depth = []
    train_epoch_loss_dx = []
    train_epoch_loss_dy = []
    train_epoch_loss_normal = []
    train_epoch_loss_l1 = []
    train_epoch_loss_ssim = []
    train_epoch_loss_l1SSIM = []
    train_epoch_loss_adv = []
    train_epoch_loss_D_real = []
    train_epoch_loss_D_fake = []
    train_epoch_loss_D = []

    train_epoch_losses.clear()
    train_epoch_loss_depth.clear()
    train_epoch_loss_dx.clear()
    train_epoch_loss_dy.clear()
    train_epoch_loss_normal.clear()
    train_epoch_loss_l1.clear()
    train_epoch_loss_ssim.clear()
    train_epoch_loss_l1SSIM.clear()
    train_epoch_loss_adv.clear()
    train_epoch_loss_D_real.clear()
    train_epoch_loss_D_fake.clear()
    train_epoch_loss_D.clear()
    
    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)

        # print("image #####################################################")
        # print(image.shape)
        # print(image.min())
        # print(image.max())
        # print(image.mean())
        # print()
        # print("depth #####################################################")
        # print(depth.shape)
        # print(depth.min())
        # print(depth.max())
        # print(depth.mean())
        # print()

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()

        output = model(image)
        # print("output #####################################################")
        # print(output.shape)
        # print(output.min())
        # print(output.max())
        # print(output.mean())
        # print()

        outputD = netD(output)
        # print("outputD #####################################################")
        # print(outputD.shape)
        # print(outputD.min())
        # print(outputD.max())
        # print(outputD.mean())
        # print()


        target_real = Variable(torch.full((outputD.size()), 1, device='cuda:0'), requires_grad=False)
        target_fake = Variable(torch.full((outputD.size()), 0, device='cuda:0'), requires_grad=False)



        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)
        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

        # loss_depth = netDBE(output, depth)
        # loss_depth = (torch.pow((output - depth), 2) / depth).mean()
        # loss_depth = (torch.pow((output - depth), 2) / depth).mean()
        # print("loss_depth #####################################################")
        # print(loss_depth)
        # print()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1.0).mean()
        # print("loss_dx #####################################################")
        # print(loss_dx)
        # print()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1.0).mean()
        # print("loss_dy #####################################################")
        # print(loss_dy)
        # print()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        # print("loss_normal #####################################################")
        # print(loss_normal)
        # print()

        loss_l1 = l1(output, depth)
        # print("loss_l1 #####################################################")
        # print(loss_l1)
        # print()
        loss_ssim = 1 - ssim(output, depth)
        # print("loss_ssim #####################################################")
        # print(loss_ssim)
        # print()
        loss_l1SSIM = (args.alpha * loss_l1) + ((1-args.alpha) * loss_ssim)
        # print("loss_l1SSIM #####################################################")
        # print(loss_l1SSIM)
        # print()
        loss_adv = mse(outputD, target_real)
        # print("loss_adv #####################################################")
        # print(loss_adv)
        # print()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)  + loss_adv
        # loss = loss_depth + loss_normal + (loss_dx + loss_dy)  + loss_adv
        # print("loss #####################################################")
        # print(loss)
        # print()

        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        train_epoch_losses.append(loss.item())
        train_epoch_loss_depth.append(loss_depth.item())
        train_epoch_loss_dx.append(loss_dx.item())
        train_epoch_loss_dy.append(loss_dy.item())
        train_epoch_loss_normal.append(loss_normal.item())
        train_epoch_loss_l1.append(loss_l1.item())
        train_epoch_loss_ssim.append(loss_ssim.item())
        train_epoch_loss_l1SSIM.append(loss_l1SSIM.item())
        train_epoch_loss_adv.append(loss_adv.item())

        # Training the Discriminator
        optimizer_D.zero_grad()

        # Real loss
        pred_real = netD(depth)
        # print("pred_real #####################################################")
        # print(pred_real.shape)
        # print(pred_real.min())
        # print(pred_real.max())
        # print(pred_real.mean())
        # print()
        loss_D_real = mse(pred_real, target_real)
        # print("loss_D_real #####################################################")
        # print(loss_D_real)
        # print()

        # Fake loss
        output = fake_buffer.push_and_pop(output)
        pred_fake = netD(output.detach())
        # print("pred_fake #####################################################")
        # print(pred_fake.shape)
        # print(pred_fake.min())
        # print(pred_fake.max())
        # print(pred_fake.mean())
        # print()
        loss_D_fake = mse(pred_fake, target_fake)
        # print("loss_D_fake #####################################################")
        # print(loss_D_fake)
        # print()
        

        # Total loss
        loss_D = loss_D_real + loss_D_fake
        # print("loss_D #####################################################")
        # print(loss_D)
        # print()
        # exit(0)
        losses_D.update(loss_D.item(), image.size(0))
        loss_D.backward()
        optimizer_D.step()

        train_epoch_loss_D_real.append(loss_D_real.item())
        train_epoch_loss_D_fake.append(loss_D_fake.item())
        train_epoch_loss_D.append(loss_D.item())


        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss_D {loss_D.val:.4f} ({loss_D.avg:.4f})\t'
          'Loss_depth {Loss_depth:.4f}'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, loss_D=losses_D, Loss_depth=loss_depth.item()))


    return mean(train_epoch_losses), mean(train_epoch_loss_depth), mean(train_epoch_loss_dx), mean(train_epoch_loss_dy),mean(train_epoch_loss_normal),mean(train_epoch_loss_l1),mean(train_epoch_loss_ssim),mean(train_epoch_loss_l1SSIM),mean(train_epoch_loss_adv),mean(train_epoch_loss_D_real),mean(train_epoch_loss_D_fake),mean(train_epoch_loss_D)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 3))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
