import argparse
import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
from backbone.mobilenet_v3 import mobilenetv3
from backbone.shufflenet_v2 import shufflenet_v2_x0_5
from torch.utils import data
import PIL
import helper.utils as utils
from dataset.data_augmentation import ImgAugTransform

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
    parser.add_argument('--end_epoch', type=int, default=150, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.05, help='start learning rate')
    parser.add_argument('--data', type=str, default="dataset", help='data path')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU True/ False')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    args = parser.parse_args()
    return args


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    """
    Let's go to train my model! So exciting!
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param epoch:
    :param args:
    :return:
    """
    model.train()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    progress = utils.ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        # Compute output
        output = model(images)
        loss = criterion(output, targets)

        # Measure accuracy and record loss
        acc1 = utils.accuracy(output, targets)
        losses.update(loss.item())
        top1.update(acc1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            progress.display(idx)

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    """
    Challenge my model
    :param val_loader:
    :param model:
    :param criterion:
    :param args:
    :return:
    """
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    with torch.no_grad():
        for inx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Compute output
            val_output = model(images)
            val_loss = criterion(val_output, targets)

            # Measure accuracy and record loss
            acc1 = utils.accuracy(val_output, targets)
            losses.update(val_loss.item())
            top1.update(acc1)

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def main_worker(args):
    """
    Define everything to train the model right here
    :param args:
    :return:
    """
    best_acc = float('-inf')
    writer = SummaryWriter()
    # Create model
    if args.pretrained:
        print("Using pre-trained model!")
        model = ""
    else:
        print("Creating the model!")
        model = shufflenet_v2_x0_5()
        model = nn.DataParallel(model)
    if args.gpu is not None:
        model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=4e-5)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir, transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                ImgAugTransform(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=32, shuffle=False,
        num_workers=1, pin_memory=True)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True)

    for epoch in range(0, args.end_epoch):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        val_loss, val_acc = validate(val_loader, model, criterion, args)

        # Update validate accuracy to scheduler
        scheduler.step(train_acc)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)
        writer.add_scalar('model/learning_rate', lr, epoch)
        writer.add_scalar('model/val_acc', val_acc, epoch)
        writer.add_scalar('model/val_loss', val_loss, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        utils.save_checkpoint(epoch, model, optimizer, best_acc, is_best, train_loss)


if __name__ == "__main__":
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    main_worker(args)
