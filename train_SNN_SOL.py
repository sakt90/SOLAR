import torch
import os, argparse, time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import SGD, lr_scheduler
from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders
from utils.utils import *
import multiprocessing as mp
mp.set_start_method('spawn', force=True) ### to avoid Caffe Warnings

# Setting Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU') if str(device) == "cuda:0" else print('GPU not Detected - CPU Selected')
print(f"GPUs Count: {torch.cuda.device_count()}") # Show how many GPUs are available

################################################ Importing Models and Width Multiplier List
from models.slimmable_ops import width_mult_list
from models.cifar10.resnet_slimmable import SlimmableResNet34, SlimmableResNet34_SOL
from models.svhn.wide_resnet_slimmable import SlimmableWideResNet_16_8, SlimmableWideResNet_16_8_SOL
from models.cifar10.mobilenet_v2_slimmable import SlimmableMobileNetV2, SlimmableMobileNetV2_SOL

def set_seed(seed=3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
set_seed()

parser = argparse.ArgumentParser(description='Training of SNN and SNN_SOL')
parser.add_argument('--method', default='SNN', choices=['SNN', 'SNN_SOL'], help='Method to Use')
parser.add_argument('--model', default='ResNet_34', choices=['ResNet_34', 'WideResNet_16_8', 'MobileNetV2'], help='Dataset to Use')
parser.add_argument('--epochs', default=100, type=int, help='Number of Epochs')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')

args = parser.parse_args()
args.gpu = '0'              # Use 1 GPU
args.cpus = 0               # Use 1 CPU in Data Loader
args.dataset = 'cifar10'
args.efficient = True   
args.use2BN = True          # Use Dual Batch Norm in Models
args.momentum = 0.9         # Optimizer's Momentum
args.wd = 5e-4              # Optimizer's Weight Decay
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

############### DATA LOADERS:
if args.dataset == 'cifar10':
    args.batch_size = 128
    train_loader, val_loader, test_loader = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    args.batch_size = 128
    train_loader, val_loader, test_loader = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    args.batch_size = 64
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

############### MODEL SELECTION:
if args.model == 'ResNet_34':
    if args.method == 'SNN':
        model_fn = SlimmableResNet34
    elif args.method == 'SNN_SOL':
        model_fn = SlimmableResNet34_SOL
    model = model_fn().cuda()

elif args.model == 'WideResNet_16_8':
    if args.method == 'SNN':
        model_fn = SlimmableWideResNet_16_8
    elif args.method == 'SNN_SOL':
        model_fn = SlimmableWideResNet_16_8_SOL
    model = model_fn(depth=16, num_classes=10, widen_factor=8, dropRate=0.0).cuda()

elif args.model == 'MobileNetV2':
    if args.method == 'SNN':
        model_fn = SlimmableMobileNetV2
    elif args.method == 'SNN_SOL':
        model_fn = SlimmableMobileNetV2_SOL
    model = model_fn().cuda()

############### MAKING DIRECTORIES:
model_str = os.path.join(model_fn.__name__)
opt_str = 'Epochs%d_BatchSize%d_LR%s' % (args.epochs, args.batch_size, args.lr)
save_folder = os.path.join('./SNN_results', args.dataset, model_str, '%s' % (opt_str))
print(save_folder)
create_dir(save_folder)

############### OPTIMIZER:
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

############### INITIALIZING VARIABLES:
start_epoch = 0
training_loss, val_TA, best_TA = [], {}, {}
val_TA, best_TA = [], 0

################################## TRAINING:
for epoch in range(start_epoch, args.epochs):
    train_fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
    val_fp = open(os.path.join(save_folder, 'val_log.txt'), 'a+')
    start_time = time.time()
    
    model.train()
    requires_grad_(model, True)
    accs, losses, lps = AverageMeter(), AverageMeter(), AverageMeter()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()

        optimizer.zero_grad()   ######################### zero out the gradients
        for width_mult in sorted(width_mult_list, reverse=True):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))     ##### Extract SLIM Model

            ############################################### CLEAN Loss:
            logits = model(imgs)
            lc = F.cross_entropy(logits, labels, reduction='none')
            loss = torch.mean(lc)
            loss.backward()

        model.apply(lambda m: setattr(m, 'width_mult', 1.0))     ############################## Activate full model width
        optimizer.step()

        ##### get current lr:
        current_lr = scheduler.get_last_lr()[0]

        ##### metrics:
        accs.append(((logits.argmax(1) == labels).float().mean().item()) * 100)
        losses.append(loss.item())

        if i % 100 == 0:
            train_str = 'Epoch %d-%d | Train | Loss: %.4f, Test Acc: %.4f' % (epoch, i, losses.avg, accs.avg)
            print(train_str)
            train_fp.write(train_str + '\n')
    scheduler.step() # lr scheduler update at the end of each epoch

    ################################## Validation:
    model.eval()
    requires_grad_(model, False) # Evaluation Mode

    eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.75*args.epochs)) # boolean
    
    if eval_this_epoch:
        val_accs = AverageMeter()
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            for width_mult in sorted(width_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'width_mult', width_mult)) 
                logits = model(imgs)
                val_accs.append((logits.argmax(1) == labels).float().mean().item())

    val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s' % (epoch, (time.time()-start_time), current_lr)
    if eval_this_epoch:
        val_str += ' Test Acc: %.4f\n' % (val_accs.avg)
    print(val_str)
    val_fp.write(val_str + '\n')
    val_fp.close() # close file pointer

    # Save Loss Curves:
    training_loss.append(losses.avg)
    plt.plot(training_loss)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'training_loss.png'))
    plt.close()

    if eval_this_epoch:
        val_TA.append(val_accs.avg)
        plt.plot(val_TA, 'r')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()

    else:
        val_TA.append(val_TA[-1]) 
        plt.plot(val_TA, 'r')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()

    # Save MODEL:
    if eval_this_epoch:
        if val_accs.avg >= best_TA:
            best_TA = val_accs.avg # update best TA
            torch.save(model.state_dict(), os.path.join(save_folder, 'Best_Model.pth'))