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
from utils.context import ctx_noparamgrad_and_eval
from utils.sample_lambda import element_wise_sample_lambda
from attacks.pgd import PGD
import multiprocessing as mp
mp.set_start_method('spawn', force=True) ### to avoid Caffe Warnings

# Setting Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU') if str(device) == "cuda:0" else print('GPU not Detected - CPU Selected')
print(f"GPUs Count: {torch.cuda.device_count()}") # Show how many GPUs are available

################################################ Importing Models and Width Multiplier List
from models.slimmable_ops import width_mult_list
from models.cifar10.resnet_slimmable_OAT import SlimmableResNet34OAT, SlimmableResNet34OAT_SOL
from models.svhn.wide_resnet_slimmable_OAT import SlimmableWideResNet_16_8_OAT, SlimmableWideResNet_16_8_OAT_SOL
from models.stl10.wide_resnet_slimmable_OAT import SlimmableWideResNet_40_2_OAT, SlimmableWideResNet_40_2_OAT_SOL

def set_seed(seed=3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
set_seed()

parser = argparse.ArgumentParser(description='Training of OATS and OATS_SOL')
parser.add_argument('--method', default='OATS', choices=['OATS', 'OATS_SOL'], help='Method to Use')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='Dataset to Use')
parser.add_argument('--epochs', default=100, type=int, help='Number of Epochs')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')

args = parser.parse_args()
args.gpu = '0'              # Use 1 GPU
args.cpus = 0               # Use 1 CPU in Data Loader
args.efficient = True   
args.use2BN = True          # Use Dual Batch Norm in Models
args.momentum = 0.9         # Optimizer's Momentum
args.wd = 5e-4              # Optimizer's Weight Decay
args.distribution = 'disc'  # Distribution for Lambda Values (as in OAT [14])
args.probs = -1             # Probs for Lambda Values (as in OAT [14])
args.eps = 8                # PGD Epsilon
args.steps = 7              # PGD Iterations
args.lambda_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]    # Lambda Values (as in OAT [14])
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

args.dim = args.batch_size  # Dimension of Lambda (as in OAT [14])

############### MODEL SELECTION:
FiLM_in_channels = args.dim

if args.dataset == 'cifar10':
    if args.method == 'OATS':
        model_fn = SlimmableResNet34OAT
    elif args.method == 'OATS_SOL':
        model_fn = SlimmableResNet34OAT_SOL
    model = model_fn(use2BN=args.use2BN, FiLM_in_channels=FiLM_in_channels).cuda()
elif args.dataset == 'svhn':
    if args.method == 'OATS':
        model_fn = SlimmableWideResNet_16_8_OAT
    elif args.method == 'OATS_SOL':
        model_fn = SlimmableWideResNet_16_8_OAT_SOL
    model = model_fn(depth=16, num_classes=10, widen_factor=8, dropRate=0.0, use2BN=True, FiLM_in_channels=FiLM_in_channels).cuda()
elif args.dataset == 'stl10':
    if args.method == 'OATS':
        model_fn = SlimmableWideResNet_40_2_OAT
    elif args.method == 'OATS_SOL':
        model_fn = SlimmableWideResNet_40_2_OAT_SOL
    model = model_fn(depth=40, num_classes=10, widen_factor=2, dropRate=0.0, use2BN=True, FiLM_in_channels=FiLM_in_channels).cuda()

############### MAKING DIRECTORIES:
model_str = os.path.join(model_fn.__name__)
opt_str = 'Epochs%d_BatchSize%d_LR%s' % (args.epochs, args.batch_size, args.lr)

attack_str = 'PGD_%d' % (args.steps)
    
save_folder = os.path.join('./OATS_results', args.dataset, model_str, '%s_%s' % (attack_str, opt_str))
print(save_folder)
create_dir(save_folder)

############### LAMBDA Encoding Matrix:
rand_mat = np.random.randn(args.dim, args.dim)
rand_otho_mat, _ = np.linalg.qr(rand_mat)
encoding_mat = rand_otho_mat

############### ATTACKER
attacker = PGD(eps=args.eps/255, steps=args.steps, use_FiLM=True)

############### Validation LAMBDA Values:
val_lambdas = args.lambda_choices

############### OPTIMIZER:
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

############### INITIALIZING VARIABLES:
start_epoch = 0
training_loss, val_TA, val_ATA, best_TA, best_ATA = [], {}, {}, 0, 0
for val_lambda in val_lambdas:
    val_TA[val_lambda], val_ATA[val_lambda] = [], []

################################## TRAINING:
for epoch in range(start_epoch, args.epochs):
    train_fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
    val_fp = open(os.path.join(save_folder, 'val_log.txt'), 'a+')
    start_time = time.time()
    
    ##### training:
    model.train()
    requires_grad_(model, True)
    accs, accs_adv, losses, lps = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        
        ##### sample _lambda:
        _lambda_flat, _lambda, num_zeros = element_wise_sample_lambda(args.distribution, args.lambda_choices, encoding_mat, batch_size=args.batch_size, probs=args.probs)
        if args.use2BN:
            idx2BN = num_zeros
        else:
            idx2BN = None

        optimizer.zero_grad()   ######################### zero out the gradients
        
        for width_mult in sorted(width_mult_list, reverse=True):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))     ##### Extract/Activate SLIM Models
        
            ############################################### CLEAN Loss:
            
            # logits for clean imgs:
            logits = model(imgs, _lambda, idx2BN)
            lc = F.cross_entropy(logits, labels, reduction='none')
            
            ############################################### ADVERSARIAL Loss:
            if args.efficient:
                ##### generate adversarial images:
                with ctx_noparamgrad_and_eval(model):
                    if args.use2BN:
                        imgs_adv = attacker.attack(model, imgs[num_zeros:], labels=labels[num_zeros:], _lambda=_lambda[num_zeros:], idx2BN=0)
                    else:
                        imgs_adv = attacker.attack(model, imgs[num_zeros:], labels=labels[num_zeros:], _lambda=_lambda[num_zeros:], idx2BN=None)
                logits_adv = model(imgs_adv.detach(), _lambda[num_zeros:], idx2BN=0)    ##### logits for adv imgs:
                
                ##### loss and update:
                la = F.cross_entropy(logits_adv, labels[num_zeros:], reduction='none') 
                la = torch.cat([torch.zeros((num_zeros,)).cuda(), la], dim=0)
            else:
                ##### generate adversarial images:
                with ctx_noparamgrad_and_eval(model):
                    imgs_adv = attacker.attack(model, imgs, labels=labels, _lambda=_lambda, idx2BN=idx2BN)
                logits_adv = model(imgs_adv.detach(), _lambda, idx2BN=idx2BN)    ##### logits for adv imgs:

                ##### loss and update:
                la = F.cross_entropy(logits_adv, labels, reduction='none')
            
            ############################################### TOTAL Loss:    
            wc = (1-_lambda_flat)
            wa = _lambda_flat
            loss = (torch.mean(wc * lc + wa * la))
            loss.backward()

        model.apply(lambda m: setattr(m, 'width_mult', 1.0))     ############################## Activate full model width
        optimizer.step()

        ##### get current lr:
        current_lr = scheduler.get_last_lr()[0]

        ##### metrics:
        accs.append((logits.argmax(1) == labels).float().mean().item())
        if args.efficient:
            accs_adv.append((logits_adv.argmax(1) == labels[num_zeros:]).float().mean().item())
        else:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())

        if i % 100 == 0:
            train_str = 'Epoch %d-%d | Train | Loss: %.4f, Test Acc: %.4f, Adversarial Test Acc: %.4f' % (
                epoch, i, losses.avg, accs.avg, accs_adv.avg)
            print(train_str)
            train_fp.write(train_str + '\n')

    scheduler.step() # lr scheduler update at the end of each epoch

    ################################## Validation:
    model.eval()
    requires_grad_(model, False) # Evaluation Mode

    eval_this_epoch = (epoch % 5 == 0) or (epoch>=int(0.5*args.epochs)) # boolean
    if eval_this_epoch:
        val_accs, val_accs_adv = {}, {}
        for val_lambda in val_lambdas:
            val_accs[val_lambda], val_accs_adv[val_lambda] = AverageMeter(), AverageMeter()
            
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.cuda(), labels.cuda()

            for j, val_lambda in enumerate(val_lambdas):
                # sample _lambda:
                if args.distribution == 'disc' and encoding_mat is not None:
                    _lambda = np.expand_dims( np.repeat(j, labels.size()[0]), axis=1 ).astype(np.uint8)
                    _lambda = encoding_mat[_lambda,:] 
                else:
                    _lambda = np.expand_dims( np.repeat(val_lambda, labels.size()[0]), axis=1 )
                _lambda = torch.from_numpy(_lambda).float().cuda()
                if args.use2BN:
                    idx2BN = int(labels.size()[0]) if val_lambda==0 else 0
                else:
                    idx2BN = None
                # Validation Accuracy:
                logits = model(imgs, _lambda, idx2BN)
                val_accs[val_lambda].append((logits.argmax(1) == labels).float().mean().item())
                
                # Validation Robustness:
                with ctx_noparamgrad_and_eval(model):
                    imgs_adv = attacker.attack(model, imgs, labels=labels, _lambda=_lambda, idx2BN=idx2BN) # generate adversarial images:
                linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
                logits_adv = model(imgs_adv.detach(), _lambda, idx2BN)
                val_accs_adv[val_lambda].append((logits_adv.argmax(1) == labels).float().mean().item())

    val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s\n' % (epoch, (time.time()-start_time), current_lr)
    if eval_this_epoch:
        #val_str += ' | linf: %.4f - %.4f\n' % (torch.min(linf_norms).data, torch.max(linf_norms).data)
        for val_lambda in val_lambdas:
            val_str += 'val_lambda%s: Test Acc: %.4f, Adversarial Test Acc: %.4f\n' % (val_lambda, val_accs[val_lambda].avg, val_accs_adv[val_lambda].avg)
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
        for val_lambda in val_lambdas:
            val_TA[val_lambda].append(val_accs[val_lambda].avg) 
            plt.plot(val_TA[val_lambda], 'r')
            val_ATA[val_lambda].append(val_accs_adv[val_lambda].avg)
            plt.plot(val_ATA[val_lambda], 'g')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_acc%s.png' % val_lambda))
            plt.close()
    else:
        for val_lambda in val_lambdas:
            val_TA[val_lambda].append(val_TA[val_lambda][-1]) 
            plt.plot(val_TA[val_lambda], 'r')
            val_ATA[val_lambda].append(val_ATA[val_lambda][-1])
            plt.plot(val_ATA[val_lambda], 'g')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_acc%s.png' % val_lambda))
            plt.close()

    # Save MODEL:
    if eval_this_epoch:
        avg_val_TA_list = []
        for val_lambda in val_lambdas:
            avg_val_TA_list.append(val_accs[val_lambda].avg)
            
        avg_val_TA = sum(avg_val_TA_list) / len(avg_val_TA_list)    # Averaging performance across all lambdas
        if avg_val_TA >= best_TA:
            best_TA = avg_val_TA # update best TA
            torch.save(model.state_dict(), os.path.join(save_folder, 'Best_Model.pth'))