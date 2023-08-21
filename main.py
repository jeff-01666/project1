import argparse
from validation import validation_binary,train_binary
import torch
from torch import nn
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from networks.initialization import weight_init_KaimingNormal,weight_init_XavierUniform,weight_init_HeUniform

from Loss.TverskyLoss import Tversky_Loss
from Dataset.dataset import BreastDataset
import utils.util
import os
import glob
import random
import numpy as np

from U_Net.unet_model import UNet   #remember amend parameter
#from Analysis_experiment.nU_Net.unet_model import UNet
#from Analysis_experiment.RAU_UNet.unet_model import UNet
#from Analysis_experiment.MSRAU_UNet_V3.unet_model import UNet
#from Analysis_experiment.MSRAU_UNet_V3_DS.unet_model import UNet
#from Analysis_experiment.UTB.unet_model import UNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard_weight',default=0.3 ,type=float)
    arg('--device_ids',type=str,default='0',help='gpu')
    arg('--root',type=str,default='./',help='root help')
    arg('--batch_size',type=int,default=2)
    arg('--n_epochs',type=int,default=200)
    arg('--lr',type=float,default=1e-3)
    arg('--worker',type=int,default=12)
    arg('--num_classes',type=int,default=2)
    arg('--modelpath',type=str,default='U_Net')
    args = parser.parse_args()
    
    root = args.root
    num_classes = args.num_classes
    #set_seed(1)
    model = UNet(1, 1)
    model.apply(weight_init_HeUniform)
    print(model)
   
    
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print("Number of trainable parameters %d in Model" % num_para)
  
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    #loss = LossBinary(jaccard_weight=args.jaccard_weight)
    dice_loss = Tversky_Loss()
    cudnn.benchmark = True
    
    def make_loader(file_names,shuffle=False,batch_size=None):
        return DataLoader(
            dataset = BreastDataset(file_names,method='train'),
            shuffle=shuffle,
            num_workers=args.worker,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    data_path = './data'
    train_file_names = glob.glob(os.path.join(data_path,'new_train','images','*.png'))
    val_file_names = glob.glob(os.path.join(data_path,'new_val','images','*.png'))
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_loader = make_loader(train_file_names,shuffle=True,batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names,shuffle=True,batch_size=len(device_ids))

    valid = validation_binary
    train_valid = train_binary

    utils.util.train(
        
        init_optimizer1 = lambda lr :Adam( model.parameters(),lr=lr,weight_decay=1e-5),

        #init_optimizer1 = lambda lr :SGD( model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-5),

        args = args,
        model = model,
        criterion = dice_loss,
        train_dataset = train_loader,
        valid_dataset = valid_loader,
        eval_funcation = valid,
        train_funcation = train_valid,
        num_classes = num_classes,
        n_epochs = args.n_epochs
    )
'''
init_optimizer = lambda lr :Adam( model.parameters(),
                                          lr=lr,
                                          weight_decay = 1e-8),
'''
if __name__ == "__main__":
    main()
    
