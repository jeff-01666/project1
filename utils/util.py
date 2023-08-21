import json
from datetime import datetime
import numpy as np
import os
import torch
import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.optim import SGD
import visdom
import csv
#from time import sleep

def adjust_learning_rate(optimizer,decay_rate=0.9):
    for param_group in  optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        return param_group['lr']

def cuda(x):
    return x.cuda(is_async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['epoch'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args,model,criterion,train_dataset,valid_dataset,eval_funcation,train_funcation,init_optimizer1,n_epochs=None,num_classes=None):
    lr = args.lr
    optimizer = init_optimizer1(lr)
    root_path = Path(args.root)
    model_path = os.path.join(root_path,'Weights',args.modelpath)

    csvfile = open('train_{fold}.csv'.format(fold=args.modelpath), 'a', newline='')
    csv_write = csv.writer(csvfile,dialect='excel')
    csv_write.writerow(["Epoch", "Lr","Train_loss", "Valid_loss", "Jaccard", "DSC", "SEN", "SPE", "PPV"])
    csvfile.close()

    if os.path.exists(model_path):
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        #lr = 0.0009
        #optimizer = init_optimizer2(lr)
    else:
        epoch = 0
        step = 0
    
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, os.path.join(root_path,'Weights',args.modelpath+'_'+str(ep) + '.pt'))

    log = root_path.joinpath('train_{fold}.log'.format(fold=args.modelpath)).open('at', encoding='utf8')

    report_each = 50
    print(epoch)
    print(n_epochs)

    vis = visdom.Visdom(env='UTB_Net')
    
    
    #结果可视化

    for epoch in range(epoch,n_epochs):
        #sleep(0.01)
        '''
        if epoch == 50:
            optimizer = init_optimizer2(lr)

        if epoch>50 and epoch%10==0 :
            lr = adjust_learning_rate(optimizer,decay_rate=0.9)
            #print("SGD")
        '''
        #lr = adjust_learning_rate(optimizer,decay_rate=0.9)

        model.train()

        tq = tqdm.tqdm(total=(len(train_dataset) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        td = train_dataset
        try:
            #nmean_loss = 0
            mean_loss = 0
            for i,(inputs,targets) in enumerate(td):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)
                    
                outputs = model(inputs)
              
                #outputs = F.sigmoid(outputs)
                loss = criterion(outputs,targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0) #获取batch_size大小
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                '''
                if(nmean_loss == mean_loss):
                    lr = lr - lr*0.2
                    for param_group in  optimizer.param_groups:
                        param_group['lr'] = lr
                '''
                #nmean_loss = mean_loss
                #if i and i % report_each == 0:
                    #write_event(log,step,loss=mean_loss)
            #write_event(log,step,loss=mean_loss)
            tq.close()
            save(epoch+1)

            train_loss, train_iou,train_dice,train_se,train_sp,train_pp = train_funcation(
            model, criterion, train_dataset, num_classes)

            valid_metrics = eval_funcation(model, criterion, valid_dataset, num_classes)
            write_event(log, epoch+1, **valid_metrics)
            csvfile = open('train_{fold}.csv'.format(fold=args.modelpath), 'a', newline='')
            csv_write = csv.writer(csvfile,dialect='excel')
            csv_write.writerow([epoch, lr,train_loss, valid_metrics['valid_loss'], valid_metrics['jaccard_loss'],
            valid_metrics['dice'], valid_metrics['sen'], valid_metrics['spe'], valid_metrics['ppv']])
            csvfile.close()
            valid_loss = valid_metrics['valid_loss']
            valid_iou = valid_metrics['jaccard_loss']
            valid_dice = valid_metrics['dice']
            valid_se = valid_metrics['sen']
            valid_sp = valid_metrics['spe']
            valid_pp = valid_metrics['ppv']
           
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_loss,valid_loss])),win='loss',opts=dict(title='LOSS',legend=['train_loss','valid_loss']),update='append')
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_iou,valid_iou])),win='iou',opts=dict(title='IOU',legend=['train_iou','valid_iou']),update='append')
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_dice,valid_dice])),win='dice',opts=dict(title='DICE',legend=['train_dice','valid_dice']),update='append')
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_se,valid_se])),win='sen',opts=dict(title='SEN',legend=['train_sen','valid_sen']),update='append')
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_sp,valid_sp])),win='spe',opts=dict(title='SPE',legend=['train_spe','valid_spe']),update='append')
            vis.line(X=np.array([epoch]),Y=np.column_stack(np.array([train_pp,valid_pp])),win='ppv',opts=dict(title='PPV',legend=['train_ppv','valid_ppv']),update='append')

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
            
            
                

                
                    

                

        
#def test()
