#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager , SetDataManager
import configs
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
import wrn_mixup_model
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
from os import path

use_gpu = torch.cuda.is_available()
image_size = 80

def train_manifold_mixup(base_loader, base_loader_test, model, start_epoch, stop_epoch, params):

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch"  , start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        for batch_idx, (input_var, target_var) in enumerate(base_loader):
            if use_gpu:
                input_var, target_var = input_var.cuda(), target_var.cuda()
            input_var, target_var = Variable(input_var), Variable(target_var)
            # print(target_var, input_var)
            lam = np.random.beta(params.alpha, params.alpha)
            _ , outputs , target_a , target_b  = model(input_var, target_var, mixup_hidden= True, mixup_alpha = params.alpha , lam = lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx%50 ==0 :
                print('{0}/{1}'.format(batch_idx,len(base_loader)), 'Loss: %.3f | Acc: %.3f%%  '
                             % (train_loss/(batch_idx+1),100.*correct/total ))
        

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
         

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f , outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                             % (test_loss/(batch_idx+1), 100.*correct/total ))

        torch.cuda.empty_cache()
        
    return model 


def train_s2m2(base_loader, base_loader_test, val_loader ,model, start_epoch, stop_epoch, params,tmp):
    val_acc_best = 0.0

    if path.exists('./val_'+ params.dataset +  '.pt'):
        loader = torch.load('./val_'+ params.dataset +  '.pt')
    else:
        loader = []
        for _ , (x,_) in enumerate(val_loader):
            loader.append(x)
        torch.save(loader,'./val_'+ params.dataset +  '.pt')


    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    rotate_classifier = nn.Sequential( nn.Linear(640,4)) 
    rotate_classifier.cuda()

    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ])

    print("stop_epoch"  , start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        rotate_loss = 0
        correct = 0
        total = 0
        torch.cuda.empty_cache()
        
        for batch_idx, (inputs, targets) in enumerate(base_loader):
            
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            lam = np.random.beta(params.alpha, params.alpha)
            f , outputs , target_a , target_b = model(inputs, targets, mixup_hidden= True , mixup_alpha = params.alpha , lam = lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
            
            bs = inputs.size(0)
            inputs_ = []
            targets_ = []
            a_ = []
            indices = np.arange(bs)
            np.random.shuffle(indices)
            
            split_size = int(bs/4)
            for j in indices[0:split_size]:
                x90 = inputs[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                inputs_ += [inputs[j], x90, x180, x270]
                targets_ += [targets[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            inputs = Variable(torch.stack(inputs_,0))
            targets = Variable(torch.stack(targets_,0))
            a_ = Variable(torch.stack(a_,0))

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                a_ = a_.cuda()

            
            rf , outputs = model(inputs)
            rotate_outputs = rotate_classifier(rf)
            rloss =  criterion(rotate_outputs,a_)
            closs = criterion(outputs, targets)
            loss = (rloss+closs)/2.0
            
            rotate_loss += rloss.data.item()
            
            loss.backward()
            
            optimizer.step()
            
            
            if batch_idx%50 ==0 :
                print('{0}/{1}'.format(batch_idx,len(base_loader)), 
                             'Loss: %.3f | Acc: %.3f%% | RotLoss: %.3f  '
                             % (train_loss/(batch_idx+1),
                                100.*correct/total,rotate_loss/(batch_idx+1)))
     

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
         

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f , outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                             % (test_loss/(batch_idx+1), 100.*correct/total ))



        valmodel =  BaselineFinetune(model_dict[params.model],5,1,loss_type='dist')
        valmodel.n_query=15
        acc_all1, acc_all2 , acc_all3 = [],[],[]
        for i, x in enumerate(loader):
            x = x.view(-1,3,image_size,image_size)
            if use_gpu:
                x = x.cuda()

            with torch.no_grad():
                f,scores = model(x)
            f = f.view(5,16,-1)
            scores  = valmodel.set_forward_adaptation(f.cpu())
            acc = []
            for each_score in scores:
                pred = each_score.data.cpu().numpy().argmax(axis = 1)
                y = np.repeat(range( 5 ), 15 )
                acc.append(np.mean(pred == y)*100 )
            acc_all1.append(acc[0])
            acc_all2.append(acc[1])
            acc_all3.append(acc[2])
                
        print('Test Acc at 100= %4.2f%%' %(np.mean(acc_all1)))
        print('Test Acc at 200= %4.2f%%' %(np.mean(acc_all2)))
        print('Test Acc at 300= %4.2f%%' %(np.mean(acc_all3)))
        
        if np.mean(acc_all3) > val_acc_best:
            val_acc_best = np.mean(acc_all3)
            bestfile =os.path.join(params.checkpoint_dir, 'best.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict() , 'rotate': rotate_classifier.state_dict()}, bestfile)

       
    return model 


def train_rotation(base_loader, base_loader_test, model, start_epoch, stop_epoch, params , tmp):
    rotate_classifier = nn.Sequential( nn.Linear(640,4)) 
    if use_gpu:
        rotate_classifier.cuda()
    
    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])
        
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ])
    
    lossfn = nn.CrossEntropyLoss()
    max_acc = 0 

    print("stop_epoch" , start_epoch, stop_epoch )

    for epoch in range(start_epoch,stop_epoch):
        rotate_classifier.train()
        model.train()

        avg_loss=0
        avg_rloss=0
        
        for i, (x,y) in enumerate(base_loader):
            bs = x.size(0)
            x_ = []
            y_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                y_ += [y[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = Variable(torch.stack(x_,0))
            y_ = Variable(torch.stack(y_,0))
            a_ = Variable(torch.stack(a_,0))

            if use_gpu:
                x_ = x_.cuda()
                y_ = y_.cuda()
                a_ = a_.cuda()

            f,scores = model.forward(x_)
            rotate_scores =  rotate_classifier(f)

            optimizer.zero_grad()
            rloss = lossfn(rotate_scores,a_)
            closs = lossfn(scores, y_)
            loss = 0.5*closs + 0.5*rloss
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+closs.data.item()
            avg_rloss = avg_rloss+rloss.data.item()
            

            if i % 50 ==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Rotate Loss {:f}'.format(epoch, i, len(base_loader), avg_loss/float(i+1),avg_rloss/float(i+1)  ))
            
            
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() , 'rotate': rotate_classifier.state_dict()}, outfile)
         

                
        model.eval()
        rotate_classifier.eval()

        with torch.no_grad():
            correct = rcorrect = total = 0
            for i,(x,y) in enumerate(base_loader_test):
                if i<2:
                    bs = x.size(0)
                    x_ = []
                    y_ = []
                    a_ = []
                    for j in range(bs):
                        x90 = x[j].transpose(2,1).flip(1)
                        x180 = x90.transpose(2,1).flip(1)
                        x270 =  x180.transpose(2,1).flip(1)
                        x_ += [x[j], x90, x180, x270]
                        y_ += [y[j] for _ in range(4)]
                        a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

                    x_ = Variable(torch.stack(x_,0))
                    y_ = Variable(torch.stack(y_,0))
                    a_ = Variable(torch.stack(a_,0))

                    if use_gpu:
                        x_ = x_.cuda()
                        y_ = y_.cuda()
                        a_ = a_.cuda()


                    f,scores = model(x_)
                    rotate_scores =  rotate_classifier(f)
                    p1 = torch.argmax(scores,1)
                    correct += (p1==y_).sum().item()
                    total += p1.size(0)
                    p2 = torch.argmax(rotate_scores,1)
                    rcorrect += (p2==a_).sum().item()

            print("Epoch {0} : Accuracy {1}, Rotate Accuracy {2}".format(epoch,(float(correct)*100)/total,(float(rcorrect)*100)/total))
        torch.cuda.empty_cache()

    return model


if __name__ == '__main__':
    params = parse_args('train')

    base_file = f'./filelists/{params.dataset}/base.json'
    val_file = f'./filelists/{params.dataset}/val.json'
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
    base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    base_datamgr_test    = SimpleDataManager(image_size, batch_size = params.test_batch_size)
    base_loader_test     = base_datamgr_test.get_data_loader( base_file , aug = False )
    test_few_shot_params     = dict(n_way = 5, n_support = 1) 
    val_datamgr             = SetDataManager(image_size, n_query = 15, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 


    if params.method == 'manifold_mixup':
        print(params.num_classes)
        model = wrn_mixup_model.wrn28_10(params.num_classes)
    elif params.method == 'S2M2_R':
        model = wrn_mixup_model.wrn28_10(params.num_classes)
    elif params.method == 'rotation':
        model = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')
 
            
    
    if params.method =='S2M2_R':
        if use_gpu:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
            model.cuda()

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir )        
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']        
            model.load_state_dict(state)        

        else:
            resume_rotate_file_dir = params.checkpoint_dir.replace("S2M2_R","rotation")
            resume_file = get_resume_file( resume_rotate_file_dir )        
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']
            state_keys = list(state.keys())

            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state[key.replace("classifier.","linear.")] =  state[key]
                    state.pop(key)
            model.load_state_dict(state)        
    
        model = train_s2m2(base_loader, base_loader_test, val_loader,  model, start_epoch, start_epoch+stop_epoch, params , {})


    elif params.method =='rotation':
        if use_gpu:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
            model.cuda()

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir )        
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']        
            model.load_state_dict(state)        

        model = train_rotation(base_loader, base_loader_test, model, start_epoch, stop_epoch, params,{})

    elif params.method == 'manifold_mixup':
        if use_gpu:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
            model.cuda()

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir )        
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']        
            model.load_state_dict(state)        

        model = train_manifold_mixup(base_loader, base_loader_test, model, start_epoch, stop_epoch, params)



  
