import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from PIL import Image

from config import *
from GalaxiesDataset import *
from rsa_loader import *
from efigi_loader import *
from pytorchtools import EarlyStopping

import pickle
from DatasetFromSubset import *
from samplers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import random

class mySiamese:
    def __init__(self,train_dl, val_dl, unseen_dl, margin=0.2, lr=0.000001, outputSize=37,
                 modelType='triplet', modelPath='./gz2_resnet50', device='cuda', BATCH_SIZE=64):
        
        self.margin = margin
        self.modelPath = modelPath
        self.modelType = modelType
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.modeltype = modelType
        self.BATCH_SIZE = BATCH_SIZE
        self.lr=lr
        self.outputSize = outputSize
	
    def genTriplets(self, batch):
        batch_size = batch['image'].shape[0]
        im_width = batch['image'].shape[2]
        in_channels = batch['image'].shape[1]
        im_height = batch['image'].shape[3]
        
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            labels[i] = batch['labels'][i].argmax()
        
        batch_img_anchor = np.zeros((batch_size, in_channels*im_width*im_height))   # image set anchors
        batch_img_positive = np.zeros((batch_size, in_channels*im_width*im_height))   # image set positive
        batch_img_negative = np.zeros((batch_size, in_channels*im_width*im_height))   # image set negatives
        
        batch_label_anchor = np.zeros((batch_size, ))    # labels for anchors
        batch_label_positive = np.zeros((batch_size,))     # labels for positives
        batch_label_negative = np.zeros((batch_size,))     # labels for negatives
        
        for i in range(batch_size):
            l = labels[i]
            #Add anchor
            batch_img_anchor[i] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            batch_label_anchor[i] = l
            
            # find and add a genuine sample
            ind_positive = np.squeeze(np.argwhere(labels == l))
            randSamp = random.sample(list(ind_positive), 1)
            batch_img_positive[i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_positive[i] = l
            
            
            # find and add a negative sample
            ind_negative = np.squeeze(np.argwhere(labels != l))
            randSamp = random.sample(list(ind_negative), 1)
            batch_img_negative[i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_negative[i] = labels[randSamp]
            
        batch_img_anchor = batch_img_anchor.reshape((-1, in_channels, im_width, im_height))
        batch_img_positive = batch_img_positive.reshape((-1, in_channels, im_width, im_height))
        batch_img_negative = batch_img_negative.reshape((-1, in_channels, im_width, im_height))
        
        batch_label_anchor = torch.from_numpy(batch_label_anchor).long()  # convert the numpy array into torch tensor
        batch_label_positive = torch.from_numpy(batch_label_positive).long()  # convert the numpy array into torch tensor
        batch_label_negative = torch.from_numpy(batch_label_negative).long()  # convert the numpy array into torch tensor
        
        batch_img_anchor = torch.from_numpy(batch_img_anchor).float()     # convert the numpy array into torch tensor
        batch_img_positive = torch.from_numpy(batch_img_positive).float()  # convert the numpy array into torch tensor
        batch_img_negative = torch.from_numpy(batch_img_negative).float()  # convert the numpy array into torch tensor
        
        return batch_img_anchor, batch_img_positive, batch_img_negative, batch_label_anchor, batch_label_positive, batch_label_negative
    
    def genPair(self, batch):
        batch_size = batch['image'].shape[0]
        im_width = batch['image'].shape[2]
        in_channels = batch['image'].shape[1]
        im_height = batch['image'].shape[3]
        
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            labels[i] = batch['labels'][i].argmax()
        
        batch_img_1 = np.zeros((2 * batch_size, in_channels*im_width*im_height))   # image set 1
        batch_img_2 = np.zeros((2 * batch_size, in_channels*im_width*im_height))   # image set 2
        batch_label_1 = np.zeros((2 * batch_size, ))    # labels for image set 1
        batch_label_2 = np.zeros((2 * batch_size,))     # labels for image set 2
        batch_label_c = np.zeros((2 * batch_size,))     # contrastive label: 0 if genuine pair, 1 if impostor pair
        
        for i in range(batch_size):
            l = labels[i]
            # find and add a genuine sample
            ind_g = np.squeeze(np.argwhere(labels == l))
            batch_img_1[2*i] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            #print('label - {} #similars - {}'.format(labels[i], len(ind_g)))
            randSamp = random.sample(list(ind_g), 1)
            batch_img_2[2*i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_1[2*i] = l
            batch_label_2[2*i] = l
            batch_label_c[2*i] = 0
            
            # find and add an impostor sample
            ind_d = np.squeeze(np.argwhere(labels != l))
            randSamp = random.sample(list(ind_d), 1)
            batch_img_1[2*i+1] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            batch_img_2[2*i+1] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_1[2*i+1] = l
            batch_label_2[2*i+1] = labels[randSamp]
            batch_label_c[2*i+1] = 1
            
        batch_img_1 = batch_img_1.reshape((-1, in_channels, im_width, im_height))
        batch_img_2 = batch_img_2.reshape((-1, in_channels, im_width, im_height))
        
        batch_label_1 = torch.from_numpy(batch_label_1).long()  # convert the numpy array into torch tensor
        #batch_label_1 = Variable(batch_label_1).cuda()          # create a torch variable and transfer it into GPU
    
        batch_label_2 = torch.from_numpy(batch_label_2).long()  # convert the numpy array into torch tensor
        #batch_label_2 = Variable(batch_label_2).cuda()          # create a torch variable and transfer it into GPU
    
        batch_label_c = batch_label_c.reshape((-1, 1))
        batch_label_c = torch.from_numpy(batch_label_c).float()  # convert the numpy array into torch tensor
        #batch_label_c = Variable(batch_label_c).cuda()           # create a torch variable and transfer it into GPU
    
        batch_img_1 = torch.from_numpy(batch_img_1).float()     # convert the numpy array into torch tensor
        #batch_img_1 = Variable(batch_img_1).cuda()              # create a torch variable and transfer it into GPU
    
        batch_img_2 = torch.from_numpy(batch_img_2).float()  # convert the numpy array into torch tensor
        #batch_img_2 = Variable(batch_img_2).cuda()           # create a torch variable and transfer it into GPU
        return batch_img_1, batch_img_2, batch_label_1, batch_label_2, batch_label_c
    
    def triplet_loss(self, a, p, n) : 
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + self.margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
    
    def  train_network(self,model, optimizer, scheduler, first_images, second_images, first_labels, second_labels, c_labels, sample_size, tr='train'):
        loss_log = []    
        if tr == 'train':
            model.train()
        elif tr == 'val':
            model.eval()
        for i in range(0, sample_size, self.BATCH_SIZE):
            img_1 = first_images[i: i + self.BATCH_SIZE].cuda()
            img_2 = second_images[i: i + self.BATCH_SIZE].cuda()
            label_1 = first_labels[i: i + self.BATCH_SIZE].cuda()
            label_2 = second_labels[i: i + self.BATCH_SIZE].cuda()
            label_c = c_labels[i: i + self.BATCH_SIZE].cuda()
            
            # Reset gradients
            optimizer.zero_grad()
    
            # Forward pass
            features_1 = model(img_1)
            features_2 = model(img_2)
    
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #euclidean_distance = cos(features_1, features_2)
            euclidean_distance = F.pairwise_distance(features_1, features_2)
            #loss_contrastive = 0.5 * torch.mean((1 - label_c) * torch.pow(euclidean_distance, 2) +
            #                              label_c * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            loss_contrastive = torch.mean(0.5*(label_c) * euclidean_distance +
                                         0.5* (1-label_c) *(torch.clamp(self.margin - euclidean_distance, min=0.0)))
    
            loss_log.append(float(loss_contrastive.data))
            #print('\nepoch: {} - batch: {}'.format(ep, counter))
            #print('Contrastive loss: ', float(loss_contrastive.data))
            
            # Backward pass and updates
            loss_contrastive.backward()                     # calculate the gradients (backpropagation)
            optimizer.step()                    # update the weights
            del img_1
            del img_2
            del label_1
            del label_2
            del label_c
            torch.cuda.empty_cache()
        epoch_loss = sum(loss_log)/len(loss_log)
        return epoch_loss
    
    def  train_network_triplets(self,model, optimizer, scheduler, anchors, positives, negatives, anchor_labels,
                            positive_labels, negative_labels, sample_size, tr='train'):
        loss_log = []    
        if tr == 'train':
            model.train()
        elif tr == 'val':
            model.eval()
        for i in range(0, sample_size, self.BATCH_SIZE):
            a = anchors[i: i + self.BATCH_SIZE].cuda()
            p = positives[i: i + self.BATCH_SIZE].cuda()
            n = negatives[i: i + self.BATCH_SIZE].cuda()
            
            label_a = anchor_labels[i: i + self.BATCH_SIZE].cuda()
            label_p = positive_labels[i: i + self.BATCH_SIZE].cuda()
            label_n = negative_labels[i: i + self.BATCH_SIZE].cuda()
            
            # Reset gradients
            optimizer.zero_grad()
    
            # Forward pass
            features_a = F.normalize(model(a), p=2)
            features_p = F.normalize(model(p), p =2)
            features_n = F.normalize(model(n), p=2)
    
            t_loss = self.triplet_loss(features_a, features_p, features_n)
    
            loss_log.append(float(t_loss.data))
            #print('\nepoch: {} - batch: {}'.format(ep, counter))
            #print('Contrastive loss: ', float(loss_contrastive.data))
            
            # Backward pass and updates
            t_loss.backward()                     # calculate the gradients (backpropagation)
            optimizer.step()                    # update the weights
            del a
            del p
            del n
            del label_a
            del label_p
            del label_n
            del features_a
            del features_p
            del features_n
            torch.cuda.empty_cache()
        epoch_loss = sum(loss_log)/len(loss_log)
        return epoch_loss
    
    def train(self, epochs=5):
        model = torch.load(self.modelPath)
        if self.outputSize != 37:
            model.fc[3] = nn.Linear(512, self.outputSize)
        model = model.to(self.device)
        #optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=0.00005)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00005)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,threshold=0.0001,factor=0.1, 
                                                   verbose=True)
        #optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)
        if self.modelType == 'triplet':
            for i, batch in enumerate(self.train_dl):  # batches loop
                #train_img_1, train_img_2, train_label_1, train_label_2, train_label_c = genPair(batch)
                self.train_img_anchor, self.train_img_positve, self.train_img_negative, self.train_label_anchor, self.train_label_positive, self.train_label_negative = self.genTriplets(batch)
           
            for i, batch in enumerate(self.val_dl):  # batches loop
                self.val_img_anchor, self.val_img_positive, self.val_img_negative, self.val_label_anchor, self.val_label_positive, self.val_label_negative = self.genTriplets(batch)
            
            self.val_sample_size = self.val_img_anchor.shape[0]
            self.train_sample_size = self.train_img_anchor.shape[0]
            train_losses=[]
            val_losses=[]
            counter = 0
            pth = './siameseEfigi_triplet/efigicheckpoint'
            early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
            print("Training starting")
            for ep in range(epochs):  # epochs loop
                counter += 1
                train_loss = self.train_network_triplets(model, optimizer, scheduler, self.train_img_anchor, self.train_img_positve, self.train_img_negative,
                                                   self.train_label_anchor, self.train_label_positive,
                                                    self.train_label_negative, self.train_sample_size, tr='train' )
                train_losses.append(train_loss)
                    
                
                val_loss = self.train_network_triplets(model, optimizer, scheduler, self.val_img_anchor, self.val_img_positive, self.val_img_negative, self.val_label_anchor,
                                                 self.val_label_positive, self.val_label_negative, self.val_sample_size, tr='val')
                val_losses.append(val_loss)
                scheduler.step(val_loss) 
                    
                print('\nepoch: {}'.format(ep))
                print('Train loss: {}, Val loss: {}'.format(train_loss, val_loss) )
                
                early_stopping(val_loss, model, ep)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                pickle.dump(train_losses,open('./triplet_train_losses_siamese_efigi','wb'))
                pickle.dump(val_losses,open('./triplet_val_losses_siamese_efigi','wb'))
            print("Training complete")  
            
        elif self.modelType == 'pair':
            for i, batch in enumerate(self.train_dl):  # batches loop
                self.train_img_1, self.train_img_2, self.train_label_1, self.train_label_2, self.train_label_c = self.genPair(batch)
            
            for i, batch in enumerate(self.val_dl):  # batches loop
                self.val_img_1, self.val_img_2, self.val_label_1, self.val_label_2, self.val_label_c = self.genPair(batch)
                
            self.val_sample_size = self.val_img_1.shape[0]
            self.train_sample_size = self.train_img_1.shape[0]
            train_losses=[]
            val_losses=[]
            counter = 0
            pth = './siameseEfigi_pair/efigicheckpoint'
            early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
            print("Training starting")
            for ep in range(epochs):  # epochs loop
                counter += 1
                #train_loss = train_network(train_img_1, train_img_2, train_label_1, 
                #                               train_label_2, train_label_c, train_sample_size)
                train_loss = self.train_network(model, optimizer, scheduler, self.train_img_1, self.train_img_2, self.train_label_1,
                                                self.train_label_2, self.train_label_c, self.train_sample_size, tr='train')
                train_losses.append(train_loss)
                    
                val_loss = self.train_network(model, optimizer, scheduler, self.val_img_1, self.val_img_2, self.val_label_1, 
                                             self.val_label_2, self.val_label_c, self.val_sample_size)
                val_losses.append(val_loss)
                scheduler.step(val_loss) 
                    
                print('\nepoch: {}'.format(ep))
                print('Train loss: {}, Val loss: {}'.format(train_loss, val_loss) )
                
                early_stopping(val_loss, model, ep)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                pickle.dump(train_losses,open('./pair_train_losses_siamese_efigi','wb'))
                pickle.dump(val_losses,open('./pair_val_losses_siamese_efigi','wb'))
            print("Training complete")
            
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
            
        return model
    
    def evaluate(self, epochs=5):
        self.model = self.train(epochs=epochs)
        if self.modelType == 'triplet':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            m = self.train_sample_size
            distMatrix = torch.empty((n, m))
            predLabels = torch.empty((n, 1))
            actLabels = torch.empty((0, 1))
            trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            #euclidean_distance = cos(features_1, features_2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = F.normalize(self.model(batch['image'].cuda()), p=2)
                    #actLabels = torch.argmax(batch['labels'], 1).float().reshape(-1,1))
                    actLabels = torch.cat((actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    preds = F.normalize(self.model(self.train_img_anchor[j: j + self.BATCH_SIZE].cuda()), p=2)
                    trainPreds = torch.cat((trainPreds, preds.cpu()), 0)
            for row in range(0, n):
                for col in range(0, m):
                    distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               trainPreds[col].reshape(1,-1).cuda()).cpu()
                    torch.cuda.empty_cache()
            for r in range(0, n):
                ind = torch.argmin(distMatrix[r])
                predLabels[r, 0] = self.train_label_anchor[ind]
                
            preds = predLabels.numpy()
            acts = actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            print('Unseen test set f1_score: {}'.format(f1_score(acts, preds, average='macro')))
            torch.save(self.model, 'efigi_siamese_triplet')
            print('Model saved: efigi_siamese_triplet')
        elif self.modelType == 'pair':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            m = self.train_sample_size
            distMatrix = torch.empty((n, m))
            predLabels = torch.empty((n, 1))
            actLabels = torch.empty((0, 1))
            trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            #euclidean_distance = cos(features_1, features_2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = self.model(batch['image'].cuda())
                    #actLabels = torch.argmax(batch['labels'], 1).float().reshape(-1,1))
                    actLabels = torch.cat((actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    preds = self.model(self.train_img_1[j: j + self.BATCH_SIZE].cuda())
                    trainPreds = torch.cat((trainPreds, preds.cpu()), 0)
            for row in range(0, n):
                for col in range(0, m):
                    distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               trainPreds[col].reshape(1,-1).cuda()).cpu()
                    torch.cuda.empty_cache()
            for r in range(0, n):
                ind = torch.argmin(distMatrix[r])
                predLabels[r, 0] = self.train_label_1[ind]
                
            preds = predLabels.numpy()
            acts = actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set (contrastive) accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            print('Unseen test set (contrastive) f1_score: {}'.format(f1_score(acts, preds, average='macro')))
            torch.save(self.model, 'efigi_siamese_pair')
            print('Model saved: efigi_siamese_pair')
        return self.model, predLabels, actLabels
    
    def evaluate2(self, epochs=5):
        self.model = self.train(epochs=epochs)
        if self.modelType == 'triplet':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            unique_labs = torch.unique(self.train_label_anchor)
            m = self.train_sample_size
            n_classes = len(unique_labs)
            avg_fts = torch.empty((n_classes, self.outputSize))
            distMatrix = torch.empty((n, n_classes))
            predLabels = torch.empty((n, 1))
            actLabels = torch.empty((0, 1))
            trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = self.model(batch['image'].cuda())
                    actLabels = torch.cat((actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    preds = self.model(self.train_img_anchor[j: j + self.BATCH_SIZE].cuda())
                    trainPreds = torch.cat((trainPreds, preds.cpu()), 0)
            
            for lab in unique_labs:
                inds = np.where(self.train_label_anchor == lab)[0]
                fts = torch.empty((0, self.outputSize))
                for k in range(0, inds.shape[0], self.BATCH_SIZE):
                    with torch.no_grad():
                        #preds = model(train_img_anchor[inds[j: j + BATCH_SIZE]].cuda())
                        predss = trainPreds[inds[k: k + self.BATCH_SIZE]]
                        fts = torch.cat((fts, predss.cpu()), 0)
                avg = torch.mean(fts, 0).reshape(1,-1)
                avg_fts = torch.cat((avg_fts, avg ), 0)
            
            for row in range(0, n):
                for col in range(0, n_classes):
                    distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               avg_fts[col].reshape(1,-1).cuda()).cpu()
                    #torch.cuda.empty_cache()
            for r in range(0, n):
                predLabels[r, 0] = torch.argmin(distMatrix[r])
                
            preds = predLabels.numpy()
            acts = actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            print('Unseen test set f1_score: {}'.format(f1_score(acts, preds, average='macro')))
            torch.save(self.model, 'efigi_siamese_triplet')
            print('Model saved: efigi_siamese_triplet')
        
        return self.model, testPreds, trainPreds, distMatrix, self.train_label_anchor, predLabels, actLabels, avg_fts

      
    
    
