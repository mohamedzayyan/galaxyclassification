from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from pytorchtools import EarlyStopping

class GZ2:
    def __init__(self,train_dl, val_dl, unseen_dl, model, optimizer, scheduler, criterion, savePath='./models/gz2_resnet50', device='cuda', BATCH_SIZE=64):
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.BATCH_SIZE = BATCH_SIZE
        self.savePath = savePath
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
    
    def train_phase(self, tr='train'):
        if tr == 'train':
            self.model.train()
            dl = self.train_dl
        if tr == 'val':
            self.model.eval()
            dl = self.val_dl
        losses = []
        for i, batch in enumerate(dl):
            inputs, labels = batch['image'], batch['labels'].float().view(-1,37)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()             # 1. Zero the parameter gradients
            outputs = self.model(inputs)           # 2. Run the model
    
            loss = self.criterion(outputs, labels) # 3. Calculate loss
            losses.append(loss.item())
            loss = torch.sqrt(loss)           #    -> RMSE loss
            loss.backward()                   # 4. Backward propagate the loss
            self.optimizer.step()                  # 5. Optimize the network
    
            
            del batch
            del inputs
            del labels
            torch.cuda.empty_cache()
        epoch_loss = np.sqrt(sum(losses) / len(losses))
        return epoch_loss    
    
    def unseen_phase(self):
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.unseen_dl):
            #torch.cuda.empty_cache()
            with torch.no_grad():
                inputs, labels = batch['image'], batch['labels'].float().view(-1,37)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)           # 2. Run the model
    
                loss = self.criterion(outputs, labels) # 3. Calculate loss
                losses.append(loss.item())
                #loss = torch.sqrt(loss) 
                #loss.backward()                   # 4. Backward propagate the loss
                self.optimizer.step() #    -> RMSE loss
                del inputs
                del labels
                del batch                
                del loss    
                torch.cuda.empty_cache()
        epoch_loss = sum(losses)/len(losses)
        return epoch_loss
    
    def train(self, n_epochs=5):      
        self.model = self.model.to(self.device)
        train_losses = []
        val_losses= []
        pth = './gz2_checkpoints/gz2checkpoint'
        early_stopping = EarlyStopping(patience=5, verbose=True, path=pth)
        print("Training beginning")
        for epoch in range(n_epochs):
            train_loss = self.train_phase(tr='train')
            train_losses.append(train_loss)
            print("[TST] Epoch: {} Train Loss: {}".format(epoch+1, train_loss))
            torch.cuda.empty_cache()
            
            val_loss = self.train_phase(tr='val')    
            val_losses.append(val_loss)
            print("[TST] Epoch: {} Val Loss: {}".format(epoch+1, val_loss))
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                ep = epoch - 10
                self.model.load_state_dict(torch.load('./gz2_checkpoints/gz2checkpoint{}.pt'.format(ep)))
                print("Early stopping")
                break
        pickle.dump(train_losses,open('./losses/gz2_train','wb'))
        pickle.dump(val_losses,open('./losses/gz2_val','wb'))
        torch.save(self.model, self.savePath)
        print('Model saved: ' + self.savePath)
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
        print("Training complete, evaluation on unseen set beginning") 
        unseen_loss = self.unseen_phase()
        print('Unseen loss: {}'.format(unseen_loss))
        return None
        
        
