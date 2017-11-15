
import torch
from torch.autograd import Variable
import os 
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from loss import *
from data_utils import custom_dset, collate_fn
import time


def train_epoch(epoch, model, trainloader, crit, optimizer, log_step):
    model.train()
    start = time.time()
    loss = 0.0
    total = 0.0
    for i, (img, score_map, geo_map, training_mask) in enumerate(trainloader):
        bs = img.size(0)
        img = Variable(img.cuda())
        score_map = Variable(score_map.cuda())
        geo_map = Variable(geo_map.cuda())
        training_mask = Variable(training_mask.cuda())
        f_score, f_geometry = model(img)
        loss1 = crit(score_map, f_score, geo_map, f_geometry, training_mask)
        loss += loss1
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        total += bs
        
    during = time.time() - start
    print("Loss : {:.6f}, Time:{:.2f} s ".format(loss.data[0]/len(trainloader), during))
    print()
        

def train(epochs, model, trainloader, crit, optimizer, 
             log_step, eval_step, save_step):
    for e in range(epochs):
        print('*'* 10)
        print('Epoch {} / {}'.format(e + 1, epochs))
        train_epoch(e, model, trainloader, crit, optimizer, log_step)
        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(e + 1))
        

def main():
    root_path = '/home/test/Documents/express_recognition/data/icdar2015/'
    train_img = root_path + 'train2015'
    train_txt = root_path + 'train_label'
    trainset = custom_dset(train_img, train_txt)
    trainloader = DataLoader(
                    trainset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    model = East()
    model = model.cuda()
    crit = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    train(epochs=500, model=model, trainloader=trainloader,
          crit=crit, optimizer=optimizer,log_step=100,
         eval_step=5, save_step=5)

if __name__ == "__main__":
    main()
    