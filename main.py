
import torch
from torch.autograd import Variable
import os 
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from loss import *
from data_utils import custom_dset, collate_fn
import time
from tensorboardX import SummaryWriter

writer = SummaryWriter()



def train(epochs, model, trainloader, crit, optimizer,
         scheduler, save_step, weight_decay):

    for e in range(epochs):
        print('*'* 10)
        print('Epoch {} / {}'.format(e + 1, epochs))
        model.train()
        start = time.time()
        loss = 0.0
        total = 0.0
        for i, (img, score_map, geo_map, training_mask) in enumerate(trainloader):
            scheduler.step()
            optimizer.zero_grad()
    
            img = Variable(img.cuda())
            score_map = Variable(score_map.cuda())
            geo_map = Variable(geo_map.cuda())
            training_mask = Variable(training_mask.cuda())
            f_score, f_geometry = model(img)
            loss1 = crit(score_map, f_score, geo_map, f_geometry, training_mask)
            
            loss += loss1.data[0]
            
            loss1.backward()
            optimizer.step()
        
        during = time.time() - start
        print("Loss : {:.6f}, Time:{:.2f} s ".format(loss/len(trainloader), during))
        print()
        writer.add_scalar('loss', loss / len(trainloader), e)
        
        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(e + 1))
        

def main():
    root_path = '/home/mathu/Documents/express_recognition/data/telephone_txt/result/'
    train_img = root_path + 'print_pic'
    train_txt = root_path + 'print_txt'
    # root_path = '/home/mathu/Documents/express_recognition/data/icdar2015/'
    # train_img = root_path + 'train2015'
    # train_txt = root_path + 'train_label'

    trainset = custom_dset(train_img, train_txt)
    trainloader = DataLoader(
                    trainset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    model = East()
    model = model.cuda()
    model.load_state_dict(torch.load('./checkpoints_total/model_1440.pth'))

    crit = LossFunc()
    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                                #  weight_decay=1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, 
                                    gamma=0.94)   
    
    train(epochs=1500, model=model, trainloader=trainloader,
          crit=crit, optimizer=optimizer,scheduler=scheduler, 
          save_step=20, weight_decay=weight_decay)

    write.close()

if __name__ == "__main__":
    main()
    