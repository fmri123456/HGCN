import os
import time
import copy
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
import scipy as scio
import utils.hypergraph_utils
import feature_ex

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg1 = get_config('config/config1.yaml')
cfg2 = get_config('config/config2.yaml')
# initialize data
data_dir1 = cfg1['modelnet40_ft']
data_dir2 = cfg2['modelnet40_ft']
fts, lbls, idx_train, idx_test, H1 = \
    load_feature_construct_H(data_dir1,
                             m_prob=cfg1['m_prob'],
                             K_neigs=cfg1['K_neigs'],
                             is_probH=cfg1['is_probH'])
fts1 = fts.reshape(470,116,-1)
fts2, _, _, _, H2 = \
    load_feature_construct_H(data_dir2,
                             m_prob=cfg2['m_prob'],
                             K_neigs=cfg2['K_neigs'],
                             is_probH=cfg2['is_probH'])
fts2 = fts2.reshape(470,45,-1)
Feat = np.concatenate((fts1,fts2),axis=1)
feat = Feat.reshape(470,-1)
H1 = H1.astype('int')
H2 = H2.astype('int')
H = np.bitwise_or(H1, H2) #H = H1 | H2
H = H.astype('float')
G = hgut.generate_G_from_H(H)
n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform data to device
feat = torch.Tensor(feat).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)

loss_list = []
acc_list = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs}')


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # idx = idx_train if phase == 'train' else idx_test
            if phase == 'train':
                idx = idx_train
            else:
                idx = idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(feat, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            with torch.set_grad_enabled(phase == 'val'):
                outputs = model(feat, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * feat.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' :
                loss_list.append(float(epoch_loss))
                acc_list.append(float(epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                TP , TN , FN , FP = utils.hypergraph_utils.stastic_indicators(outputs[idx_test] , lbls[idx_test])
                ACC = (TP + TN) / (TP + TN + FP + FN)
                SEN = TP / (TP + FN)
                SPE = TN / (FP + TN)
                BAC = (SEN + SPE) / 2
        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)


    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # 保存模型训练的最高准确率
    acc = max(acc_list)
    print(f'Best Acc:, {acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,TP,TN,FN,FP,outputs,ACC,SEN,SPE,BAC



print(f"Classification on {cfg1['on_dataset']} dataset!!! class number: {n_class}")
print('Configuration -> Start')
#pp.pprint(cfg)
print('Configuration -> End')

model_ft = HGNN(in_ch=feat.shape[1],
            n_class=n_class,
            n_hid=cfg1['n_hid'],
            dropout=cfg1['drop_out'])
model_ft = model_ft.to(device)

optimizer = optim.Adam(model_ft.parameters(), lr=cfg1['lr'],
                   weight_decay=cfg1['weight_decay'])
# optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                       milestones=cfg1['milestones'],
                                       gamma=cfg1['gamma'])
criterion = torch.nn.CrossEntropyLoss()

model_ft,TP,TN,FN,FP,preds = train_model(model_ft, criterion, optimizer, schedular, cfg1['max_epoch'], print_freq=cfg1['print_freq'])
x = range(1000)
plt.figure(num=1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, loss_list)
plt.figure(num=2)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, acc_list)
plt.show()


hgw1 = model_ft.state_dict()['hgc1.weight']
hgw2 = model_ft.state_dict()['hgc2.weight']
feat_score,feat_idx = feature_ex.feature_extraction(hgw1,hgw2)
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc
#画ROC曲线
y_test = lbls[idx_test].numpy()

y_score = F.softmax(preds,1)
y_score = y_score.detach().numpy()
y_scores = y_score[0:y_score.shape[0],1]
y_scores1 = y_scores[idx_test]
# for i in range(y_score.shape[0]):
#     if y_test[i] == 0:
#         y_scores[i] = y_score[i, 0]
#     else:
#         y_scores[i] = y_score[i ,1]
fpr,tpr,thr = roc_curve(y_test,y_scores1)

roc_auc = auc(fpr,tpr)

lw = 2
plt.figure(figsize=(10 , 10))
plt.plot(fpr , tpr , color='darkorange' ,
         lw=lw , label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0 , 1] , [0 , 1] , color='navy' , lw=lw , linestyle='--')
plt.xlim([0.0 , 1.0])
plt.ylim([0.0 , 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()



np.save('TP.npy',TP)
np.save('FP.npy',FP)
np.save('TN.npy',TN)
np.save('FN.npy',FN)
np.save('TPR.npy',tpr)
np.save('FPR.npy',fpr)
np.save('feat_score',feat_score)
np.save('feat_idx',feat_idx)
np.save('ACC.npy',ACC)
np.save('SEN.npy',SEN)
np.save('SPE.npy',SPE)
np.save('BAC.npy',BAC)