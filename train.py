import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config as c
from model import get_cs_flow_model, load_model, save_model, FeatureExtractor, nf_forward
from utils import *
import csv

def train(train_loader, test_loader):
    model = get_cs_flow_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    z_obs = Score_Observer('AUROC')

    train_start = int(round(time.time() * 1000)) # ms

    f = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for epoch in range(c.meta_epochs):
        # train some epochs
        model.train()

        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            epoch_start = int(round(time.time() * 1000)) # ms
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()

                inputs, labels = preprocess_batch(data)  # move to device and reshape
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()

            epoch_end = int(round(time.time() * 1000)) # ms
            mean_train_loss = np.mean(train_loss)
            if c.verbose:
                wr.writerow([mean_train_loss])
                print('Epoch: {:d}.{:d} \t train loss: {:.4f} \t time: {:d}ms'.format(epoch, sub_epoch, mean_train_loss, epoch_end-epoch_start))

        train_end = int(round(time.time() * 1000)) # ms

        print('\nTotal train time: {:d}ms\n'.format(train_end-train_start))
        
        model.eval()

        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()

        predict_time = 0

        with torch.no_grad():
            count = 0
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                if not c.pre_extracted:
                    inputs = fe(inputs)
                count += 16
                predict_start = int(round(time.time() * 1000)) # ms
                z, jac = nf_forward(model, inputs)
                z_concat = t2np(concat_maps(z))
                # print("z_concat", z_concat)
                score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
                # print("score", score)
                predict_end = int(round(time.time() * 1000)) # ms
                predict_time += predict_end-predict_start
                loss = get_loss(z, jac)
                test_z.append(score)
                test_loss.append(t2np(loss))
                
                test_labels.append(t2np(labels))
            print("count:", count)
        # save_model(model, "bb_" + c.modelname)


        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f} \t prediction time: {:d}ms'.format(epoch, test_loss, predict_time))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        maxEpoch = z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                     print_score=c.verbose or epoch == c.meta_epochs - 1)

        # 최대 AUROC일 때 모델 저장
        if maxEpoch == epoch:
            # save_model(model, c.modelname + "_e:" + str((epoch+1)*c.sub_epochs))
            save_model(model.state_dict(), "S_" + c.modelname + "_e:" + str((epoch+1)*c.sub_epochs))
            #save_model(model, c.modelname)

    f.close()

    # if c.save_model:
    #    model.to('cpu')
    #    save_model(model, c.modelname)
        #evaluate(model, test_loader)

    return z_obs.max_score, z_obs.last, z_obs.min_loss_score

def save_state():
    model = load_model(c.modelname)
    model.to(c.device)
    # 전체 모델 대신 parameter만 저장하는 방식
    save_model(model.state_dict(), "S_" + c.modelname)

# save_state()