import os
os.chdir(r'K:\Isoprene')
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from models import get_model, set_parameter_not_requires_grad
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV



#####
def set_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
class CustomLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, x, y):
        if self.loss_type == 'mse':
            return nn.functional.mse_loss(x, y)
        elif self.loss_type == 'mae':
            return nn.functional.l1_loss(x, y)
        elif self.loss_type == 'both':
            return 0.5 * nn.functional.mse_loss(x, y) + 0.5 * nn.functional.l1_loss(x, y)
        elif self.loss_type == 'smoothl1':
            return nn.functional.smooth_l1_loss(x, y)


def adjust_learning_rate(optimizer, epoch, args):
    '''
    args: {'lradj': ['type1', 'type2'], 'lr': 1e-5}
    
    '''
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args['lradj'] == 'type1':
        lr_adjust = {epoch: args['learning_rate'] * (0.5 ** (epoch // 1))}
    elif args['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('\nUpdating learning rate to {}'.format(lr))



### 模型预训练
citys = ['GZ_GYQ', 'WH_U', 'WH_B', 'BJ_HK', 'CD_HK', 'SH', 'NJ', 'CQ_JC', 
         'HK_HJ', 'HK_DY']

def pretrain(res, city = 'NJ', mode = 'MLP', dropout = 0.2, seed = 1024, 
             act_name = 'relu', loss_type = 'both'):
    train_x, test_x = res['train_x'], res['test_x']
    train_y, test_y = res['train_y'], res['test_y']
    tf_train_x, tf_test_x = res['tf_train_x'], res['tf_test_x']
    tf_train_y, tf_test_y = res['tf_train_y'], res['tf_test_y']
    x = torch.cat([train_x, test_x])
    y = torch.cat([train_y, test_y])
    tf_x = torch.cat([tf_train_x[0], tf_test_x[0]])
    tf_y = torch.cat([tf_train_y[0], tf_test_y[0]])

    model = get_model(mode, train_x.shape[1], dropout, act_name = act_name).cuda()
    criterion = CustomLoss(loss_type)
    lr = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    epochs = 18000
    loss_tot = []
    set_torch_seed(seed)
    model.train()
    for epoch in tqdm(range(epochs)):
        predict = model(x.cuda())
        loss = criterion(predict, y.cuda())
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_tot.append(loss.item())
        if (epoch + 1) % (epochs // 6) == 0:
            print('\nepoch: ', epoch, 'loss: ', loss.item())
            args = {'learning_rate': lr, 'lradj': 'type1'}
            adjust_learning_rate(optim, (epoch + 1) // (epochs // 6), args)
    torch.save(model.state_dict(), 'model_result/{}_pretrain_dropout={}_{}.pth'.format(mode, int(dropout*100), city))
    
    set_torch_seed(seed)
    model.eval()
    stat_res = []
    with torch.no_grad():
        train_pred = model(train_x.cuda()).detach().cpu().numpy()
        test_pred = model(test_x.cuda()).detach().cpu().numpy()
        
        tf_pred = model(tf_x.cuda()).detach().cpu().numpy()


### 模型迁移
# 模型迁移
def transfer_ph(res, city, varbs, mode = 'MLP', dropout = 0.2, seed = 1024, loss_type = 'both',
                kfold = True, vi_month = False, physical = True, weight_decay = 0.008, 
                act_name = 'relu', varb_ph = ['veh_emi', 'VI']):
    tf_train_x_kfd, tf_test_x_kfd = res['tf_train_x'], res['tf_test_x']
    tf_train_y_kfd, tf_test_y_kfd = res['tf_train_y'], res['tf_test_y']
    ind = []
    for varb in varb_ph:
        ind.append(varbs.index(varb))
    loss_tot_kfold = []
    m = len(tf_train_x_kfd) if kfold else 1
    for i in range(m):
        tf_train_x, tf_test_x = tf_train_x_kfd[i].cuda(), tf_test_x_kfd[i].cuda()
        tf_train_y, tf_test_y = tf_train_y_kfd[i].cuda(), tf_test_y_kfd[i].cuda()

        model = get_model(mode, tf_train_x.shape[1], dropout, act_name = act_name).cuda()
        try:
            model.load_state_dict(torch.load('model_result/{}_pretrain_dropout={}_{}.pth'.format(mode, int(100*dropout), city)))
        except:
            model.load_state_dict(torch.load('model_result/{}_pretrain_dropout=30_{}.pth'.format(mode, city)))
        set_parameter_not_requires_grad(model.feature_extract)

        criterion = CustomLoss(loss_type)
        lr = 1e-4
        if physical:
            optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        else:
            optim = torch.optim.Adam(model.parameters(), lr = lr)
        epochs = 24000
        loss_tot, loss_mse_tot, loss_ph_tot, loss_sl_tot = [], [], [], []
        set_torch_seed(seed)
        model.train()
        tf_train_x.requires_grad = True
        for epoch in tqdm(range(epochs)):
            predict = model(tf_train_x)
            loss_mse = criterion(predict, tf_train_y)
            loss_mse_tot.append(loss_mse.item())
            
            grad_output = torch.autograd.grad(outputs = predict.sum(), 
                                              inputs = tf_train_x, 
                                              create_graph = True,
                                              allow_unused = True)[0]
            loss_ph = torch.mean(torch.clamp(-grad_output[:, ind], min = 0))
            loss_ph_tot.append(loss_ph.item())
            
            loss_sl = torch.tensor(0.).cuda()
            for params in model.parameters():
                loss_sl += torch.sum(torch.square(params))
            loss_sl = loss_sl / 20000
            loss_sl_tot.append(loss_sl.item())
            
            if physical:
                loss = loss_mse + loss_ph + loss_sl
            else:
                loss = loss_mse
            loss_tot.append(loss.item())
            
            optim.zero_grad()
            loss.backward(retain_graph = True)
            optim.step()
            num = 1000 if physical else (epochs // 6)
            if (epoch + 1) % num == 0:
                print('\nepoch: ', epoch, 'loss_mse: ', round(loss_mse.item(), 6), 
                      'loss_ph: ', round(loss_ph.item(), 6), 'loss_sl: ', 
                      round(loss_sl.item(), 6))
            if (epoch + 1) % (epochs // 6) == 0:
                args = {'learning_rate': lr, 'lradj': 'type1'}
                adjust_learning_rate(optim, (epoch + 1) // (epochs // 6), args)
        
        fig, axes = plt.subplots(2, 2, figsize = (14, 14), dpi = 300)
        ax = axes.flatten()
        ax[0].plot(range(len(loss_tot)), loss_tot)
        ax[1].plot(range(len(loss_tot)), loss_mse_tot)
        ax[2].plot(range(len(loss_tot)), loss_ph_tot)
        ax[3].plot(range(len(loss_tot)), loss_sl_tot)
        ax[0].set_title('Transfer loss {}'.format(city))
        ax[1].set_title('Transfer mse loss {}'.format(city))
        ax[2].set_title('Transfer physical loss {}'.format(city))
        ax[3].set_title('Transfer structural loss {}'.format(city))
        plt.show()
        
        loss_tot = pd.DataFrame(loss_tot, columns = ['loss_tot'])
        loss_tot['loss_mse'] = loss_mse_tot
        loss_tot['loss_ph'] = loss_ph_tot
        loss_tot['loss_sl'] = loss_sl_tot
        loss_tot['Fold'] = 'KFold' + str(i + 1)
        loss_tot_kfold.append(loss_tot)


##### MLP模型预测
def mlp_train(x, y, city, data_type, mode = 'MLP', dropout = 0.2, seed = 1024, 
              loss_type = 'both', act_name = 'relu'):
    model = get_model(mode, x[0].shape[1], dropout, act_name = act_name).cuda()
    criterion = CustomLoss(loss_type)
    lr = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    
    set_torch_seed(seed)
    loss_tot = []
    for epoch in tqdm(range(18000)):
        model.train()
        predict = model(x[0])
        loss = criterion(predict, y[0])
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_tot.append(loss.item())
        if (epoch + 1) % 3000 == 0:
            print('\nepoch: ', epoch, 'loss: ', loss.item())
            args = {'learning_rate': lr, 'lradj': 'type1'}
            adjust_learning_rate(optim, (epoch + 1) // 3000, args)


def mlp_predict(res, city, mode = 'MLP', dropout = 0.2, seed = 1024, loss_type = 'both',
                kfold = True, act_name = 'relu'):
    train_x, train_y = res['train_x'].cuda(), res['train_y'].cuda()
    tf_train_x_kfd, tf_test_x_kfd = res['tf_train_x'], res['tf_test_x']
    tf_train_y_kfd, tf_test_y_kfd = res['tf_train_y'], res['tf_test_y']

    m = len(tf_train_x_kfd) if kfold else 1
    for i in range(m):
        tf_train_x, tf_test_x = tf_train_x_kfd[i].cuda(), tf_test_x_kfd[i].cuda()
        tf_train_y, tf_test_y = tf_train_y_kfd[i].cuda(), tf_test_y_kfd[i].cuda()
        train_x_ = torch.concat([train_x, tf_train_x], axis = 0)
        train_y_ = torch.concat([train_y, tf_train_y], axis = 0)
        
        # Single
        x, y = [tf_train_x, tf_test_x], [tf_train_y, tf_test_y]
        mlp_train(x, y, city, 'single_kfd{}'.format(i + 1), mode = mode, 
                  dropout = dropout, loss_type = loss_type, seed = seed,
                  act_name = act_name)

        # All
        x, y = [train_x_, tf_test_x], [train_y_, tf_test_y]
        mlp_train(x, y, city, 'all_kfd{}'.format(i + 1), mode = mode, 
                  dropout = dropout, loss_type = loss_type, seed = seed,
                  act_name = act_name)


##### 机器学习模型预测
def ml_predict(res, city, seed = 2543, kfold = True):
    train_x, train_y = res['train_x'], res['train_y']
    tf_train_x_kfd, tf_test_x_kfd = res['tf_train_x'], res['tf_test_x']
    tf_train_y_kfd, tf_test_y_kfd = res['tf_train_y'], res['tf_test_y']
    
    stat_res = []
    m = len(tf_train_x_kfd) if kfold else 1
    for i in range(m):
        tf_train_x, tf_test_x = tf_train_x_kfd[i], tf_test_x_kfd[i]
        tf_train_y, tf_test_y = tf_train_y_kfd[i].flatten(), tf_test_y_kfd[i].flatten()
        x, y = [tf_train_x, tf_test_x], [tf_train_y, tf_test_y]
        
        # Single: XGBoost, RandomForest, GBDT, SVR, LinearRegression
        xgb_param_space = {'n_estimators': [100, 200, 300],
                           'max_depth': [20, 30],
                           'learning_rate': [0.2, 0.5, 0.8, 1],
                           'colsample_bytree': [0.8, 1.0]}
        xgb_sig_reg = xgb.XGBRFRegressor(random_state = seed)
        grid = GridSearchCV(xgb_sig_reg, xgb_param_space, scoring = 'neg_mean_squared_error')
        grid.fit(tf_train_x, tf_train_y)
        xgb_sig_reg = grid.best_estimator_
        
        rf_param_space = {'n_estimators': [100, 200, 300],
                          'min_samples_split': [5, 10, 15, 20],
                          'max_depth': [10, 20]}
        rf_sig_reg = RandomForestRegressor(random_state = seed)
        grid = GridSearchCV(rf_sig_reg, rf_param_space, scoring = 'neg_mean_squared_error')
        grid.fit(tf_train_x, tf_train_y)
        rf_sig_reg = grid.best_estimator_
        
        gbdt_param_space = {'n_estimators': [100, 200, 300],
                            'learning_rate': [0.1, 0.3, 0.6, 0.8, 1]}
        gbdt_sig_reg = GradientBoostingRegressor(random_state = seed)
        grid = GridSearchCV(gbdt_sig_reg, gbdt_param_space, scoring = 'neg_mean_squared_error')
        grid.fit(tf_train_x, tf_train_y)
        gbdt_sig_reg = grid.best_estimator_
        
        svr_param_space = {'C': [1, 5, 10, 100, 1000], 
                            'kernel': ['linear', 'poly', 'rbf']}
        svr_sig_reg = SVR()
        grid = GridSearchCV(svr_sig_reg, svr_param_space, scoring = 'neg_mean_squared_error')
        grid.fit(tf_train_x, tf_train_y)
        svr_sig_reg = grid.best_estimator_
        
        lr_sig_reg = LinearRegression().fit(tf_train_x, tf_train_y)
        
        # All: MLP, XGBoost, RandomForest, GBDT, SVR, LinearRegression
        train_x_ = torch.concat([train_x, tf_train_x], axis = 0)
        train_y_ = torch.concat([train_y.flatten(), tf_train_y], axis = 0)
        x, y = [train_x_, tf_test_x], [train_y_, tf_test_y]
        xgb_all_reg = xgb.XGBRegressor(n_estimators = 500,
                                       random_state = seed).fit(train_x_, train_y_)
        
        rf_all_reg = RandomForestRegressor(n_estimators = 500,
                                           random_state = seed).fit(train_x_, train_y_)
        
        gbdt_all_reg = GradientBoostingRegressor(n_estimators = 500,
                                                 random_state = seed).fit(train_x_, train_y_)
        
        svr_all_reg = SVR().fit(train_x_, train_y_)
        
        lr_all_reg = LinearRegression().fit(train_x_, train_y_)


