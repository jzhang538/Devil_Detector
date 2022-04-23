import os
from torch import autograd, optim
from models import *
from helper import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torchvision.utils as vutils
import pickle
import datetime
import time
import pandas as pd
pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:.3f}'.format(x)
pd.options.display.max_columns = 20
pd.options.display.width = 100
import torch.nn as nn
criterion = nn.CrossEntropyLoss().cuda()



def test_in(model, test_loader, num_classes, df_test, df_test_avg, epoch):
    max_evi = 0.0
    min_evi = 10000


    v_ls = []
    with torch.no_grad():
        df_tmp = pd.DataFrame(
            columns=['idxs_mask', 'in_ent', 'in_vac', 'in_dis', 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis'])

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # onehot = torch.eye(num_classes, device=device)[target]
            alpha = model(data)  # TODO
            p = alpha / alpha.sum(1, keepdim=True)

            max_alpha = torch.max(alpha, -1)[0]
            max_evi = max(torch.max(max_alpha), max_evi)
            min_evi = min(torch.min(max_alpha), min_evi)
            for a in max_alpha.cpu().numpy():
                v_ls.append(a)

            pred = p.argmax(dim=1, keepdim=True)
            idxs_mask = pred.eq(target.view_as(pred)).view(-1)
            ent = cal_entropy(p)
            disn = getDisn(alpha)

            vac_in = (num_classes / torch.sum(alpha, dim=1))
            succ_ent = ent[idxs_mask]
            fail_ent = ent[~idxs_mask]
            succ_dis = disn[idxs_mask]
            fail_dis = disn[~idxs_mask]

            df_tmp.loc[len(df_tmp)] = [i.tolist() for i in
                                       [idxs_mask, ent, vac_in, disn, succ_ent, fail_ent, succ_dis, fail_dis]]

        # print(df_test_batch.keys())
        in_score = df_tmp.sum()
                                    #26032
        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_ent'], in_score['fail_ent'])
        bnd_dect_ent = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_dis'], in_score['fail_dis'])
        bnd_dect_dis = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        df_test.loc[len(df_test)] = [epoch, *in_score, bnd_dect_ent, bnd_dect_dis]
        df_test_avg.loc[len(df_test_avg)] = [epoch, *in_score.apply(np.average), bnd_dect_ent['auroc'],
                                             bnd_dect_dis['auroc']]
        return df_test, df_test_avg, in_score, np.array(v_ls)

'''
First pretrain a classifier to reach a good accuracy. 
Then feed the pre-trained classifier into the wgan framework to calibrate its uncertainty. 
We already have pre-trained weights in the folder 'pretrain'
'''

def train(model, optimizer, train_loader, poisoned_test_loader, test_loader, num_classes, epochs=300, use_softmax=False):
    # for i in range(epochs):
    #     # if i + 1 in [50, 75, 90]:
    #     if i + 1 in [100, 150]:
    #         for group in optimizer.param_groups:
    #             group['lr'] *= .1
    for i in range(epochs):
        model.train()
        model.apply(update_bn_stats)
        accuracy = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            onehot = torch.eye(num_classes, device=device)[target]

            optimizer.zero_grad()

            ##########################
            # (1) Loss from P real
            ###########################

            alpha = model(data)
            if use_softmax:
                acc_loss = criterion(alpha, target)
                pred = alpha.argmax(dim=1, keepdim=True)
            else:
                s = alpha.sum(1, keepdim=True)
                # s = torch.sum(alpha, dim=1, keepdim=True)
                p = alpha / s
                # vac_in = num_classes / s
                acc_loss = torch.sum((onehot - p) ** 2, dim=1).mean() + \
                           torch.sum(p * (1 - p) / (s + 1), axis=1).mean()
                pred = p.argmax(dim=1, keepdim=True)
            
            accuracy += pred.eq(target.view_as(pred)).sum().item() / data.size(0)
            loss1 = acc_loss
            loss1.backward()
            optimizer.step()
        print('epoch, %d\tTrain acc: %.4f ' % (i, accuracy / len(train_loader)), end='\t')


        model.eval()
        model.apply(freeze_bn_stats)

        with torch.no_grad():
            accuracy = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                # onehot = torch.eye(num_classes, device=device)[target]
                alpha = model(data)  # TODO
                p = alpha / alpha.sum(1, keepdim=True)

                pred = p.argmax(dim=1, keepdim=True)
                # idxs_mask = pred.eq(target.view_as(pred)).view(-1)
                accuracy += pred.eq(target.view_as(pred)).sum().item() / data.size(0)
            print('Clean testing acc: %.4f' % (accuracy / len(test_loader)), end='\t')

        with torch.no_grad():
            accuracy = 0
            for batch_idx, (data, target) in enumerate(poisoned_test_loader):
                data, target = data.to(device), target.to(device)
                # onehot = torch.eye(num_classes, device=device)[target]
                alpha = model(data)  # TODO
                p = alpha / alpha.sum(1, keepdim=True)

                pred = p.argmax(dim=1, keepdim=True)
                # idxs_mask = pred.eq(target.view_as(pred)).view(-1)
                accuracy += pred.eq(target.view_as(pred)).sum().item() / data.size(0)
            print('Poisoned testing acc: %.4f' % (accuracy / len(poisoned_test_loader)))

    return model

def params_init():
    matplotlib.rc('xtick', labelsize=50)
    matplotlib.rc('ytick', labelsize=50)
    matplotlib.rc('lines',markersize=20)
    font = {'family' : 'serif',
                'size'   : 50,
                }
    matplotlib.rc('font', **font)

    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    params = {'legend.fontsize': 50}  
    plt.rcParams.update(params)

def main(
    gamma = 0.01,
    network = 'resnet20',
    dataname = 'CIFAR10',
    grad_clip = 1,
    n_epochs = 60,
    batch_size = 128,
    poison_rate = 0.1,
    use_softmax = False,
    lambda_term=10,
    weight_decay=0.0001,
    nz=128,
    log_interval=5,
    use_augment=True,
    use_validation = False,
    root = 'results',
    train_flag = True,
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldname = '{}_beta_{}_time_{}'.format(network, gamma, current_time)

    path = '{}/{}/{}'.format(root, dataname, foldname)
    # print('\nPATH: ', path)
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    num_classes, split = (10, 10000)

    in_channel = 1 if dataname in ['MNIST'] else 3

    args =['path', 'dataname', 'network', 'gamma', 'batch_size', 'n_epochs', 'grad_clip','nz'\
    , 'use_augment', 'lambda_term', 'weight_decay', 'log_interval', 'use_validation', 'num_classes', 'split', 'train_flag', 'poison_rate', 'use_softmax']

    record ={}
    print('---------%s-----------\n'%current_time)
    for arg in args:
        print('{}:\t{}'.format(arg, eval(arg)), end='\n')
        record[arg] = eval(arg)
    print('\n')
    with open('{}/log.txt'.format(path), "a") as file:
        print('---------%s-----------\n' % current_time, file=file)
        for arg in args:
            print('{}:\t{}'.format(arg, eval(arg)), end='\n', file=file)
        print('\n', file=file)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    if dataname == 'CIFAR10':
        model_name = 'resnet20_cifar10_eval.pt'
        model = ENN(resnet20(10)).to(device)
        if not train_flag:
            model.classifier.load_state_dict(torch.load('ckpt/{}'.format(model_name)))

        train_loader, poisoned_test_loader, test_loader = get_backdoored_cifar(dataname, batch_size, poison_rate, use_augment=use_augment, use_split= False)
    if dataname == 'MNIST':
        model_name = 'lenet_mnist_eval.pt'
        model = ENN(lenet5(1)).to(device)
        if not train_flag:
            model.classifier.load_state_dict(torch.load('ckpt/{}'.format(model_name)))

        train_loader, poisoned_test_loader, test_loader = get_backdoored_mnist(dataname, batch_size, poison_rate, use_augment=use_augment, use_split= False)



    # #######################
    # Train #
    # #######################
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if train_flag==True:
        model = train(model, optimizer, train_loader, poisoned_test_loader, test_loader, num_classes, epochs=30, use_softmax=use_softmax)
        torch.save(model.classifier.state_dict(), 'ckpt/{}'.format(model_name))



    df_test = pd.DataFrame(
        columns=['epoch', 'idxs_mask', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_roc', 'bnd_dis_roc'])
    df_test_avg = pd.DataFrame(
        columns=['epoch', 'test_acc', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_auroc', 'bnd_dis_auroc'])
    total_start = time.time()

    # #######################
    #  Test clean #
    # #######################
    model.eval()
    model.apply(freeze_bn_stats)
    df_test, df_test_avg, in_score, clean_v_ls = test_in(model, test_loader, num_classes, df_test, df_test_avg, 0)
    df = df_test_avg.tail(1)
    test_log ='Clean Testing Dataset:\tacc: {:.3f},\t'\
              'ent: {:.3f}({:.3f}/{:.3f}),\t'\
              'vac: {:.3f},\t'\
              'disn: {:.3f}({:.3f}/{:.3f}),\t'\
              'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                            df['in_ent'].iloc[0],
                                                            df['succ_ent'].iloc[0],
                                                            df['fail_ent'].iloc[0],
                                                            df['in_vac'].iloc[0],
                                                            df['in_dis'].iloc[0],
                                                            df['succ_dis'].iloc[0],
                                                            df['fail_dis'].iloc[0],
                                                            df['succ_ent'].iloc[0],
                                                            df['bnd_ent_auroc'].iloc[0],
                                                            df['bnd_dis_auroc'].iloc[0])
    print(test_log)

    # #######################
    #  Test poison #
    # #######################
    model.eval()
    model.apply(freeze_bn_stats)
    df_test, df_test_avg, in_score, bd_v_ls = test_in(model, poisoned_test_loader, num_classes, df_test, df_test_avg, 0)
    df = df_test_avg.tail(1)
    test_log ='Poisoned Testing Dataset:\tacc: {:.3f},\t'\
              'ent: {:.3f}({:.3f}/{:.3f}),\t'\
              'vac: {:.3f},\t'\
              'disn: {:.3f}({:.3f}/{:.3f}),\t'\
              'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                            df['in_ent'].iloc[0],
                                                            df['succ_ent'].iloc[0],
                                                            df['fail_ent'].iloc[0],
                                                            df['in_vac'].iloc[0],
                                                            df['in_dis'].iloc[0],
                                                            df['succ_dis'].iloc[0],
                                                            df['fail_dis'].iloc[0],
                                                            df['succ_ent'].iloc[0],
                                                            df['bnd_ent_auroc'].iloc[0],
                                                            df['bnd_dis_auroc'].iloc[0])
    print(test_log)

    # #######################
    # Draw #
    # #######################    
    params_init()
    plt.figure(figsize=(21,14))
    color_ls = ['r', 'g']
    alpha_ls = [0.2, 0.6]
    label_ls = ['Clean dataset','Backdoored dataset']

    n, bins, patches = plt.hist(clean_v_ls, bins=80, density=True, color=color_ls[0], alpha=alpha_ls[0], label=label_ls[0])
    plt.plot(bins[:-1],n,'--',linewidth=1)
    n, bins, patches = plt.hist(bd_v_ls, bins=80, density=True, color=color_ls[1], alpha=alpha_ls[1], label=label_ls[1])
    plt.plot(bins[:-1],n,'--',linewidth=1)

    xlabel = 'Value of maximum evidence'
    ylabel = 'Fraction of sample in each dataset'
    plt.legend(loc='upper left',shadow=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pp = PdfPages('./dist.pdf')
    pp.savefig()
    pp.close()
    plt.close()

    # #######################
    # # Finish #
    # #######################
    total_time = str(datetime.timedelta(seconds=time.time() - total_start))
    print('----- finished -------- elapsed: %s'%total_time)
    for arg in args:
        print('{}:\t{}'.format(arg, eval(arg)), end=', ')
        record[arg] = eval(arg)
    print('\n')
    return record, model
