from train_backdoor_model import *



# pretrain myself
_, _ = main(dataname = 'MNIST', gamma=0.1, network='resnet20', grad_clip= 1, n_epochs=50, train_flag=True)
# _, _ = main(dataname = 'MNIST', gamma=0.1, network='resnet20', grad_clip= 1, n_epochs=50, train_flag=False)