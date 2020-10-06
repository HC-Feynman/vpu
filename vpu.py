import math
from utils.checkpoint import *
from utils.func import *
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def run_vpu(config, loaders, NetworkPhi):
    """
    run VPU.

    :param config: arguments.
    :param loaders: loaders.
    :param NetworkPhi: class of the model.
    """

    lowest_val_kl = math.inf
    highest_test_acc = -1

    # set up loaders
    (p_loader, x_loader, val_p_loader, val_x_loader, test_loader) = loaders

    # set up model \Phi
    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10']:
        model_phi = NetworkPhi()
    elif config.dataset in ['pageblocks', 'grid', 'avila']:
        input_size = len(p_loader.dataset[0][0])
        model_phi = NetworkPhi(input_size=input_size)
    if torch.cuda.is_available():
        model_phi = model_phi.cuda()

    # set up the optimizer
    lr_phi = config.learning_rate
    opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    for epoch in range(config.epochs):

        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        # train the model \Phi
        phi_loss, kl_loss, reg_loss, phi_p_mean, phi_x_mean = train(config, model_phi, opt_phi, p_loader, x_loader)

        # evaluate the model \Phi
        val_kl, test_acc, test_auc = evaluate(model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch,
                                              phi_loss, kl_loss, reg_loss)

        # assessing performance of the current model and decide whether to save it
        is_val_kl_lowest = val_kl < lowest_val_kl
        is_test_acc_highest = test_acc > highest_test_acc
        lowest_val_kl = min(lowest_val_kl, val_kl)
        highest_test_acc = max(highest_test_acc, test_acc)
        if is_val_kl_lowest:
            test_auc_of_best_val = test_auc
            test_acc_of_best_val = test_acc
            epoch_of_best_val = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_phi.state_dict(),
            'optimizer': opt_phi.state_dict(),
        }, is_val_kl_lowest, is_test_acc_highest, config=config)

    # inform users model in which epoch is finally picked
    print('Early stopping at {:}th epoch, test AUC : {:.4f}, test acc: {:.4f}'.format(epoch_of_best_val,
                                                                                      test_auc_of_best_val,
                                                                                      test_acc_of_best_val))


def train(config, model_phi, opt_phi, p_loader, x_loader):
    """
    One epoch of the training of VPU.

    :param config: arguments.
    :param model_phi: current model \Phi.
    :param opt_phi: optimizer of \Phi.
    :param p_loader: loader for the labeled positive training data.
    :param x_loader: loader for training data (including positive and unlabeled)
    """

    # setup some utilities for analyzing performance
    phi_p_avg = AverageMeter()
    phi_x_avg = AverageMeter()
    phi_loss_avg = AverageMeter()
    kl_loss_avg = AverageMeter()
    reg_avg = AverageMeter()

    # set the model to train mode
    model_phi.train()

    for batch_idx in range(config.val_iterations):

        try:
            data_x, _ = x_iter.next()
        except:
            x_iter = iter(x_loader)
            data_x, _ = x_iter.next()

        try:
            data_p, _ = p_iter.next()
        except:
            p_iter = iter(p_loader)
            data_p, _ = p_iter.next()

        if torch.cuda.is_available():
            data_p, data_x = data_p.cuda(), data_x.cuda()

        # calculate the KL divergence
        data_all = torch.cat((data_p, data_x))
        output_phi_all = model_phi(data_all)
        log_phi_all = output_phi_all[:, 1]
        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        output_phi_x = output_phi_all[idx_x]
        kl_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)

        # perform Mixup and calculate the regularization
        target_x = output_phi_x[:, 1].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda() if torch.cuda.is_available() else target_p
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(config.mix_alpha, config.mix_alpha)
        lam = m.sample()
        data = lam * data_x + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out_log_phi_all = model_phi(data)
        reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

        # calculate gradients and update the network
        phi_loss = kl_loss + config.lam * reg_mix_log
        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()

        # update the utilities for analysis of the model
        reg_avg.update(reg_mix_log.item())
        phi_loss_avg.update(phi_loss.item())
        kl_loss_avg.update(kl_loss.item())
        phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        phi_x_avg.update(phi_x.mean().item(), len(phi_x))

    return phi_loss_avg.avg, kl_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg


def evaluate(model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch, phi_loss, kl_loss, reg_loss):
    """
    evaluate the performance on test set, and calculate the KL divergence on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param test_loader: loader for the test set (fully labeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to kl_loss + reg_loss.
    :param kl_loss: KL divergence of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()

    # calculate KL divergence of the validation set consisting of PU data
    val_kl = cal_val_kl(model_phi, val_p_loader, val_x_loader)

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (data, _) in enumerate(x_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        log_max_phi = max(log_max_phi, model_phi(data)[:, 1].max())

    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            log_phi = model_phi(data)[:, 1]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))
    pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
    log_phi_all = np.array(log_phi_all.cpu().detach())
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all)
    test_auc = roc_auc_score(target_all, log_phi_all)
    print('Train Epoch: {}\t\tphi_loss: {:.4f}\tkl_div: {:.4f}\t\treg_loss: {:.4f}\tTest accuracy: {:.4f}\tVal kl{:.4f}' \
          .format(epoch, phi_loss, kl_loss, reg_loss, test_acc, val_kl))
    return val_kl, test_acc, test_auc


def cal_val_kl(model_phi, val_p_loader, val_x_loader):
    """
    Calculate KL divergence on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate KL divergence
    with torch.no_grad():
        for idx, (data_x, _) in enumerate(val_x_loader):
            if torch.cuda.is_available():
                data_x = data_x.cuda()
            output_phi_x_curr = model_phi(data_x)
            if idx == 0:
                output_phi_x = output_phi_x_curr
            else:
                output_phi_x = torch.cat((output_phi_x, output_phi_x_curr))
        for idx, (data_p, _) in enumerate(val_p_loader):
            if torch.cuda.is_available():
                data_p = data_p.cuda()
            output_phi_p_curr = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_x = output_phi_x[:, 1]
        kl_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - torch.mean(log_phi_p)
        return kl_loss.item()
