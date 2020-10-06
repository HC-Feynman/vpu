import argparse
from vpu import *

parser = argparse.ArgumentParser(description='Pytorch Variational Positive Unlabeled Learning')
parser.add_argument('--dataset', default='cifar10',
                    choices=['cifar10', 'fashionMNIST', 'stl10', 'avila', 'pageblocks', 'grid'])
parser.add_argument('--gpu', type=int, default=9)
parser.add_argument('--val-iterations', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--num_labeled', type=int, default=3000, help="number of labeled positive samples")
parser.add_argument('--learning-rate', type=float, default=3e-5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--mix-alpha', type=float, default=0.3, help="parameter in Mixup")
parser.add_argument('--lam', type=float, default=0.03, help="weight of the regularizer")

args = parser.parse_args()

if args.dataset == 'cifar10':
    from model.model_cifar import NetworkPhi
    from dataset.dataset_cifar import get_cifar10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 1, 8, 9])
elif args.dataset == 'fashionMNIST':
    from model.model_fashionmnist import NetworkPhi
    from dataset.dataset_fashionmnist import get_fashionMNIST_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1, 4, 7])
elif args.dataset == 'stl10':
    from model.model_stl import NetworkPhi
    from dataset.dataset_stl import get_stl10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 2, 3, 8, 9])
elif args.dataset == 'pageblocks':
    from model.model_vec import NetworkPhi
    from dataset.dataset_pageblocks import get_pageblocks_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[2, 3, 4, 5])
elif args.dataset == 'grid':
    from model.model_vec import NetworkPhi
    from dataset.dataset_grid import get_grid_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1])
elif args.dataset == 'avila':
    from model.model_vec import NetworkPhi
    from dataset.dataset_avila import get_avila_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=['A'])
else:
    assert False
args = parser.parse_args()


def main(config):
    # set up cuda if it is available
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    # set up the loaders
    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx = get_loaders(batch_size=config.batch_size,
                                                                                       num_labeled=config.num_labeled,
                                                                                       positive_label_list=config.positive_label_list)
    elif config.dataset in ['avila', 'pageblocks', 'grid']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(batch_size=config.batch_size,
                                                                                  num_labeled=config.num_labeled,
                                                                                  positive_label_list=config.positive_label_list)
    loaders = (p_loader, x_loader, val_p_loader, val_x_loader, test_loader)

    # please read the following information to make sure it is running with the desired setting
    print('==> Preparing data')
    print('    # train data: ', len(x_loader.dataset))
    print('    # labeled train data: ', len(p_loader.dataset))
    print('    # test data: ', len(test_loader.dataset))
    print('    # val x data:', len(val_x_loader.dataset))
    print('    # val p data:', len(val_p_loader.dataset))

    # something about saving the model
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    # call VPU
    run_vpu(config, loaders, NetworkPhi)


if __name__ == '__main__':
    main(args)
