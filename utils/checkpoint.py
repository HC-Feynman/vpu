from utils.func import *
import shutil
import torch


def get_checkpoint_path(config):
    """
    return the path of saving current model.
    """
    checkpoint_path = os.path.join(os.getcwd(), config.dataset, 'P=' + str(config.positive_label_list),
                                   'lr=' + str(config.learning_rate), 'lambda=' + str(config.lam),
                                   'alpha=' + str(config.mix_alpha))
    return checkpoint_path


def save_checkpoint(state, is_lowest_on_val, is_highest_on_test, config, filename='checkpoint.pth.tar'):
    """
    Save the current model to the checkpoint_path

    :param state: information of the model and training.
    :param is_lowest_on_val: indicating whether the current model has the lowest KL divergence on the validation set.
    :param is_highest_on_test: indicating whether the current model has the highest test accuracy.
    :param config: arguments.
    :param filename: name of the file that saves the model.
    """
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_lowest_on_val:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_lowest_on_val.pth.tar'))
    if is_highest_on_test:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_highest_on_test.pth.tar'))
