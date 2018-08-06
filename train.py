from argparse import ArgumentParser
from torchvision import models
import deep_learner
import torch


def main():
    args = get_args()
    model = get_model(args.arch)
    trained_model = deep_learner.do_deep_learning(model,
                                                  args.data_dir,
                                                  args.learning_rate,
                                                  args.hidden_units,
                                                  args.epochs,
                                                  args.use_gpu)

    save_checkpoint(trained_model, args.arch, args.save_dir)


def save_checkpoint(model, arch, save_dir):
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'opt_state_dict': model.optimizer.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'arch': arch,
                  'epochs': model.epochs}

    torch.save(checkpoint, '{}/checkpoint_{}'.format(save_dir, arch))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-s", "--save_dir", dest="save_dir",
                        help="where to save the trained model", default='./')
    parser.add_argument('data_dir', type=str, help="Directory containing data for training")
    parser.add_argument("-a", "--arch", dest="arch", help="model architecture to use", default='vgg13')
    parser.add_argument("-r", "--learning_rate", dest="learning_rate", help="learning rate for training", type=float,
                        default=.01),
    parser.add_argument("-u", "--hidden_units", dest="hidden_units", type=int, help="Number of hidden units",
                        default=512)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, help="Number of epochs", default=3)
    parser.add_argument("-g", "--gpu", dest="use_gpu", help="Use gpu for training", type=bool, default=False)
    return parser.parse_args()


def get_model(name):
    return getattr(models, name)(pretrained=True)


main()
