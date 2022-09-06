import os
from argparse import ArgumentParser
import model as md
from utils import create_link
import test as tst


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset_dir', type=str, default='/home/hs21/data/dhcp/dhcp-2d')
    parser.add_argument('--labels_path', type=str, default='/home/hs21/data/dhcp/labels_ants_full.pkl')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/dhcp')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    parser.add_argument('--aug_rician_noise', type=int, default=0, help='whether to use rician noise augmentation' 'options: 0, ([0, 10])')
    parser.add_argument('--aug_bspline_deformation', type=float, default=0, help='whether to use bspline_deformation' 'options: 0, ([5],[0, 2])')
    parser.add_argument('--resize_image', type=bool, default=False)
    parser.add_argument('--resize_size', type=float, default=(256,256))
        
    args = parser.parse_args()
    return args


def main():
  args = get_args()

  # create_link(args.dataset_dir)

  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      args.gpu_ids.append(id)
  print(not args.no_dropout)
  if args.training:
      print("Training")
      model = md.cycleGAN(args)
      model.train(args)
  if args.testing:
      print("Testing")
      tst.test(args)


if __name__ == '__main__':
    main()