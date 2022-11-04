from argparse import ArgumentParser
import train_and_eval_lib


def parse_args():
    
    parser = ArgumentParser(description='Pytorch implemention of <model name>')
    
    parser.add_argument('--model_dir', type=str, default='./model_save', help='Path to output model directory where event and checkpoint files will be written.')
    
    parser.add_argument('--root', type=str, default=None, help='Path to root data path.')
    
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    
    parser.add_argument('--category', type=int, default=0, help='category')
    
    parser.add_argument('--is_validation', type=bool, default=False, help='validation')
    
    parser.add_argument('--aug_list', type=str, default='hflip+jitter,hflip+jitter+cutout0.3', help='input augmentation list')
    
    parser.add_argument('--input_shape', nargs='+', type=int, default=[32, 32, 3], help='data input shape')
    
    parser.add_argument('--distaug_type', type=int, help='number of distribution augmentation')
    
    parser.add_argument('--net_type', type=str, default='resnet18', help='network type')
    
    parser.add_argument('--net_width', default=1, help='network width (# channels)')
    
    parser.add_argument('--head_dims', nargs='+', type=int, default=[512], help='MLP architecture')
    
    parser.add_argument('--latent_dim', type=int, default=2, help='number of hidden units for FC layer')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--force_init', type=bool, default=False, help='force training from scratch')
    
    parser.add_argument('--optim_type', type=str, default='sgd', help='stochastic optimizer')
    
    parser.add_argument('--sche_type', type='str', default='cos', help='learning rate scheduler')
    
    parser.add_argument('--sched_freq', type=str, default='epoch', help='update frequency. `step` or `epoch`')
    
    parser.add_argument('--sched_step_size', type=int, default=1, help='step size for step LR')
    
    parser.add_argument('--sched_gamme', type=float, default=0.995, help='gamma for step LR')
    
    parser.add_argument('--sched_min_rate', type=float, default=0.0, help='minimum rate for cosine LR')
    
    parser.add_argument('--sched_level', type=int, default=7,  help='level for half-cosine cycle')
    
    parser.add_argument('--learning rate', type=float, default=0.3, help='learning rate')
    
    parser.add_argument('--weight decay', type=float, default=0.00001, help='weight decay')
    
    parser.add_argument('--regularize_bn', type=bool, default=False, help='regularize BN parameters')
    
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    parser.add_argument('--nesterov', type=bool, default=False, help='nesterov')
    
    parser.add_argument('--num_epoch', type=int, default=2048, help='number of training epochs')
    
    parser.add_argument('--num_batch', type=int, default=0, help='number of batches per epoch')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    
    parser.add_argument('--ckpt_prefix', type=str, default='', help='checkpoint prefix')
    
    parser.add_argument('--ckpt_epoch', type=int, default=32, help='frequency to save checkpoints')
    
    parser.add_argument('--file_path', type=str, default=None, help='file path')
    
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = train_and_eval_lib.get_trainer(args)
    
    trainer.config()
    trainer.train()
    
    
    