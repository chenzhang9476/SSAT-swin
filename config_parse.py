import argparse

def get_configs ():
    parser = argparse.ArgumentParser()

    # Training arguments/home/czhang5/Data/model/Swin-Unet/data/Synapse/pre_train/train_npz
    parser.add_argument('--root_path', type=str,
                         default='data/Synapse', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synpase', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=3, help='output channel of network')
    parser.add_argument('--output_dir', default='./result/model', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=5000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', default='./configs/swin_tiny_patch4_window7_224_lite.yaml', type=str, required=False,
                        metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # Test arguments
    parser.add_argument('--volume_path', type=str,
                        default='./data/Synapse/',
                        help='root dir for validation volume data')
    parser.add_argument('--test_save_dir', type=str, default='./result/prediction',
                        help='saving prediction as nii!')
    parser.add_argument('--is_savenii', default=True, action="store_true",
                        help='whether to save results during inference')

    args = parser.parse_args()
    return args
