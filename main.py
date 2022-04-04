"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from src.FFOE.dataset import Dictionary, GQAFeatureDataset
import src.FFOE.base_model as base_model
from src.FFOE.train import train
import src.utils as utils
try:
    import _pickle as pickle
except:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE CFRF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/GQA',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=12,
                        help='the number of epoches')
    parser.add_argument('--lr', default=7e-4, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--update_freq', default='4', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Data
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')

    # Choices of models
    parser.add_argument('--model', type=str, default='CFRF_Model', choices=['CFRF_Model'],
                        help='the model we use')
    parser.add_argument('--dataset', type=str, default='GQA', choices=['GQA'],
                        help='Dataset to train and evaluate')

    # INTERACTION LEARNING COMPONENTS HYPER-PARAMETERS------------------------------------------------------------------
    # BAN
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')


    # CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--testing', action='store_true', default=False,
                        help='for fast testing 1 epoch')
    parser.add_argument('--print_interval', default=200, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')
    parser.add_argument('--weight_init', type=str, default='none', choices=['none', 'kaiming_normal'],
                        help='dynamic weighting with Kaiming normalization')

    # Bounding box set
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    # Stat word
    parser.add_argument('--num_stat_word', default=30, type=int, metavar='N',
                        help='number of statistical word')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Framework hyper-params
    parser.add_argument('--activation', type=str, default='swish', choices=['relu', 'swish'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.45, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Debugging
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)
    parser.add_argument('--lxmert_lr', default=5e-5, type=float, metavar='lr',
                        help='initial learning rate')

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Fine-tuning arguments
    parser.add_argument('--omega_q', type=float, default=0.1,
                        help='omega for control the effect of question instructions')
    parser.add_argument('--omega_v', type=float, default=0.01,
                        help='omega for control the effect of image semantics')
    parser.add_argument('--fusion_ratio', type=float, default=0.1,
                        help='ratio for control the effect of adapted weight')

    parser.add_argument('--topk', default='6', type=int)

    # Return args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'GQA':
        dictionary = Dictionary.load_from_file('data/gqa/dictionary.pkl')
        train_dset = GQAFeatureDataset(args, 'train', dictionary, adaptive=True)
        val_dset = GQAFeatureDataset(args, 'val', dictionary, adaptive=True)
    else:
        raise BaseException("Dataset name not found!")

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)
    model = model.to(device)

    if args.multiGPU:
        model = nn.DataParallel(model)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input, map_location=device)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = None
        epoch = model_data['epoch'] + 1

    if args.use_both:  # use train & val splits to optimize
        trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate,
                                  pin_memory=True)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate,
                                  pin_memory=True)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate,
                                 pin_memory=False)
        # eval_loader = None

    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)

