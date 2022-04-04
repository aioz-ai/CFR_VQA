"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.FFOE.dataset import Dictionary, GQAFeatureDataset
import src.FFOE.base_model as base_model
from src.FFOE.train import evaluate
import src.utils as utils

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE CFRF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/GQA',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results/GQA',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=str, default='12',
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')

    # Choices of models
    parser.add_argument('--model', type=str, default='CFRF_Model', choices=['CFRF_Model'],
                        help='the model we use')

    # INTERACTION LEARNING COMPONENTS HYPER-PARAMETERS------------------------------------------------------------------
    # BAN
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')

    #CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - gpu
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    #Bounding box set
    parser.add_argument('--max_boxes', default=40, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')

    # Stat word
    parser.add_argument('--num_stat_word', default=30, type=int, metavar='N',
                        help='number of statistical word')

    # Question embedding
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

    # Data
    parser.add_argument('--dataset', type=str, default='GQA', choices=['GQA'],
                        help='Dataset to train and evaluate')

    # Debugging
    parser.add_argument("--tiny", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

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
    parser.add_argument('--omega_v', type=float, default=0.1,
                        help='omega for control the effect of image semantics')

    parser.add_argument('--topk', type=str, default='6')
    # Return args
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'GQA':
        dictionary = Dictionary.load_from_file('data/gqa/dictionary.pkl')
        eval_dset = GQAFeatureDataset(args, args.split, dictionary, dataroot='data/gqa', adaptive=True)
    else:
        raise BaseException("Dataset name not found!")

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    model_path = args.input + '/model_epoch%s.pth' % args.epoch
    print('loading %s' % model_path)
    model_data = torch.load(model_path, map_location=args.device)

    # Comment because do not use multi gpu
    # model = nn.DataParallel(model)
    model = model.to(args.device)
    model.load_state_dict(model_data.get('model_state', model_data))

    print("Evaluating...")
    model.train(False)
    eval_cfrf_score, _, _, _, bound = evaluate(model, eval_loader, args)
    print('\tCFRF score: %.2f (%.2f)' % (100 * eval_cfrf_score, 100 * bound))
