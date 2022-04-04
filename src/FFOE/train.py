"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import torch
import src.utils as utils
import torch.nn as nn
from src.FFOE.trainer import Trainer
from lxrt.optimization import BertAdam
warmup_updates = 4000


def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    device = args.device
    lr_default = args.lr
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10, 20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 0
    grad_clip = args.clip_norm
    bert_optim = None

    utils.create_dir(output)

    if args.model == 'CFRF_Model':
        batch_per_epoch = int(len(train_loader.dataset) / args.batch_size) + 1
        t_total = int(batch_per_epoch * args.epochs)
        ignored_params = list(map(id, model.lxmert_encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        bert_optim = BertAdam(list(model.lxmert_encoder.parameters()),
                              lr=args.lxmert_lr,
                              warmup=0.1,
                              t_total=t_total)

        optim = torch.optim.Adamax(list(base_params), lr=lr_default)

    else:
        raise BaseException("Model not found!")

    N = len(train_loader.dataset)
    num_batches = int(N / args.batch_size + 1)


    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    trainer = Trainer(args, model, criterion, optim, bert_optim)
    update_freq = int(args.update_freq)
    wall_time_start = time.time()
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        train_question_type_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        if args.model == 'CFRF_Model':
            if epoch < len(gradual_warmup_steps):
                trainer.optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
                logger.write('gradual warmup lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            elif epoch in lr_decay_epochs:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay_rate
                logger.write('decreased lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            else:
                logger.write('lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
        else:
            raise BaseException("Model not found!")

        for i, (v, b, w, e, attr, q, s, a, ans) in enumerate(train_loader):
            v = v.to(device)
            b = b.to(device)
            e = e.to(device)
            w = w.to(device)
            # attr = attr.to(device)
            q = q.to(device)
            a = a.to(device)
            ans = ans.to(device)
            sample = [w, q, a, attr, e, ans, v, b, s]

            if i < num_batches - 1 and (i + 1) % update_freq > 0:
                trainer.train_step(sample, update_params=False)
            else:
                loss, grad_norm, batch_score, batch_question_type_score = trainer.train_step(sample, update_params=True)
                total_norm += grad_norm
                count_norm += 1

                total_loss += loss.item()
                train_score += batch_score
                num_updates += 1
                if num_updates % int(args.print_interval / update_freq) == 0:
                    print("Iter: {}, Loss {:.4f}, Norm: {:.4f}, Total norm: {:.4f}, Num updates: {}, Wall time: {:.2f},"
                          "ETA: {}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm, num_updates,
                                           time.time() - wall_time_start, utils.time_since(t, i / num_batches)))
                    if args.testing:
                        break

        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)
        train_question_type_score = 100 * train_question_type_score / (num_updates * args.batch_size)

        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            eval_cfrf_score, fg_score, coarse_score, ens_score, bound = evaluate(model, eval_loader, args)
            trainer.model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f, question type score: %.2f' %
                     (total_loss, total_norm/count_norm, train_score, train_question_type_score))
        if eval_loader is not None:
            logger.write('\tCFRF score: %.2f (%.2f)' % (100 * eval_cfrf_score, 100 * bound))

        # Save per epoch
        if epoch >= saving_epoch:
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, trainer.optimizer)
            # Save best epoch
            if eval_loader is not None and eval_cfrf_score > best_eval_score:
                model_path = os.path.join(output, 'model_epoch_best.pth')
                utils.save_model(model_path, model, epoch, trainer.optimizer)
                best_eval_score = eval_cfrf_score


def evaluate(model, dataloader, args):
    device = args.device
    cfrf_score = 0
    ens_score = 0
    fg_score = 0
    coarse_score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad():
        for i, (v, b, w, e, attr, q, s, a, ans) in enumerate(dataloader):
            v = v.to(device)
            b = b.to(device)
            w = w.to(device)
            q = q.to(device)
            a = a.to(device)
            e = e.to(device)
            ans = ans.to(device)
            attr = attr.to(device)
            final_preds = None

            if args.model == 'CFRF_Model':
                fusion_preds, lxmert_preds, ban_preds = model(v, b, q, s, e, w)
                ens_preds = fusion_preds + lxmert_preds + ban_preds
                fg_score += compute_score_with_logits(ban_preds, a).sum()
                coarse_score += compute_score_with_logits(lxmert_preds, a).sum()
                ens_score += compute_score_with_logits(ens_preds, a).sum()
                final_preds = fusion_preds
            else:
                raise BaseException("Model not found!")

            batch_score = compute_score_with_logits(final_preds, a).sum()
            cfrf_score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += final_preds.size(0)

    cfrf_score = cfrf_score / len(dataloader.dataset)
    fg_score = fg_score / len(dataloader.dataset)
    coarse_score = coarse_score / len(dataloader.dataset)
    ens_score = ens_score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    return cfrf_score, fg_score, coarse_score, ens_score, upper_bound

