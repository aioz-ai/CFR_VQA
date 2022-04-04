"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch
import src.utils as utils
import contextlib
from collections import defaultdict, OrderedDict
from src.meters import AverageMeter, TimeMeter

class Trainer(object):
    """
    Main class for training.
    """
    def __init__(self, args, model, criterion, optimizer=None, bert_optimizer=None):
        self.args = args

        # copy model and criterion on current device
        self.model = model.to(self.args.device)
        self.criterion = criterion.to(self.args.device)

        # initialize meters
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        self.meters['wall'] = TimeMeter()      # wall time in seconds

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        if optimizer is not None:
            self._optimizer = optimizer
        self._bert_optim = bert_optimizer
        self.total_loss = 0.0
        self.train_score = 0.0
        self.total_norm = 0.0
        self.count_norm = 0.0

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def bert_optimizer(self):
        if self._bert_optim is None:
            self._build_optimizer()
        return self._bert_optim

    def _build_optimizer(self):
        # self._optimizer = optim.build_optimizer(self.args, self.model.parameters())
        # self._optimizer =
        # self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)
        pass

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        # if distributed_utils.is_master(self.args):  # only save one checkpoint
        #     extra_state['train_meters'] = self.meters
        #     utils.save_state(
        #         filename, self.args, self.model, self.criterion, self.optimizer,
        #         self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
        #     )
        pass

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.model)

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                # self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                    self.optimizer.load_state_dict(last_optim_state)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters = extra_state['train_meters']
            del extra_state['train_meters']

        return extra_state

    def train_step(self, sample, update_params=True):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        # seed = self.args.seed + self.get_num_updates()
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # forward and backward pass
        sample = self._prepare_sample(sample)
        loss, sample_size, oom_fwd, batch_score, batch_question_type_score = self._forward(sample)
        oom_bwd = self._backward(loss)

        # buffer stats and logging outputs
        # self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['sample_sizes'].append(1)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)

            # aggregate stats and logging outputs
            grad_denom = sum(sample_sizes)

            grad_norm = 0
            try:
                # all-reduce and rescale gradients, then take an optimization step
                grad_norm = self._all_reduce_and_rescale(grad_denom)
                self._opt()

                # update meters
                if grad_norm is not None:
                    self.meters['gnorm'].update(grad_norm)
                    self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)

                self.meters['oom'].update(ooms_fwd + ooms_bwd)

            except OverflowError as e:
                self.zero_grad()
                print('| WARNING: overflow detected, ' + str(e))

            self.clear_buffered_stats()

            return loss, grad_norm, batch_score, batch_question_type_score
        else:
            return None  # buffering updates

    def _forward(self, sample, eval=False):
        # prepare model and optimizer
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss = None
        sample_size = 0
        oom = 0
        batch_score = 0
        batch_question_type_score = None
        if sample is not None:
            try:
                with torch.no_grad() if eval else contextlib.ExitStack():
                    # calculate loss and sample size
                    answers = sample[2]
                    teacher_logits = sample[-1]

                    if self.args.model == 'CFRF_Model':
                        fusion_preds, lxmert_preds, ban_preds = self.model(sample[6], sample[7], sample[1], sample[8],
                                                                           sample[4], sample[0])

                        # Fusion loss
                        loss_1 = self.criterion(fusion_preds.float(), answers)
                        loss_1 /= answers.size()[0]

                        # Ban loss
                        loss_2 = self.criterion(ban_preds.float(), answers)
                        loss_2 /= answers.size()[0]

                        # Lxmert loss
                        loss_3 = self.criterion(lxmert_preds.float(), answers)
                        loss_3 /= answers.size()[0]

                        # Total loss
                        loss = self.args.fusion_ratio * loss_1 + loss_2 + loss_3
                        final_preds = fusion_preds
                    batch_score = compute_score_with_logits(final_preds, answers.data).sum()

            except RuntimeError as e:
                if not eval and 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    loss = None
                else:
                    raise e
        return loss, len(sample[0]), oom, batch_score, batch_question_type_score  # TODO: Not sure about sample size, need to recheck

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _all_reduce_and_rescale(self, grad_denom):
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(self._flat_grads)

        # rescale and clip gradients
        flat_grads.div_(grad_denom)
        grad_norm = utils.clip_grad_norm_(flat_grads, self.args.clip_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

        return grad_norm

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
                #raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         #'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset+numel].view_as(g))
            offset += numel

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        if self._bert_optim is not None:
            self._bert_optim.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        # self.lr_scheduler.step_update(self._num_updates)

    def zero_grad(self):
        self.optimizer.zero_grad()
        if self._bert_optim is not None:
            self._bert_optim.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

