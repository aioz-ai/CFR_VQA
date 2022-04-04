"""
Coarse to Fine Adaption Flow
HuyTran
https://arxiv.org/abs/1805.07932

This code is written by Huy Tran.
"""
import torch
import torch.nn as nn
from src.attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from src.classifier import SimpleClassifier
from src.fc import FCNet
from src.bc import BCNet
from src.counting import Counter
from src.utils import tfidf_loading
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from pretrain.qa_answer_table import load_lxmert_qa
MAX_VQA_LENGTH = 20


class BanFusion(nn.Module):
    def __init__(self, dataset, b_att, b_net, q_prj, c_prj, counter, gamma, omega):
        super(BanFusion, self).__init__()
        self.dataset = dataset
        self.glimpse = gamma
        self.omega = omega
        self.v_att = b_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.counter = counter

    def forward(self, mod1, b, mod2):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        if self.counter is not None:
            boxes = b[:, :, :4].transpose(1, 2)
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(mod1, mod2)  # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(mod1, self.omega * mod2, att[:, g, :, :])
            mod1 = self.omega * self.q_prj[g](b_emb[g].unsqueeze(1)) * mod1 + mod1

            if self.counter is not None:
                atten, _ = logits[:, g, :, :].max(2)
                embed = self.counter(boxes, atten)

            if self.counter is not None:
                mod1 = mod1 + self.c_prj[g](embed).unsqueeze(1)
        return mod1


class Lxmert_Model(nn.Module):
    def __init__(self, lxmert_model, classifier):
        super(Lxmert_Model, self).__init__()
        self.lxrt_encoder = lxmert_model
        self.logit_fc = classifier

    def forward(self, v, b, q):
        q_emb, v, logits = self.lxrt_encoder(q, (v, b[:, :, :4]))
        return q_emb, v, self.logit_fc(logits)

    def dim(self):
        return 768


class CFRF_Model(nn.Module):
    def __init__(self, dataset, args, lxmert_encoder, w_emb, q_emb, sw_emb, s_emb, ew_emb, qe_joint, vs_joint, vq_joint,
                 classifier, gamma):
        super(CFRF_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.sw_emb = sw_emb
        self.s_emb = s_emb
        self.ew_emb = ew_emb
        self.qe_joint = qe_joint
        self.vs_joint = vs_joint
        self.vq_joint = vq_joint
        self.gamma = gamma
        self.lxmert_encoder = lxmert_encoder
        self.adapted_w = nn.Parameter(torch.ones(2, dataset.num_ans_candidates))
        self.classifier = classifier

    def forward(self, v, b, q, s, e, w):

        # Question embedding
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)

        # Semantic embedding
        sw_emb = self.sw_emb(w)
        s_emb = self.s_emb.forward_all(sw_emb)

        # Entity embedding
        e_emb = self.ew_emb(e)

        # Lxmert encoder
        _, _, lxmert_logit = self.lxmert_encoder(v, b, s)

        # Question-entity
        q_emb = self.qe_joint(q_emb, b, e_emb)

        # Image-semantic
        v = self.vs_joint(s_emb, b, v)

        # Image question
        q_emb = self.vq_joint(q_emb, b, v)

        ban_logits = self.classifier(q_emb.sum(1))

        logits = torch.cat([ban_logits.unsqueeze(1), lxmert_logit.unsqueeze(1)], 1)
        adapted_w = torch.softmax(self.adapted_w, 0)

        logits = torch.mul(logits, adapted_w.unsqueeze(0)).sum(1)

        return logits, lxmert_logit, ban_logits


def build_ban_fusion(dataset, args, dim_1, dim_2, gamma, omega, priotize_using_counter=False):
    """
    :param dim_1: 1st modality dimension.
    :param dim_2: 2nd modality dimension.
    :param gamma: number of residual.
    :param omega: ratio fusion between two modalities.
    :return: joint representation between two modalities.
    """
    b_net = []
    q_prj = []
    c_prj = []
    b_att = BiAttention(dim_1, dim_2, dim_1, gamma)

    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter

    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    for i in range(args.gamma):
        b_net.append(BCNet(dim_1, dim_2, dim_1, None, k=1))
        q_prj.append(FCNet([dim_1, dim_1], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, dim_1], 'ReLU', .0))

    if use_counter or priotize_using_counter:
        counter = Counter(objects)
    else:
        counter = None
    return BanFusion(dataset, b_att, b_net, q_prj, c_prj, counter, gamma, omega)


def build_lxmert(dataset, args):
    lxrt_encoder = LXRTEncoder(
        args,
        max_seq_length=MAX_VQA_LENGTH
    )
    h_dim = lxrt_encoder.dim
    logit_fc = nn.Sequential(
        nn.Linear(h_dim, h_dim * 2),
        GeLU(),
        BertLayerNorm(h_dim * 2, eps=1e-12),
        nn.Linear(h_dim * 2, dataset.num_ans_candidates)
    )
    if args.load_lxmert_qa is not None:
        logit_fc.apply(lxrt_encoder.model.init_bert_weights)
        load_lxmert_qa(args.load_lxmert_qa, lxrt_encoder, logit_fc,
                       label2ans=dataset.label2ans, device=args.device)

    return Lxmert_Model(lxrt_encoder, logit_fc)


def build_CFRF_Model(dataset, args):
    lxrt_encoder = build_lxmert(dataset, args)

    # Initial question embedding
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    # Initial stat-word embedding
    sw_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    s_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    # Initial entity embedding
    ew_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)

    if hasattr(args, 'tfidf'):
        if args.dataset == 'GQA':
            w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data/gqa')
            sw_emb = tfidf_loading(args.tfidf, sw_emb, args, 'data/gqa')
            ew_emb = tfidf_loading(args.tfidf, ew_emb, args, 'data/gqa')

        elif args.dataset == 'VQA':
            w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data/vqa')
            sw_emb = tfidf_loading(args.tfidf, sw_emb, args, 'data/vqa')
            ew_emb = tfidf_loading(args.tfidf, ew_emb, args, 'data/vqa')

    qe_joint = build_ban_fusion(dataset, args, args.num_hid, 600, gamma=1, omega=args.omega_q)
    vs_joint = build_ban_fusion(dataset, args, args.num_hid, dataset.v_dim, gamma=1, omega=args.omega_v)
    vq_joint = build_ban_fusion(dataset, args, args.num_hid, args.num_hid, args.gamma, omega=1, priotize_using_counter=False)
    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)

    return CFRF_Model(dataset, args, lxrt_encoder, w_emb, q_emb, sw_emb, s_emb, ew_emb, qe_joint, vs_joint, vq_joint,
                        classifier, args.gamma)
