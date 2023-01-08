
import torch
from torch import nn
from .functions import load_embeddings,normalize_embeddings

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        #params
        self.emb_dim = 300
        self.dis_layers = 2
        self.dis_hid_dim = 2048
        self.dis_dropout = 0.
        self.dis_input_dropout = 0.1
        #Build the network
        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


def build_model(params, with_dis,n_embeddings=10000):
    """
    Build all components of the model.
    """
    # source embeddings
    _src_emb, params.src_id2word, params.src_word2id = load_embeddings(params.src_emb,nmax=n_embeddings)
    _src_emb=normalize_embeddings(_src_emb)
    
    _src_emb_torch=torch.from_numpy(_src_emb).float()
    _src_emb_torch=_src_emb_torch.cuda()

    src_emb = nn.Embedding(len(_src_emb_torch), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb_torch)
    
    # target embeddings
    if params.tgt_lang:
        _tgt_emb,params.tgt_id2word,params.tgt_word2id = load_embeddings(params.tgt_emb,nmax=n_embeddings)
        _tgt_emb=normalize_embeddings(_tgt_emb)

        _tgt_emb_torch=torch.from_numpy(_tgt_emb).float()
        _tgt_emb_torch=_tgt_emb_torch.cuda()

        tgt_emb = nn.Embedding(len(_tgt_emb_torch), params.emb_dim, sparse=True)

        tgt_emb.weight.data.copy_(_tgt_emb_torch)
    else:
        tgt_emb = None

    # mapping
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    #Init mapping to identity matrix
    mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator()

    # cuda
    if params.cuda:
        src_emb=src_emb.cuda()
        if params.tgt_lang:
            tgt_emb=tgt_emb.cuda()
        mapping=mapping.cuda()
        if with_dis:
            discriminator=discriminator.cuda()


    return _src_emb,_tgt_emb,src_emb, tgt_emb, mapping, discriminator