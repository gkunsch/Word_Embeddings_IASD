import os
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim

class Trainer(object):

    def __init__(self, _src_emb,_tgt_emb,src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self._src_emb=_src_emb
        self._tgt_emb=_tgt_emb
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        

        # optimizers
        
        
        self.map_optimizer = optim.SGD(self.mapping.parameters(),lr=0.1)
        
        self.dis_optimizer = optim.SGD(self.discriminator.parameters(),lr=0.1)
        

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
    
    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch (embeddings) / output target (labels).
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self._src_emb), len(self._src_emb))
        src_ids = torch.LongTensor(bs).random_(len(self._src_emb) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self._tgt_emb) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y
    
    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data.item())

        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

        #Orthogonalization
        if self.params.beta > 0:
            W = self.mapping.weight.data
            beta = self.params.beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        return 2 * self.params.batch_size
        
    
    def procrustes(self):

        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def update_lr(self, metric):
        """
        Update learning rate when using SGD.
        """
        
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            
            print("Decreasing learning_rate :",old_lr,"->",new_lr)
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and metric >= -1e7:
            if metric < self.best_valid_metric:
                print("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (metric, self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    print("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, metric,n_refinement=0,procrustes=False):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if metric > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = metric
            print('* Best value :',metric)
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join("","dumps/"+self.params.src_lang+'_'+self.params.tgt_lang+'_mapping.pth')
            print('* Saving the mapping to %s ...' % path)
            torch.save(W, path)
        if  procrustes:
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join("","dumps/"+self.params.src_lang+'_'+self.params.tgt_lang+str(n_refinement)+ '_mapping_proscrutes.pth')
            print('* Saving the mapping to %s ...' % path)
            torch.save(W, path)
            
    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join("","dumps/"+self.params.src_lang+'_'+self.params.tgt_lang+'_mapping.pth')
        print('* Reloading the best model from %s ...' %self.params.src_lang+'_'+self.params.tgt_lang+path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))