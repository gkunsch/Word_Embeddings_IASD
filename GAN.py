import os
import time
import argparse
import numpy as np
import torch
from src.model import build_model
from src.trainer import Trainer
from src.functions import get_dico, eval_w, save_plot


parser = argparse.ArgumentParser(description='Unsupervised training')

#Arguments
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='fr', help="Target language")

parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
#Refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
#Normalization
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

#Discriminator parameters
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=10000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

#training parameters
parser.add_argument("--adversarial", type=bool, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=100000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

#Dict
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")

params = parser.parse_args()

params.cuda=True
params.emb_dim=300
params.beta=0.001 #beta for proscrutes
params.nb_embeddings=100000
params.dis_most_frequent=30000
#Assertion
assert torch.cuda.is_available()
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert os.path.isfile(params.dico_eval)


_src_emb,_tgt_emb,src_emb, tgt_emb, mapping, discriminator = build_model(params, True,n_embeddings=params.nb_embeddings)
trainer = Trainer(_src_emb,_tgt_emb,src_emb, tgt_emb, mapping, discriminator, params)

eng_fr_dict,fr_eng_dict=get_dico(params)

if params.src_lang=="en":
    params.dico=eng_fr_dict
else:
    params.dico=fr_eng_dict

print(list(params.dico.keys())[0:50])
print(list(params.tgt_word2id.keys())[0:50])

"""
Adversarial training loop
"""

print('---->  TRAINING <----\n\n')
# training loop
cos_list=[]
dis_loss_list=[]
dis_loss_list2=[]
for n_epoch in range(params.n_epochs):
    print('Starting  epoch %i...' % n_epoch)
    tic = time.time()
    n_words_proc = 0
    stats = {'DIS_COSTS': []}
    for n_iter in range(0, params.epoch_size, params.batch_size):
        # discriminator training
        for _ in range(params.dis_steps):
            trainer.dis_step(stats)
        # mapping training (discriminator fooling)
        n_words_proc += trainer.mapping_step(stats)
        dis_loss_list.append(stats['DIS_COSTS'][-1])
        # log stats
        if n_iter % 500 == 0:
            stats_str = [('DIS_COSTS', 'Discriminator loss')]
            stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                         for k, v in stats_str if len(stats[k]) > 0]
            stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
            print(('%06i - ' % n_iter) + ' - '.join(stats_log))
            # reset
            tic = time.time()
            n_words_proc = 0
            dis_loss_list2+=stats['DIS_COSTS']
            stats["DIS_COSTS"]=[]      
    # embeddings / discriminator evaluation

    w = trainer.mapping.weight.data.cpu().numpy()
    cos_sim=eval_w(w,params.dico,_src_emb,params.src_id2word,params.src_word2id,_tgt_emb,params.tgt_id2word,params.tgt_word2id,params.dico.keys())
    print("cos_sim= ",cos_sim)
    cos_list.append(cos_sim)
    trainer.save_best(cos_sim)
    print('End of epoch %i.\n\n' % n_epoch)

    # update the learning rate (stop if too small)
    trainer.update_lr(cos_sim)
    if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
        print('Learning rate < 1e-6. BREAK.')
        break

save_plot("Cosine similarity",cos_list,params)
save_plot("Discriminator Loss",dis_loss_list,params)
save_plot("Discriminator Loss all steps",dis_loss_list2,params)

"""
Refinement
"""


if params.n_refinement > 0:
    # Get the best mapping according to VALIDATION_METRIC
    print('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()

    
    # training loop
    for n_iter in range(params.n_refinement):
        W = trainer.mapping.weight.data
        w = trainer.mapping.weight.data.cpu().numpy()
        print('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        X=_src_emb
        Y=w@X.T
        
        # apply the Procrustes solution
        #trainer.procrustes()
        U, S, V_t = np.linalg.svd(Y@X, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

        w = trainer.mapping.weight.data.cpu().numpy() 
        cos_sim=eval_w(w,params.dico,_src_emb,params.src_id2word,params.src_word2id,_tgt_emb,params.tgt_id2word,params.tgt_word2id,params.dico.keys())

        print("cos_sim= ",cos_sim)
        # embeddings evaluation

        trainer.save_best(cos_sim,params.n_refinement,procrustes=True)
        print('End of refinement iteration %i.\n\n' % n_iter)


