
import io
import numpy as np
import torch
#from numba import jit,cuda

def load_embeddings(emb_path, nmax=50000):
    
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)



    return embeddings, id2word, word2id

def normalize_embeddings(emb):

    for i in range(len(emb)):
        emb[i]=emb[i]/np.linalg.norm(emb[i])
    
    return emb

def get_dico(params):
    f = open(params.dico_eval, "r",encoding="utf-8")
    if params.src_lang=="en":
        src_word2id=params.src_word2id
        tgt_word2id=params.tgt_word2id
    else:
        tgt_word2id=params.src_word2id
        src_word2id=params.tgt_word2id

    eng_fr_dict={}
    fr_eng_dict={}
    for line in f.readlines():

        eng_word=line.split()[0]


        fr_word=line.split()[1]


        if eng_word not in eng_fr_dict.keys():
            eng_fr_dict[eng_word]=[fr_word]
        else:
            eng_fr_dict[eng_word].append(fr_word)

        if fr_word not in fr_eng_dict.keys():
            fr_eng_dict[fr_word]=[eng_word]
        else:
            fr_eng_dict[fr_word].append(eng_word)

    for word in list(eng_fr_dict.keys()):
        if word not in src_word2id.keys():
            del eng_fr_dict[word]
        else:
            if eng_fr_dict[word][0] not in tgt_word2id.keys():
                del eng_fr_dict[word]
    

    for word in list(fr_eng_dict.keys()):
        if word not in tgt_word2id.keys():
            del fr_eng_dict[word]
        else:
            if fr_eng_dict[word][0] not in src_word2id.keys():
                del fr_eng_dict[word]
    
    return eng_fr_dict,fr_eng_dict

#get words with closest embeds to target embedding in tgt_emb space
#@jit(target_backend="cuda")
def get_nn_with_emb(word_emb ,tgt_emb, tgt_id2word, K=5):

    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    #for i, idx in enumerate(k_best):
    #    print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
    return scores[k_best[0]],tgt_id2word[k_best[0]],scores,k_best  #Return le mot le plus proche de l'embedding
    
def eval_w(w,dict,src_embeddings,src_id2word,src_word2id,tgt_embeddings,tgt_id2word,tgt_word2id,test_list):
    print("Eval Running...")

    cost=0
    for word in test_list:
        trad=dict[word][0]
        #cost+=np.linalg.norm(w.dot(src_embeddings[src_word2id[word]])-tgt_embeddings[tgt_word2id[trad]])/len(words)
        cost+=(tgt_embeddings[tgt_word2id[trad]] / np.linalg.norm(tgt_embeddings[tgt_word2id[trad]])).dot(w@(src_embeddings[src_word2id[word]])/ np.linalg.norm(w@(src_embeddings[src_word2id[word]])))/len(test_list)

    return cost
 
import plotly.express as px
import plotly.graph_objects as go


def save_plot(name,cost_list,params):
    fig = px.line(cost_list)
    fig.data[0].name = name
    fig.update_layout(title=name+" evolution")
    fig.update_xaxes(title_text="iterations")
    fig.update_yaxes(title_text=name)
    fig.write_html('dumps/'+params.src_lang+'_'+params.tgt_lang+name+'.html')
