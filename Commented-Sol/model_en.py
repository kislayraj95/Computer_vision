# Importing the pytorch module
import torch

# Creating a class encoder that aims to learn how to reconstruct data
# torch.nn.module contains different classess that help us build NN models
class Encoder(torch.nn.Module):
    # the __init__ function is a constructor used to initialize variables
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, class_num, sememe_num, lexname_num, rootaffix_num, encoder):
        # calls the first base class init method in torch.nn.module
        super().__init__()

        # Creating a neural network by initializing parameters
        self.vocab_size = vocab_size
        self.embed_dim = 300
        self.hidden_dim = 768
        self.layers = layers
        self.class_num = class_num
        self.sememe_num = sememe_num
        self.lexname_num = lexname_num
        self.rootaffix_num = rootaffix_num

        # torch.nn.Embedding just creates a Lookup Table, to get the word embedding given a word index
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False

        # Each channel will be zeroed out independently on every forward call
        self.embedding_dropout = torch.nn.Dropout(0.2)
        self.encoder = encoder

        # Creates single-single layer feed forward network with n inputs and m output
        self.fc = torch.nn.Linear(self.hidden_dim, self.embed_dim)
        self.fc_s = torch.nn.Linear(self.hidden_dim, self.sememe_num)
        self.fc_l = torch.nn.Linear(self.hidden_dim, self.lexname_num)
        self.fc_r = torch.nn.Linear(self.hidden_dim, self.rootaffix_num)
        # Computes the cross entropy loss between input and target
        self.loss = torch.nn.CrossEntropyLoss()

        # It replaces all the negative elements in the input tensor with 0
        self.relu = torch.nn.ReLU()

    # The forward function computes output Tensors from input Tensors
    def forward(self, operation, x=None, w=None, ws=None, wl=None, wr=None, msk_s=None, msk_l=None, msk_r=None, mode=None):
        # x: T(bat, max_word_num)
        # w: T(bat)
        # h: T(bat, max_word_num, 768)
        # A boolean tensor that is True where input is greater than x and False elsewhere.
        attention_mask = torch.gt(x, 0).to(torch.int64)
        
        # Generates an “n-layer” coding of the given input and attempts to reconstruct the input using the code generated
        h = self.encoder(x, attention_mask=attention_mask)[0]
        #h = self.encoder(x)[0]
        #h = self.embedding_dropout(h)
        
        ## word prediction
        # vd: T(bat, embed_dim)
        #h_1 = torch.max(h, dim=1)[0]
        #h_1 = h[:,0,:] # The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
        h_1 = self.embedding_dropout(h[:,0,:])
        vd = self.fc(h_1)
        # score0: T(bat, 30000) = [bat, emb] .mm [class_num, emb].t()
        score0 = vd.mm(self.embedding.weight.data[[range(self.class_num)]].t())
        # BertVec: 30000, class_num: 50477+2
        score = score0
        
        if 's' in mode:
            ## sememe prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc_s(h)
            # sem_score: T(bat, sememe_num)
            sem_score, _ = torch.max(pos_score, dim=1)
            #sem_score = torch.sum(pos_score * alpha, 1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score_s = self.relu(sem_score.mm(ws.t()))
            #----------add mean sememe score to those who have no sememes
            # mean_sem_sc: T(bat)
            mean_sem_sc = torch.mean(score_s, 1)
            # msk: T(class_num)
            score_s = score_s + mean_sem_sc.unsqueeze(1).mm(msk_s.unsqueeze(0))
            #----------
            score = score + score_s
        if 'r' in mode:
            ## root-affix prediction
            pos_score_ = self.fc_r(h)
            ra_score, _ = torch.max(pos_score_, dim=1)
            score_r = self.relu(ra_score.mm(wr.t()))
            mean_ra_sc = torch.mean(score_r, 1)
            score_r = score_r + mean_ra_sc.unsqueeze(1).mm(msk_r.unsqueeze(0))
            score = score + score_r
        if 'l' in mode:
            ## lexname prediction
            lex_score = self.fc_l(h_1)
            score_l = self.relu(lex_score.mm(wl.t()))
            mean_lex_sc = torch.mean(score_l, 1)
            score_l = score_l + mean_lex_sc.unsqueeze(1).mm(msk_l.unsqueeze(0))
            score = score + score_l
        #_, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, w)
            return loss, score #, indices
        elif operation == 'test':
            return score#, indices
