# Importing the pytorch module
import torch

# Creating a class encoder that aims to learn how to reconstruct data
# torch.nn.module contains different classess that help us build NN models
class Encoder(torch.nn.Module):

    # the __init__ function is a constructor used to initialize variables
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, class_num, encoder, sememe_num, chara_num, mode):

        # calls the first base class init method in torch.nn.module
        super().__init__()
        # Creating a neural network by initializing parameters
        self.vocab_size = vocab_size
        self.embed_dim = 200
        self.hidden_dim = 768
        self.layers = layers
        self.class_num = class_num
        self.sememe_num = sememe_num
        self.chara_num = chara_num

        # torch.nn.Embedding just creates a Lookup Table, to get the word embedding given a word index
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False

        # Each channel will be zeroed out independently on every forward call
        self.embedding_dropout = torch.nn.Dropout(0.2)
        self.encoder = encoder

        # Creates single layer feed forward network with n inputs and m output
        self.fc = torch.nn.Linear(self.hidden_dim, self.embed_dim)

        # Computes the cross entropy loss between input and target
        self.loss = torch.nn.CrossEntropyLoss()

        # It replaces all the negative elements in the input tensor with 0
        self.relu = torch.nn.ReLU()

        # Here for some conditional statements
            # We are determining the feature vector size of the h_n(hidden state)
        if 'P' in mode:
            self.fc2 = torch.nn.Linear(self.hidden_dim, 13)
        if 's' in mode:
            self.fc1 = torch.nn.Linear(self.hidden_dim, self.sememe_num)
        if 'c' in mode:
            self.fc3 = torch.nn.Linear(self.hidden_dim, self.chara_num)
        if 'C' in mode:
            self.fc_C1 = torch.nn.Linear(self.hidden_dim, 12)
            self.fc_C2 = torch.nn.Linear(self.hidden_dim, 95)
            self.fc_C3 = torch.nn.Linear(self.hidden_dim, 1425)
        
    # The forward function computes output Tensors from input Tensors
    def forward(self, operation, x=None, w=None, ws=None, wP=None, wc=None, wC=None, msk_s=None, msk_c=None, mode=None):
        # x: T(bat, max_word_num)
        # w: T(bat)
        # x_embedding: T(bat, max_word_num, embed_dim)

        # Convert x to a long integer
        x = x.long()

        # A boolean tensor that is True where input is greater than x and False elsewhere.
        attention_mask = torch.gt(x, 0).to(torch.int64)
        # Generates an “n-layer” coding of the given input and attempts to reconstruct the input using the code generated
        h = self.encoder(x, attention_mask=attention_mask)[0]
        h_1 = self.embedding_dropout(h[:,0,:])
        vd = self.fc(h_1)
        # score0: T(bat, 30000) = [bat, emb] .mm [class_num, emb].t()
        score0 = vd.mm(self.embedding.weight.data[[range(self.class_num)]].t())
        score = score0
        
        if 'C' in mode:
            # scC[i]: T(bat, Ci_size)
            # Cilin's hierarchical classification training is slow, in fact, this is unfair and unbalanced, 
            # because word prediction converges first, and cilin's classification has no effect. The use of 
            # other information also has the same problem, and it does not necessarily converge at the same time! ! !
            scC = [self.fc_C1(h_1), self.fc_C2(h_1), self.fc_C3(h_1)]

            # A tensor filled with the scalar value 0 , with the shape defined by the variable argument size
            score2 = torch.zeros((score0.shape[0], score0.shape[1]), dtype=torch.float32)
            rank = 0.6
            for i in range(3):
                # wC[i]: T(class_num, Ci_size)
                # C_sc: T(bat, class_num)
                score2 += self.relu(scC[i].mm(wC[i].t())*(rank**i))
            #----------add mean cilin-class score to those who have no cilin-class
            mean_cilin_sc = torch.mean(score2, 1)
            score2 = score2*(1-msk_c) + mean_cilin_sc.unsqueeze(1).mm(msk_c.unsqueeze(0))
            #----------
            score = score + score2/2
        if 'P' in mode:
            ## POS prediction
            # score_POS: T(bat, 13) pos_num=12+1
            score_POS = self.fc2(h_1)
            # s: (class_num, 13) multi-hot
            # weight_sc: T(bat, class_num) = [bat, 13] .mm [class_num, 13].t()
            weight_sc = self.relu(score_POS.mm(wP.t()))
            #print(torch.max(weight_sc), torch.min(weight_sc))
            score = score + weight_sc
        if 's' in mode:
            ## sememe prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc1(h)
            # sem_score: T(bat, sememe_num)
            sem_score, _ = torch.max(pos_score, dim=1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score1 = self.relu(sem_score.mm(ws.t()))
            #----------add mean sememe score to those who have no sememes
            # mean_sem_sc: T(bat)
            mean_sem_sc = torch.mean(score1, 1)
            # msk: T(class_num)
            score1 = score1 + mean_sem_sc.unsqueeze(1).mm(msk_s.unsqueeze(0))
            #----------
            score = score + score1
        if 'c' in mode:
            ## character prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc3(h)
            # chara_score: T(bat, chara_num)
            chara_score, _ = torch.max(pos_score, dim=1)
            #chara_score = torch.sum(pos_score * alpha, 1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score3 = self.relu(chara_score.mm(wc.t()))
            score = score + score3
        '''
        if RD_mode ==  'CC':
            # fine-tune depended on the target word shouldn't exist in the definition.
            #score_res = score.clone().detach()
            mask1 = torch.lt(x, self.class_num).to(torch.int64)
            mask2 = torch.ones((score.shape[0], score.shape[1]), dtype=torch.float32)
            for i in range(x.shape[0]):
                mask2[i][x[i]*mask1[i]] = 0.
            score = score * mask2 + (-1e6)*(1-mask2)
        '''
        # _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, w.long())
            return loss, score #, indices
        elif operation == 'test':
            return score #, indices

