import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "dist"):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type

    def set_forward(self, x, is_feature = True, fine_tune_epochs = 300, firth_c=None):
        return self.set_forward_adaptation(x,is_feature, fine_tune_epochs = fine_tune_epochs,
                                           firth_c=firth_c)  # Baseline always do adaptation

    def set_forward_adaptation(self, x, is_feature=True, fine_tune_epochs = 300, firth_c=None):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = y_support.to(device_name)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        if self.loss_type == 'mlp':
            class Net(nn.Module):
                def __init__(self, feat_dim, n_way):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(feat_dim, 5000)
                    self.fc2 = nn.Linear(5000, 500)
                    self.fc3 = nn.Linear(500, n_way)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.fc2(F.relu(x))
                    x = self.fc3(F.relu(x))
                    return x
            linear_clf = Net(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.to(device_name)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device_name)

        batch_size = 4
        support_size = self.n_way* self.n_support
        scores_eval = []
        for epoch in range(fine_tune_epochs + 1):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size) ]).to(device_name)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                if firth_c:
                    # P_tilde
                    logp_tilde = scores
                    logp_hat = logp_tilde - torch.logsumexp(logp_tilde, axis=1, keepdim=True)
                    firth_term = logp_hat.mean()
                    loss_firth = (-firth_c) * firth_term
                    loss = loss + loss_firth
                loss.backward()
                set_optimizer.step()
            if epoch %100 ==0 and epoch !=0:
                scores_eval.append(linear_clf(z_query))
        return scores_eval


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
