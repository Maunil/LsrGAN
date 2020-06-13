import torch
import torch.nn as nn

rdc_text_dim = 1000
z_dim = 100
h_dim = 4096
hidden_layer = 100

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

# reduce to dim of text first
class _netG(nn.Module):
    '''
    Generator 
    '''
    def __init__(self, text_dim=11083, X_dim=3584):
        super(_netG, self).__init__()
        self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output, rdc_text

#class _regressor (nn.Module):
    def __init__(self, X_dim = 3584, text_dim=11083):
        super(_regressor, self).__init__()
        self.text_gen = nn.Sequential(nn.Linear(X_dim, h_dim), nn.LeakyReLU(), 
                            nn.Linear(h_dim, text_dim), 
                            nn.Tanh())
 
    def forward(self, input):
        text_gen = self.text_gen(input)
        return text_gen
        
class _netD(nn.Module):
    '''
    Discriminator
    '''
    def __init__(self, y_dim=200, X_dim=3584):
        super(_netD, self).__init__()
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim),
                                      nn.ReLU())
        self.D_gan = nn.Linear(h_dim, 1) # 
        self.D_aux = nn.Linear(h_dim, y_dim)
 
    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h) 

#class _seen_vs_unseen(nn.Module):
    def __init__(self, X_dim=3584):
        super(_seen_vs_unseen, self).__init__()
        self.decision = nn.Sequential(nn.Linear(X_dim, hidden_layer), nn.LeakyReLU(), 
                            nn.Linear(hidden_layer, 2))

    def forward(self, input):
        return self.decision(input)
        
# In GBU setting, using attribute
class _netG_att(nn.Module):
    def __init__(self, opt, att_dim, X_dim):
        super(_netG_att, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.z_dim + att_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())
    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        output = self.main(input)
        return output


