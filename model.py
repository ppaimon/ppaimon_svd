import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()

        U, S, V = torch.svd(torch.rand(hidden_size,input_size), some=False)

        self.S = nn.Parameter( torch.ones(len(S)) , requires_grad=True )
        self.register_buffer('U', U)
        self.register_buffer('V', V)

        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        x = self.V.t() @ x
        x = F.pad( 
            torch.diag(self.S) , 
            (
                0, self.V.t().shape[1]-self.S.shape[0], 
                0, self.U.shape[0]-self.S.shape[0]
            ), 'constant', 0
        ) @ x
        
        x = self.U @ x

        x = F.sigmoid(x)

        x = self.output(x)
        return x
