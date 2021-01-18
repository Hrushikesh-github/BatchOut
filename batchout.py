# A module that implements BatchOut layer
import torch
import torch.nn as nn
import numpy as np

class BatchOut(nn.Module):
    def __init__(self, n, k):
        super(BatchOut, self).__init__()
        # Define n and k, the factor we multiply the perturbation and fraction of samples to sample respectively
        self.n = n
        self.k = k

    # The forward method that takes activations 'x' (at some layer) of a mini-batch 
    def forward(self, x):
        '''
        We don't want any gradients to be computed for calculations in Batchout, so we use detach method
        The "PyTorch Gradients notebook" is created to understand this
        '''
        y = x.detach()

        # Sample a random int from [1, m] where m is batch size 
        _r = np.random.randint(0, y.shape[0], y.shape[0]) 
        # Obtain the feature of on the samples 
        _sample = y[_r] 
        # Compute the direction of feature perturbation
        _d = (_sample - y)
        # Augment and replace the 1st 'k' samples 
        _number = int(self.k * y.shape[0])
        y[1: _number] = y[1: _number] + (self.n * _d[1: _number])
        
        # Change the data value
        x.data = y 
        
        return x

# Run a test case to ensure class is working correctly
if __name__ == '__main__':
    
    v = BatchOut(0.3, 12)
    print(v)

    A = torch.rand([20, 100], requires_grad=True) # 20 is the batch size

    augmented = v(A)

    print(augmented.shape)
    
    print(augmented.grad)
    print(A.grad)

    print(augmented.requires_grad)

    print((A[12:] - augmented[12:] == torch.zeros(8, 100)).all())
