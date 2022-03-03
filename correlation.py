import torch



def pairwise_dist_torch(A):
    sigma = torch.Tensor([1e-7]).to(A.device)
    r = torch.sum(A*A, axis = 1)
    r = r.view(-1, 1)
    # print(r)
    D = torch.maximum(r - 2*torch.matmul(A, A.t()) + r.t(), sigma)
    D = torch.sqrt(D)
    return D


def dist_corr_torch(X, Y):
    n = float(X.size()[0])
    
    a = pairwise_dist_torch(X)
    b = pairwise_dist_torch(Y)
    # print(a, b)
    A = a - torch.mean(a, axis=1) - torch.unsqueeze(torch.mean(a, axis=0), axis=1) + torch.mean(a)
    B = b - torch.mean(b, axis=1) - torch.unsqueeze(torch.mean(b, axis=0), axis=1) + torch.mean(b)
    # print(A,B)
    dCovXY = torch.sqrt(torch.sum(A*B) / (n ** 2))
    dVarXX = torch.sqrt(torch.sum(A*A) / (n ** 2))
    dVarYY = torch.sqrt(torch.sum(B*B) / (n ** 2))
    dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
    return dCorXY

def dist_corr(img,activation):
    dcor = dist_corr_torch(img,activation)
    return dcor

def test():
    x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.Tensor([[1.0, 3.0], [4.0, 2.0]])
    print(dist_corr_torch(x, y))

# test()