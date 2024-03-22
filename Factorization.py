import torch
import matplotlib.pyplot as plt
from typing import Tuple, Union


class Quantizer():
    def __init__(self, W: torch.Tensor, b: int = 8, z: float = 0, method: str = 'mse'):
        self.z = z
        self.base = b
        if method == 'mse':
            self.q_max = 9.89675982 * torch.mean(torch.abs(W - W.mean()))
            self.scale = 2*self.q_max/(2**self.base-1)
        elif method == 'min-max':
            self.scale = (torch.max(W) - torch.min(W)) / (2**self.base - 1)
        else:
            raise AttributeError(f'No such method: {method}')             

    def transform(self, W: torch.Tensor) -> torch.Tensor:
        return torch.clip(torch.round(W/self.scale) + self.z, -2**(self.base-1), 2**(self.base-1) - 1) 

    def inverse(self, W: torch.Tensor) -> torch.Tensor:
        return self.scale * (W - self.z)

    def projection(self, W: torch.Tensor) -> torch.Tensor:
        return self.inverse(self.transform(W))


def ADMM(B: torch.Tensor, U: torch.Tensor, K: torch.Tensor, G: torch.Tensor, 
         rank: int, quant: Quantizer, eps: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
    
    ro = torch.trace(G) / rank

    LL = G + ro * torch.eye(*G.size())
    L = torch.linalg.cholesky(LL)
    LL_inv = torch.cholesky_inverse(L)

    r = torch.inf
    s = torch.inf

    while (r > eps) or (s > eps):
        B_ = LL_inv @ (K + ro * (B + U)).T
        B0 = torch.clone(B)
        
        B = quant.projection(B_.T - U)
        U = U + B - B_.T

        r = torch.norm(B - B_.T, p='fro')**2 / torch.norm(B, p='fro')**2
        s = torch.norm(B - B0, p='fro')**2 / torch.norm(U, p='fro')**2
    
    return B, U


def e_quant(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.norm(X - A @ B.T, p='fro') / torch.norm(X, p='fro')


def factorize(W: torch.Tensor, rank: int, plot: bool = False, return_int: bool = False, method: str = 'mse', use_ADMM: bool = True) \
    -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Factorizes the given tensor using Singular Value Decomposition (SVD) and the ADMM method.

    Args:
        W (torch.Tensor): The original tensor to be factorized.
        rank (int): The rank for factorization.
        plot (bool, optional): If True, plots the change in quantization error. Defaults to False.
        return_int (bool, optional): If True, returns matrices A and B in int8 format. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        - If return_int = False, returns the compressed(factorized) matrix W.
        - If return_int = True, returns decomposition of W in form of two matrices A and B in int8 dtype.
    """
    quant = Quantizer(W, b=8, z=0, method=method)    
    W = quant.transform(W)

    print(f'rank: {rank}')
    
    U, s, Vt = torch.linalg.svd(W, full_matrices=False)
    S = torch.diag(s[:rank])
    
    A = U[:, :rank] @ S
    B = Vt[:rank, :].T

    if use_ADMM:
        U_A = torch.zeros_like(A)
        U_B = torch.zeros_like(B)
    
        e_new = e_quant(W, A, B)
    
        e_new_ = []
        for _ in range(50):
            K = W @ B
            G = B.T @ B
            A, U_A = ADMM(A, U_A, K, G, rank, quant)
    
            K = W.T @ A
            G = A.T @ A
            B, U_B = ADMM(B, U_B, K, G, rank, quant)
    
            # e_old = e_new
            e_new = e_quant(W,A,B)
            e_new_.append(e_new)

    if plot:
        e_new_np = torch.tensor(e_new_).detach().numpy()
        plt.plot(e_new_np)
        plt.show()

    A_int = (A / quant.scale).to(torch.int8)
    B_int = (B / quant.scale).to(torch.int8)
    
    if return_int:
        return A_int, B_int, quant.scale
    
    return A_int.float() @ B_int.T.float() * quant.scale**3



def SVD_quant(W: torch.Tensor, rank: int, return_ab: bool = False, method: str = 'mse') -> torch.Tensor:
    
    U, s, Vt = torch.linalg.svd(W, full_matrices=False)
    S = torch.diag(s[:rank])
    
    A = U[:, :rank] @ S
    B = Vt[:rank, :].T
    
    A_quant = Quantizer(A, 8, 0, method=method)
    B_quant = Quantizer(B, 8, 0, method=method)
    
    A_int = (A / A_quant.scale).to(torch.int8)
    B_int = (B / B_quant.scale).to(torch.int8)
    
    if return_ab:
        return A_int, B_int, A_quant.scale, B_quant.scale
    
    return A_int.float() @ B_int.T.float() * A_quant.scale * B_quant.scale

    return A @ B.T