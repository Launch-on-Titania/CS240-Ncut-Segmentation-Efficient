import numpy as np
import torch

class FastNcut:
    def __init__(
        self,
        A=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0]]),
        const=np.array([[0, 1]]),
        max_iter=1000,
        opt_tol=1e-12,
    ):
        self.A = A
        self.const = const
        self.max_iter = max_iter
        self.opt_tol = opt_tol
        self.B, self.c = self._init_const(self.const)
        self.BB_inv = np.linalg.pinv(
            self.B @ self.B.T
        )  # Precompute and store for reuse

    def _init_const(self, const):
        const_num = len(const)
        B = np.zeros([const_num, len(self.A)])
        for i, pair in enumerate(const):
            B[i][pair] = [1, -1]
        c = np.zeros([const_num, 1])
        return B, c

    def _init_P(self):
        return np.eye(len(self.A)) - self.B.T @ self.BB_inv @ self.B

    def _init_n(self):
        return self.B.T @ self.BB_inv @ self.c

    def _init_ganma(self, n):
        return np.sqrt(1 - np.linalg.norm(n) ** 2)

    def _init_v(self, ganma, P, n):
        PA = P @ self.A
        return ganma * PA @ n / np.linalg.norm(PA @ n) + n

    def _projected_powermethod(self):
        A = torch.from_numpy(self.A).to(torch.float32).cuda()
        B = torch.from_numpy(self.B).to(torch.float32).cuda()
        c = torch.from_numpy(self.c).to(torch.float32).cuda()

        # 计算P和PA
        P = (
            torch.eye(len(A), dtype=torch.float32).cuda()
            - B.T @ torch.linalg.pinv(B @ B.T) @ B
        )
        PA = P @ A  # Precompute PA

        k = 0
        n_0 = B.T @ torch.linalg.pinv(B @ B.T) @ c
        ganma = torch.sqrt(1 - torch.norm(n_0) ** 2)

        if torch.count_nonzero(B) == 0:
            v = ganma * PA @ n_0 / torch.norm(PA @ n_0) + n_0
        else:
            v = torch.rand(len(A), 1, dtype=torch.float32).cuda()

        obj = v.T @ A @ v
        obj_old = obj

        while k < self.max_iter:
            v /= torch.norm(v)
            u = ganma * PA @ v / torch.norm(PA @ v)
            v = u + n_0
            k += 1
            obj = v.T @ A @ v
            if self.opt_tol > abs(obj - obj_old):
                break
            obj_old = obj
        return v.cpu().numpy(), k

    def fit(self, X):
        const_num = 1
        const_eig_vec, iter_num = self._projected_powermethod()
        return const_eig_vec

# class FastNcut:
#     def __init__(self, A =  np.array([[2, 0, 0],[0, 1, 0],[0, 0, 0]]) , const=np.array([[0,1]]), max_iter= 1000, opt_tol=1e-12):
#         self.A = A
#         self.const = const
#         self.max_iter = max_iter
#         self.opt_tol = opt_tol
#         self.B, self.c = self._init_const(self.const)

#     def _init_const(self, const):
#         const_num = len(const)
#         B = np.zeros([const_num,len(self.A)])
#         for i, pair in enumerate(const):
#             B[i][pair] = [1, -1]
#         c = np.zeros([const_num,1])
#         return B, c

#     def _init_P(self):
#         return np.eye(len(self.A))- self.B.T@np.linalg.pinv(self.B@self.B.T)@self.B

#     def _init_n(self):
#         return self.B.T@ np.linalg.pinv(self.B@self.B.T)@self.c

#     def _init_ganma(self,n):
#         return np.sqrt(1- np.linalg.norm(n)**2)


#     def _init_v(self,ganma, P, n):
#         return ganma* P@self.A@n/np.linalg.norm(P@self.A@n)+n


#     def _projected_powermethod(self):
#         """zeros
#         Projected Powermethod optimize a problem
#         max_v v^T A v, ||v|| = 1, Bv=c

#         Simple constrain: you want to same value i-th and j-th element in eigen vector.
#         You can insert 1 and -1 to i-th and j-th element.
#         """

#         P = self._init_P()
#         k = 0
#         n_0 = self._init_n()
#         ganma = self._init_ganma(n_0)
#         # if c is zero vector, escape NaN. 
#         if np.count_nonzero(self.B) == 0:
#             v = self._init_v(ganma, P, n_0)
#         else :
#             v = np.random.rand(len(self.A))[np.newaxis].T
#         obj = v.T@self.A@v
#         obj_old = obj

#         while  k < self.max_iter:
#             v = v/np.linalg.norm(v)
#             u = ganma * P@self.A@v/np.linalg.norm(P*self.A*v)
#             v = u+n_0
#             k+=1
#             obj = v.T@self.A@v
#             if self.opt_tol > abs(obj-obj_old):
#                 break
#             obj_old = obj
#         return v, k

#     def fit(self,X):
#         const_num = 1
#         const_eig_vec, iter_num = self._projected_powermethod()
#         return const_eig_vec