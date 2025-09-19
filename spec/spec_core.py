import numpy as np
import torch

from log import make_logger


logger = make_logger('logs','spec-logs')


class SPEC:
    def __init__(self, similarity_function=None, sigma=None):
        if similarity_function is None and sigma is None:
            raise ValueError("Both similarity_function and sigma can not be None")
        self.similarity = similarity_function
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_by_eigendecomposition(self, x, y, kernel_function, eta, args=None):
        kxx = kernel_function(x, x)
        kyy = kernel_function(y, y)

        diff_kernel = kxx - eta * kyy
        diff_kernel /= diff_kernel.shape[0]

        diff_kernel = diff_kernel.type(torch.float)
        eigenvalues, eigenvectors = torch.linalg.eigh(diff_kernel)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        return eigenvalues, eigenvectors

    def compute_by_covariance_matrix(self, x, y, cov_function=None, phi_x=None, phi_y=None, eta=1, method='cholesky', eps=1e-7, args=None):
        if phi_x is None and phi_y is None and cov_function is not None:
            phi_x = cov_function(x)
            phi_y = cov_function(y)

        eigenvalues, eigenvectors = None, None
        C_11 = phi_x.T @ phi_x / phi_x.shape[0]
        C_12 = phi_x.T @ phi_y / np.sqrt(phi_x.shape[0] * phi_y.shape[0])
        C_21 = C_12.T
        C_22 = phi_y.T @ phi_y / phi_y.shape[0]
        if method == 'direct':
            matrix_first_row = torch.hstack([C_11, C_12])
            torch.cuda.empty_cache()
            matrix_second_row = torch.hstack([-eta * C_21, -eta * C_22])
            matrix = torch.vstack([matrix_first_row, matrix_second_row])
            eigenvalues, eigenvectors = torch.linalg.eig(matrix)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
        
        elif method == 'cholesky':
            # Form the symmetric PSD matrix C
            C = torch.vstack([torch.hstack([C_11, np.sqrt(eta) * C_12]), torch.hstack([np.sqrt(eta) * C_21, C_22])])

            # Construct D matrix
            D = torch.diag(torch.cat([
                torch.ones(phi_x.shape[1], device=phi_x.device),
                -torch.ones(phi_y.shape[1], device=phi_x.device)
            ]))

            # Cholesky decomposition
            Z = torch.linalg.cholesky(C + eps * torch.eye(C.shape[0], device=C.device), upper=True)
            del C
            torch.cuda.empty_cache()

            Theta = Z @ D @ Z.T
            eigenvalues, eigvecs = torch.linalg.eigh(Theta)  # Eigendecomposition of symmetric Theta

            # Compute eigenvectors of the original matrix
            eigenvectors = D @ Z.T @ eigvecs
            eigenvectors = eigenvectors / torch.linalg.norm(eigenvectors, dim=0)

        return eigenvalues, eigenvectors
