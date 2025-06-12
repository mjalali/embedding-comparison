import numpy as np
from tqdm import tqdm
import torch
from torch.linalg import eigh, eigvalsh, eigvals

from log import make_logger


logger = make_logger('logs','diff-embeddings')


class EmbeddingEvaluation:
    def __init__(self, similarity_function=None, sigma=None):
        if similarity_function is None and sigma is None:
            raise ValueError("Both similarity_function and sigma can not be None")
        self.similarity = similarity_function
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def DiffEmbed_by_eigendecomposition(self, x, y, kernel_function, eta, args=None):
        kxx = kernel_function(x, x)
        kyy = kernel_function(y, y)

        diff_kernel = kxx - eta * kyy
        diff_kernel /= diff_kernel.shape[0]  # TODO do we need that?

        diff_kernel = diff_kernel.type(torch.float)
        print('kernels computed', diff_kernel.shape)
        
        print('Start computing eigen-decomposition')
        eigenvalues, eigenvectors = torch.linalg.eigh(diff_kernel)
        print('Finish computing eigen-decomposition')
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        return eigenvalues, eigenvectors

    def DiffEmbed_by_covariance_matrix(self, x, y, cov_function=None, phi_x=None, phi_y=None, eta=1, args=None):
        # phi_x = cov_function(x) if phi_x is None else phi_x  # TODO check it later
        # phi_y = cov_function(y) if phi_y is None else phi_y
        if phi_x is None and phi_y is None and cov_function is not None:  # TODO change cov_function name
            phi_x = cov_function(x)
            phi_y = cov_function(y)

        matrix_first_row = torch.hstack([phi_x.T @ phi_x / phi_x.shape[0], phi_x.T @ phi_y / np.sqrt(phi_x.shape[0] * phi_y.shape[0])])
        torch.cuda.empty_cache()
        matrix_second_row = torch.hstack([-eta * phi_y.T @ phi_x / np.sqrt(phi_x.shape[0] * phi_y.shape[0]), -eta * phi_y.T @ phi_y / phi_y.shape[0]])
        matrix = torch.vstack([matrix_first_row, matrix_second_row])
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)

        # eps = 1e-7
        # with torch.no_grad():
        #     cholesky_matrix = matrix + eps * torch.eye(x.shape[1]+y.shape[1], device=self.device)
        #     U = torch.linalg.cholesky(cholesky_matrix, upper=True)
        #     diagonal = torch.cat([torch.ones(x.shape[0], device=x.device), -1 * torch.ones(y.shape[0], device=y.device)])
        #     d_matrix = torch.diag(diagonal)
        #     assert len(diagonal) == x.shape[0] + y.shape[0]

        #     matrix_to_be_decomposed = U @ d_matrix @ U.T
        #     eigenvalues, eigenvectors = torch.linalg.eigh(matrix_to_be_decomposed)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        return eigenvalues, eigenvectors
