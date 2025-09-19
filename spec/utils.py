import inspect
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, pairwise_distances, \
    adjusted_mutual_info_score
from torchvision.utils import save_image
from tqdm import tqdm


def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    # batched_rff_cos = torch.cos(product) # [B, feature_dim]  # Commented for more efficiency
    # batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([torch.cos(product), torch.sin(product)], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]
    del product  # There is no need for product after this line

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]
    torch.cuda.empty_cache()
    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
        torch.cuda.empty_cache()
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2
    return cov, batched_rff.squeeze()



def cov_rff(x, feature_dim, std, batchsize=16, normalise=True):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


def load_and_concatenate_all(image_folders, captions_files, image_feats_paths, text_feats_paths, img_feats_key='img_feats', txt_feats_key='txt_feats', img_format='jpg', num_imgs_per_path=None):

    image_paths = []
    for i, path in enumerate(image_folders):
        # with open(path) as f:
        #     for line in f:
        #         image_paths.append(f'{image_folders[i]}/{line.strip()}.png')
        # print(path)
        image_paths += natsorted(glob(f'{path}/*.{img_format}'), key=str)[:num_imgs_per_path]
    image_feats = np.concatenate([np.load(path)[img_feats_key][:num_imgs_per_path] for path in image_feats_paths], axis=0)
    if text_feats_paths[0].split('.')[-1] == 'npy':
        text_feats = np.concatenate([np.load(path) for path in text_feats_paths], axis=0)
    else:
        text_feats = np.concatenate([np.load(path)[txt_feats_key] for path in text_feats_paths], axis=0)
    
    # Load and concatenate captions
    captions = []
    for file in captions_files:
        with open(file, 'r') as f:
            captions.extend([caption.strip() for caption in f.readlines()])

    return image_paths, captions, image_feats, text_feats


def gaussian_kernel_decorator(function):
    def wrap_kernel(self, *args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(function)
        params = list(sig.parameters.keys())
        
        # Determine if `compute_kernel` specific parameter is in args or kwargs
        bound_args = sig.bind_partial(*args, **kwargs).arguments
        compute_kernel = bound_args.get('compute_kernel', True)

        if compute_kernel is True:
            args = list(args)  # To be able to edit args
            if 'X' in params:
                index = params.index('X') - 1
                args[index] = self.gaussian_kernel(args[index])

            if 'Y' in params:
                index = params.index('Y') - 1
                if args[index] is not None:
                    args[index] = self.gaussian_kernel(args[index])

        return function(self, *args, **kwargs)

    return wrap_kernel


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * torch.log2(p_)).sum()
    if q == "inf":
        return -torch.log2(torch.max(p))
    return torch.log2((p_ ** q).sum()) / (1 - q)

def gaussian_kernel(x, y=None, sigma=None, batchsize=256, normalize=True, device="cuda"):
    '''
    calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

    x:
        input, dim: [batch, dims]
    y:
        input, dim: [batch, dims], If y is `None` then y = x and it will compute k(x, x).
    sigma:
        bandwidth parameter
    batchsize:
        Batchify the formation of kernel matrix, trade time for memory
        batchsize should be smaller than length of data

    return:
        scalar : mean of kernel values
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        y = x if y is None else torch.from_numpy(y).to(device)
    else:
        x = x.to(device)
        y = x if y is None else y.to(device)

    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    # if sigma is None:
    #     sigma = self.sigma

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx*batchsize:min((batchidx+1)*batchsize, y.shape[0])]
        res = torch.norm(x.unsqueeze(1)-y_slice, dim=2, p=2).pow(2)
        res = torch.exp((- 1 / (2*sigma*sigma)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    if normalize is True:
        total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

    return total_res


def cosine_kernel(x, y=None, batchsize=256, normalize=True, device="cuda"):
    '''
    Calculate the cosine similarity kernel matrix. The shape of x and y should be equal except for the batch dimension.

    x:
        Input tensor, dim: [batch, dims]
    y:
        Input tensor, dim: [batch, dims]. If y is `None`, then y = x and it will compute cosine similarity k(x, x).
    batchsize:
        Batchify the formation of the kernel matrix, trade time for memory.
        batchsize should be smaller than the length of data.

    return:
        Scalar: Mean of cosine similarity values
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        y = x if y is None else torch.from_numpy(y).to(device)
    else:
        x = x.to(device)
        y = x if y is None else y.to(device)

    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx * batchsize:min((batchidx + 1) * batchsize, y.shape[0])]
        
        # Normalize x and y_slice
        x_norm = x / x.norm(dim=1, keepdim=True)
        y_norm = y_slice / y_slice.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        res = torch.mm(x_norm, y_norm.T)

        total_res = torch.hstack([total_res, res])

        del res, y_slice

    if normalize is True:
        total_res = total_res / (x.shape[0] * y.shape[0])

    return total_res

def cosine_features(x, device="cuda"):
    """
    Compute cosine normalized features for the input matrix x. (Divide each row by its L2 norm)
    x: Input data, shape (n_samples, n_features)
    return: Cosine normalized features, shape (n_samples, n_features)
    """
    l2_norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized_matrix = x / l2_norms
    return torch.from_numpy(normalized_matrix).to(device)

def gaussian_covariance(x, rff_dim, sigma, batchsize=256, normalize=True, device="cuda", return_features=False):
    """
    Compute the Gaussian covariance matrix and Gaussian features using Random Fourier Features (RFF).
    x: Input data, shape (n_samples, n_features)
    rff_dim: Dimensionality of the RFF features.
    sigma: Bandwidth parameter for the Gaussian kernel.
    batchsize: Batch size for processing.
    normalize: Whether to normalize the covariance matrix.
    return_features: If True, returns covariance matrix, Random Fourier Features and Estimated Gaussian features of x.
    return: Covariance matrix, shape (n_samples, n_samples), Random Fourier Features, shape (n_features, rff_dim), Estimated Gaussian features of x, shape (n_samples, rff_dim)
    """
    x_cov, omegas, x_feature = cov_rff(x, rff_dim, sigma, batchsize, normalize)
    if return_features is True:
        return x_cov, omegas, x_feature
    else:
        return x_cov


def violin_visualization(cluster_indices, x, save_path, n_clusters, points_per_cluster):
    combined_indexes = sum(cluster_indices.values(), [])
    distance_matrix = pairwise_distances(x[combined_indexes], metric='euclidean')

    n = x.shape[0]  # Number of vectors

    distances = squareform(pdist(x, 'euclidean'))

    upper_tri = np.triu(distances, k=1)
    avg_dist_original = np.sum(upper_tri) / (n * (n - 1) / 2)  # Average distance

    # Compute scaling factor
    c = 1.0 / avg_dist_original

    # Scale all vectors
    x_scaled = x * c

    normalized_dist = pairwise_distances(x_scaled[combined_indexes], metric='euclidean')

    # Step 2: Compute distances from each point to its cluster center
    intra_cluster_distances = []
    intra_cluster_means = []
    intra_cluster_variances = []

    compare_to_center = False
    cluster_ranges = [(i*points_per_cluster, (i+1)*points_per_cluster) for i in range(n_clusters)]
    if compare_to_center is True:
        centers = {}
        for cluster_id, indices in cluster_indices.items():
            centers[cluster_id] = np.mean(x_scaled[indices], axis=0)  # Centroid = mean of points

        for cluster_id, indices in cluster_indices.items():
            points = x_scaled[indices]
            center = centers[cluster_id]
            distances = cdist(points, [center], metric='euclidean').flatten()  # Shape: (n_points, 1) -> (n_points,)
            
            intra_cluster_distances.append(distances)
            intra_cluster_means.append(np.mean(distances))
            intra_cluster_variances.append(np.var(distances))

    else:
        for start, end in cluster_ranges:
            # Extract the submatrix for the current cluster
            submatrix = normalized_dist[start:end, start:end]
            
            # Exclude diagonal (distance=0) and use upper triangular to avoid duplicates
            upper_tri_indices = np.triu_indices_from(submatrix, k=1)
            intra_distances = submatrix[upper_tri_indices]

            # Compute mean and variance
            intra_cluster_means.append(np.mean(intra_distances))
            intra_cluster_variances.append(np.var(intra_distances))
            intra_cluster_distances.append(intra_distances)
        print(f'cluster means: {intra_cluster_means}')
        print(f'cluster variances: {intra_cluster_variances}')
        print(f'cluster std: {np.sqrt(intra_cluster_variances)}')

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=intra_cluster_distances, palette="tab10", linewidth=1.)

    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Extract distances between cluster i and cluster j
            submatrix = normalized_dist[
                cluster_ranges[i][0]:cluster_ranges[i][1],
                cluster_ranges[j][0]:cluster_ranges[j][1]
            ]
            inter_cluster_distances.extend(submatrix.flatten())

    inter_mean = np.mean(inter_cluster_distances)
    print(f'inter mean: {inter_mean}, varaince: {np.var(inter_cluster_distances)}, covariance: {np.std(inter_cluster_distances)}')
    plt.axhline(y=inter_mean, color='red', linestyle='--', label=f'Inter-cluster Mean: {inter_mean:.2f}', alpha=0.7)
    plt.legend(fontsize=23)

    plt.xticks(ticks=range(n_clusters), labels=[f'{i}' for i in range(n_clusters)], fontsize=15)
    # plt.title('Violin Plot of Intra-cluster Distances by Cluster for DINOv2', fontsize=24)
    plt.ylabel('Normalized Intra-cluster Distances', fontsize=20)
    plt.xlabel('Clusters', fontsize=24)
    plt.ylim(0.2, 1.8)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}.png' , bbox_inches='tight', dpi=400)



def compare_violin_visualization(cluster_indices, x, y, save_path, n_clusters, points_per_cluster):
    combined_indexes = sum(cluster_indices.values(), [])
    from sklearn.metrics import pairwise_distances
    distance_matrix = pairwise_distances(x[combined_indexes], metric='euclidean')
    
    n = x.shape[0]  # Number of vectors

    distances = squareform(pdist(x, 'euclidean'))

    upper_tri = np.triu(distances, k=1)
    avg_dist_original = np.sum(upper_tri) / (n * (n - 1) / 2)  # Average distance
    
    # Compute scaling factor
    c = 1.0 / avg_dist_original
    
    # Scale all vectors
    x_scaled = x * c
    
    normalized_dist = pairwise_distances(x_scaled[combined_indexes], metric='euclidean')

    centers = {}
    for cluster_id, indices in cluster_indices.items():
        centers[cluster_id] = np.mean(x_scaled[indices], axis=0)  # Centroid = mean of points

    # Step 2: Compute distances from each point to its cluster center
    intra_cluster_distances = []
    intra_cluster_means = []
    intra_cluster_variances = []

    for cluster_id, indices in cluster_indices.items():
        points = x[indices]
        center = centers[cluster_id]
        distances = cdist(points, [center], metric='euclidean').flatten()  # Shape: (n_points, 1) -> (n_points,)
        
        intra_cluster_distances.append(distances)
        intra_cluster_means.append(np.mean(distances))
        intra_cluster_variances.append(np.var(distances))

    cluster_ranges = [(i*points_per_cluster, (i+1)*points_per_cluster) for i in range(n_clusters)]
    # # [(0, 16), (16, 32), ..., (144, 160)]

    intra_cluster_means = []
    intra_cluster_variances = []
    intra_cluster_distances = []

    for start, end in cluster_ranges:
        # Extract the submatrix for the current cluster
        submatrix = normalized_dist[start:end, start:end]
        
        # Exclude diagonal (distance=0) and use upper triangular to avoid duplicates
        upper_tri_indices = np.triu_indices_from(submatrix, k=1)
        intra_distances = submatrix[upper_tri_indices]

        # Compute mean and variance
        intra_cluster_means.append(np.mean(intra_distances))
        intra_cluster_variances.append(np.var(intra_distances))
        intra_cluster_distances.append(intra_distances)
 
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=intra_cluster_distances, palette="tab10", linewidth=1.)

    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Extract distances between cluster i and cluster j
            submatrix = normalized_dist[
                cluster_ranges[i][0]:cluster_ranges[i][1],
                cluster_ranges[j][0]:cluster_ranges[j][1]
            ]
            inter_cluster_distances.extend(submatrix.flatten())

    inter_mean = np.mean(inter_cluster_distances)
    plt.axhline(y=inter_mean, color='red', linestyle='--', label=f'Inter-cluster Mean: {inter_mean:.2f}', alpha=0.7)
    plt.legend(fontsize=23)

    plt.xticks(ticks=range(n_clusters), labels=[f'{i}' for i in range(n_clusters)], fontsize=15)
    # plt.title('Violin Plot of Intra-cluster Distances by Cluster for DINOv2', fontsize=24)
    plt.ylabel('Normalized Intra-cluster Distances', fontsize=20)
    plt.xlabel('Clusters', fontsize=24)
    plt.ylim(0.2, 2)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}.png' , bbox_inches='tight', dpi=400)



def plot_combined_violin_clusters(x, y, cluster_indices_x, cluster_indices_y, save_path, n_clusters, model_names, colors=None):
    """
    Plot violin plots for x and y samples with clusters arranged side by side.
    
    Parameters:
    - x, y: Input data matrices (n_samples x n_features)
    - cluster_indices_x, cluster_indices_y: Dictionaries mapping cluster IDs to sample indices
    - save_path: Path to save the plot
    - n_clusters: Number of clusters
    """
    name_x, name_y = model_names
    if colors is None:
        color_x, color_y = '#4a5f25', '#5271ff'
    elif colors == 'reverse':
        color_x, color_y = '#5271ff', '#4a5f25'
    else:
        color_x, color_y = colors

    # Process x data
    x_scaled, x_normalized_dist, x_cluster_ranges = preprocess_data(x, cluster_indices_x)
    x_intra_distances = calculate_intra_cluster_distances(x_normalized_dist, x_cluster_ranges, n_clusters)
    
    # Process y data
    y_scaled, y_normalized_dist, y_cluster_ranges = preprocess_data(y, cluster_indices_y)
    y_intra_distances = calculate_intra_cluster_distances(y_normalized_dist, y_cluster_ranges, n_clusters)
    
    # Combine data for plotting
    combined_distances = []
    cluster_labels = []
    dataset_labels = []
    
    for i in range(n_clusters):
        combined_distances.extend(x_intra_distances[i])
        cluster_labels.extend([f'Cluster {i}'] * len(x_intra_distances[i]))
        dataset_labels.extend([name_x] * len(x_intra_distances[i]))
        
        combined_distances.extend(y_intra_distances[i])
        cluster_labels.extend([f'Cluster {i}'] * len(y_intra_distances[i]))
        dataset_labels.extend([name_y] * len(y_intra_distances[i]))
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Distance': combined_distances,
        'Cluster': cluster_labels,
        'Dataset': dataset_labels
    })
    
    # Plot
    plt.figure(figsize=(24, 6))
    sns.violinplot(
        x='Cluster', 
        y='Distance', 
        hue='Dataset', 
        data=plot_data, 
        palette={
        name_x: color_x,  # First tab10 color for CLIP
        name_y: color_y  # Second tab10 color for DINOv2
    },
        linewidth=0.5,
    )
    
    # Calculate and plot inter-cluster means
    x_inter_mean = calculate_inter_cluster_mean(x_normalized_dist, x_cluster_ranges, n_clusters)
    y_inter_mean = calculate_inter_cluster_mean(y_normalized_dist, y_cluster_ranges, n_clusters)
    
    plt.axhline(y=x_inter_mean, color=color_x, linestyle='--', label=f'{name_x} Inter-cluster Mean: {x_inter_mean:.2f}', alpha=0.7)
    plt.axhline(y=y_inter_mean, color=color_y, linestyle='--', label=f'{name_y} Inter-cluster Mean: {y_inter_mean:.2f}', alpha=0.7)
    
    plt.legend(fontsize=20)
    plt.ylabel('Normalized Intra-cluster Distances', fontsize=18)
    plt.xlabel('Clusters', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylim(0.2, 2)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}{".png" if "png" not in save_path else ""}' , bbox_inches='tight', dpi=400)
    plt.close()

def preprocess_data(data, cluster_indices):
    """Normalize data and compute distance matrix"""
    combined_indexes = sum(cluster_indices.values(), [])
    n = data.shape[0]
    
    # Compute scaling factor
    distances = squareform(pdist(data, 'euclidean'))
    upper_tri = np.triu(distances, k=1)
    avg_dist_original = np.sum(upper_tri) / (n * (n - 1) / 2)
    c = 1.0 / avg_dist_original
    
    # Scale data and compute normalized distances
    data_scaled = data * c
    normalized_dist = pairwise_distances(data_scaled[combined_indexes], metric='euclidean')
    
    # Create cluster ranges
    cluster_ranges = []
    start = 0
    for i in range(len(cluster_indices)):
        end = start + len(cluster_indices[i])
        cluster_ranges.append((start, end))
        start = end
    
    return data_scaled, normalized_dist, cluster_ranges

def calculate_intra_cluster_distances(normalized_dist, cluster_ranges, n_clusters):
    """Calculate intra-cluster distances for each cluster"""
    intra_cluster_distances = []
    
    for start, end in cluster_ranges:
        submatrix = normalized_dist[start:end, start:end]
        upper_tri_indices = np.triu_indices_from(submatrix, k=1)
        intra_distances = submatrix[upper_tri_indices]
        intra_cluster_distances.append(intra_distances)
    
    return intra_cluster_distances

def calculate_inter_cluster_mean(normalized_dist, cluster_ranges, n_clusters):
    """Calculate mean inter-cluster distance"""
    inter_cluster_distances = []
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            submatrix = normalized_dist[
                cluster_ranges[i][0]:cluster_ranges[i][1],
                cluster_ranges[j][0]:cluster_ranges[j][1]
            ]
            inter_cluster_distances.extend(submatrix.flatten())
    
    return np.mean(inter_cluster_distances)



def KMeans_validation(cluster_indices, x, y):
    combined_indexes = sum(cluster_indices.values(), [])

    labels = sum([[i] * len(cluster_indices[i]) for i in cluster_indices.keys()], [])
    n_repeats = 50
    n_clusters = len(np.unique(labels))  # Number of clusters based on true labels
    nmi_scores_x = []
    nmi_scores_y = []
    ami_scores_x = []
    ami_scores_y = []
    kmeans_sil_scores_x = []
    kmeans_sil_scores_y = []

    for _ in range(n_repeats):
        # Generate a random state for this iteration
        random_state = np.random.randint(0, 10000)  # Random seed for variability

        # Apply KMeans on x
        kmeans_x = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_labels_x = kmeans_x.fit_predict(x[combined_indexes])

        # Calculate the Normalized Mutual Information score for x
        # print(kmeans_labels_x.shape)
        nmi_x = normalized_mutual_info_score(labels, kmeans_labels_x)
        ami_x = adjusted_mutual_info_score(labels, kmeans_labels_x)
        nmi_scores_x.append(nmi_x)
        ami_scores_x.append(ami_x)

        kmeans_sil_scores_x.append(silhouette_score(x[combined_indexes], kmeans_labels_x))

        # Apply KMeans on y
        kmeans_y = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_labels_y = kmeans_y.fit_predict(y[combined_indexes])

        # Calculate the Normalized Mutual Information score for y
        nmi_y = normalized_mutual_info_score(labels, kmeans_labels_y)
        ami_y = adjusted_mutual_info_score(labels, kmeans_labels_y)
        nmi_scores_y.append(nmi_y)
        ami_scores_y.append(ami_y)
        kmeans_sil_scores_y.append(silhouette_score(y[combined_indexes], kmeans_labels_y))

    # Calculate average and variance for x
    average_nmi_x = np.mean(nmi_scores_x)
    variance_nmi_x = np.var(nmi_scores_x)
    
    average_ami_x = np.mean(ami_scores_x)
    variance_ami_x = np.var(ami_scores_x)
    
    average_kmeans_sil_x = np.mean(kmeans_sil_scores_x)
    variance_kmeans_sil_x = np.var(kmeans_sil_scores_x)
    

    # Calculate average and variance for y
    average_nmi_y = np.mean(nmi_scores_y)
    variance_nmi_y = np.var(nmi_scores_y)

    average_ami_y = np.mean(ami_scores_y)
    variance_ami_y = np.var(ami_scores_y)
    
    average_kmeans_sil_y = np.mean(kmeans_sil_scores_y)
    variance_kmeans_sil_y = np.var(kmeans_sil_scores_y)

    # Report results
    print(f"Average NMI for x: {average_nmi_x:.4f}, Variance: {variance_nmi_x:.4f}")
    print(f"Average NMI for y: {average_nmi_y:.4f}, Variance: {variance_nmi_y:.4f}")

    # Report results
    print(f"Average AMI for x: {average_ami_x:.4f}, Variance: {variance_ami_x:.4f}")
    print(f"Average AMI for y: {average_ami_y:.4f}, Variance: {variance_ami_y:.4f}")

    print(f"Average KMeans SIL for x: {average_kmeans_sil_x:.4f}, Variance: {variance_kmeans_sil_x:.4f}")
    print(f"Average KMeans SIL for y: {average_kmeans_sil_y:.4f}, Variance: {variance_kmeans_sil_y:.4f}")
    
    print(f"SPEC SIL for x: {silhouette_score(x[combined_indexes], labels):.4f}")
    print(f"SPEC SIL for y: {silhouette_score(y[combined_indexes], labels):.4f}")



def tsne_visulization(cluster_indices, x, y, save_dir):
    # if x is None or y is None:
    # raise ValueError("x or y can not be None when plotting T-SNE")
    for data_features, data_name in ((x, 'x'), (y, 'y')):
        data_points = []
        labels = []
        for label, idx_list in cluster_indices.items():
            for idx in idx_list:
                data_points.append(data_features[idx])
                labels.append(label)

        data_points = np.array(data_points)
        labels = np.array(labels)
        # Step 2: Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data_points)

        # Step 3: Plot the results
        plt.figure(figsize=(5, 4))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=30)
        cbar1 = plt.colorbar(scatter, fraction=0.05, pad=0.02)

        # Add legend
        handles, _ = scatter.legend_elements()
        # plt.legend(handles, np.unique(labels), title="Classes")

        # plt.title("t-SNE Visualization")
        # plt.xlabel("t-SNE Component 1")
        # plt.ylabel("t-SNE Component 2")
        plt.savefig(os.path.join(save_dir, f'tsne_summary_{data_name}.png'), bbox_inches='tight', dpi=400)


def umap_visualization(cluster_indices, x, y, save_dir):
    """UMAP visualization with identical interface to pacmap_visualization().
    
    Args:
        cluster_indices: Dictionary {label: [indices]} 
        x: First set of features (e.g., original data)
        y: Second set of features (e.g., reconstructed data)
        save_dir: Directory to save plots
    """
    print('Using UMAP')
    
    distances = {}
    all_distances = {}
    for data_features, data_name in ((x, 'x'), (y, 'y')):
        data_points = []
        labels = []
        for label, idx_list in cluster_indices.items():
            for idx in idx_list:
                data_points.append(data_features[idx])
                labels.append(label)

        data_points = np.array(data_points)
        labels = np.array(labels)
        
        # Apply UMAP with similar intent to PaCMAP params
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,       # Balances local/global like FP_ratio=5.0
            min_dist=0.1,         # Creates some space between clusters
            spread=5,
            metric='euclidean',
            random_state=42
        )
        umap_results = reducer.fit_transform(data_points)

        # Plot with identical styling
        plt.figure(figsize=(5, 4))
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], 
                             c=labels, cmap='tab10', s=30)
        plt.colorbar(scatter, fraction=0.05, pad=0.02)
        
        # Uncomment if you want the legend
        # handles, _ = scatter.legend_elements()
        # plt.legend(handles, np.unique(labels), title="Classes")
        
        plt.savefig(
            os.path.join(save_dir, f'umap_summary_{data_name}.png'), 
            bbox_inches='tight', 
            dpi=400
        )
        plt.close()


def visualize_modes_covariance(eigenvalues, eigenvectors, x_feature, y_feature, num_visual_mode, save_dir, dataset,
                               absolute=False, data_type='image', num_samples_per_mode=50, plot_tsne=True,
                               x=None, y=None, save_file=True, model_names=('X', 'Y'), plot_violin=False):
    """
    Visualizes the top `num_visual_mode` modes.

    Args:
        eigenvalues: The eigenvalues of the difference matrix `DiffEmbed_by_covariance_matrix` function
        eigenvectors: The eigenvalues of the difference matrix `DiffEmbed_by_covariance_matrix` function
        x_feature: The features of the images. In cosine similarity it is output of the `cosine_features`, while  in
        gaussian covariance it is output of the `gaussian_covariance` function (RFF features)
        y_feature: The features of the images. In cosine similarity it is output of the `cosine_features`, while  in
        gaussian covariance it is output of the `gaussian_covariance` function (RFF features)
        num_visual_mode: Number of top modes to visualize
        save_dir: saving directory
        dataset: dataset of images
        absolute: consider abs of eigenvector to compute, default is False.
        data_type: Type of the data, can be 'image' or 'text'.
        num_samples_per_mode: Number of samples to visualize per mode
        plot_tsne: Whether to plot T-SNE visualizations. If True it plots T-SNE and UMAP visualizations for both x and y features.
        x: The original x features, required if `plot_tsne` is True.
        y: The original y features, required if `plot_tsne` is True.
        save_file: Whether to save the visualizations and summaries to files.
        model_names: Names of the models for x and y features, used in violin plot titles.
        plot_violin: Whether to plot violin visualizations. If True it plots violin plots for both x and y features.

    Returns: None

    """
    os.makedirs(save_dir, exist_ok=True)
    plt.scatter(eigenvalues.cpu().numpy(), [0] * eigenvalues.shape[0], s=5, c='blue')
    plt.savefig(f'{save_dir}/diff-eigvals.png')
    np.save(f'{save_dir}/eigenvalues.npy', eigenvalues.cpu().numpy())
    np.save(f'{save_dir}/eigenvectors.npy', eigenvectors.cpu().numpy())
    m, max_id = eigenvalues.topk(num_visual_mode)

    summary_indexes = {}
    for i in range(num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]
        if top_eigenvector.sum() < 0:
            top_eigenvector = -top_eigenvector
        
        if absolute:
            top_eigenvector = top_eigenvector.abs() 

        eig_x, eig_y = top_eigenvector[:x_feature.shape[1]], top_eigenvector[x_feature.shape[1]:]

        scores = x_feature @ eig_x + y_feature @ eig_y
        if scores.sum() < 0:
            scores = -scores
        if data_type == 'image':
            top_image_ids = scores.sort(descending=True)[1]
            save_folder_name = os.path.join(save_dir, str(i))

            if save_file is True:
                os.makedirs(save_folder_name, exist_ok=True)

            summary = []
            summary_indexes[i] = []
            for j, top_image_id in enumerate(top_image_ids[:num_samples_per_mode]):
                idx = top_image_id
                if save_file is True and j < 16:
                    top_img = dataset[idx][0]
                    save_image(top_img, os.path.join(save_folder_name, f'{j}_ref_{str(dataset[idx][1])[:5]}.jpg'), nrow=1)
                    summary.append(top_img)
                summary_indexes[i].append(int(idx))
            if save_file is True:
                save_image(summary, os.path.join(save_dir, f'mode={i}_summary.jpg'), nrow=4)
                save_image(summary[:9], os.path.join(save_dir, f'mode={i}_summary_3.jpg'), nrow=3)
                save_image(summary[:4], os.path.join(save_dir, f'mode={i}_summary_2.jpg'), nrow=2)

        elif data_type == 'text':
            assert type(dataset) == list
            top_text_ids = scores.sort(descending=True)[1]

            summary = []
            summary_indexes[i] = []
            for idx in top_text_ids[:num_samples_per_mode]:
                if save_file is True:
                    summary.append(dataset[idx])
                summary_indexes[i].append(int(idx))

            if save_file is True:
                with open(os.path.join(save_dir, f'mode={i}_summary.txt'), "w") as output:
                    output.write(str('---\n---'.join(summary)))
        else:
            raise ValueError(f"`data_type` {data_type} is not supported. Choose from ['image', 'text'].")


    if plot_tsne is True:
        if x is None or y is None:
            raise ValueError("x or y can not be None when plotting T-SNE")
        tsne_visulization(summary_indexes, x=x, y=y, save_dir=save_dir)
        umap_visualization(summary_indexes, x=x, y=y, save_dir=save_dir)

    if plot_violin is True:
        violin_visualization(summary_indexes, x, os.path.join(save_dir, 'violin_x.png'), n_clusters=num_visual_mode, points_per_cluster=num_samples_per_mode)
        violin_visualization(summary_indexes, y, os.path.join(save_dir, 'violin_y.png'), n_clusters=num_visual_mode, points_per_cluster=num_samples_per_mode)
        plot_combined_violin_clusters(x, y, summary_indexes, summary_indexes, os.path.join(save_dir, 'violin_xy.png'), n_clusters=num_visual_mode, model_names=model_names, colors='reverse')
        KMeans_validation(summary_indexes, x, y)
    return summary_indexes
