"""
Advanced color clustering for painting preprocessing that preserves distinct regions
and prevents inappropriate color blending between neighboring areas.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def extract_dominant_colors_adaptive(image, max_colors=12, min_colors=4):
    """
    Adaptively determine the optimal number of colors based on image complexity
    
    Args:
        image: Input RGB image (uint8)
        max_colors: Maximum number of colors to consider
        min_colors: Minimum number of colors to use
    
    Returns:
        int: Optimal number of colors for this image
    """
    # Calculate color variance to estimate complexity
    data = image.reshape((-1, 3)).astype(np.float32)
    
    # Use elbow method to find optimal clusters
    inertias = []
    K_range = range(min_colors, max_colors + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point (where improvement starts diminishing)
    if len(inertias) > 2:
        # Calculate second derivative to find elbow
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Find the point where second derivative is maximum (sharpest bend)
        elbow_idx = np.argmax(second_diffs) + min_colors
        optimal_k = min(elbow_idx + 1, max_colors)
    else:
        optimal_k = max_colors
    
    return optimal_k


def spatial_color_clustering(image, n_colors=8, spatial_weight=0.3, preserve_boundaries=True):
    """
    Advanced K-means clustering that considers spatial proximity to preserve boundaries
    
    Args:
        image: Input RGB image (uint8)
        n_colors: Number of color clusters
        spatial_weight: Weight for spatial coordinates (0.0-1.0)
        preserve_boundaries: Whether to preserve sharp color boundaries
    
    Returns:
        numpy.ndarray: Image with spatially-aware color clustering
    """
    h, w = image.shape[:2]
    
    # Create feature matrix combining color and spatial information
    color_data = image.reshape((-1, 3)).astype(np.float32)
    
    if spatial_weight > 0:
        # Create spatial coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial_data = np.column_stack([
            x_coords.flatten() / w,  # Normalize to [0,1]
            y_coords.flatten() / h
        ]) * spatial_weight * 255  # Scale to match color range
        
        # Combine color and spatial features
        features = np.hstack([color_data, spatial_data])
    else:
        features = color_data
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(features, n_colors, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
    
    # Extract only color centers (ignore spatial components)
    color_centers = centers[:, :3] if spatial_weight > 0 else centers
    color_centers = np.uint8(np.clip(color_centers, 0, 255))
    
    # Apply boundary preservation if enabled
    if preserve_boundaries:
        labels = preserve_color_boundaries(image, labels.flatten(), color_centers)
    
    # Create output image
    clustered_data = color_centers[labels.flatten()]
    clustered_image = clustered_data.reshape(image.shape)
    
    return clustered_image


def preserve_color_boundaries(image, labels, centers):
    """
    Post-process clustering to preserve sharp boundaries between distinct regions
    
    Args:
        image: Original RGB image
        labels: Cluster labels for each pixel
        centers: Cluster centers
    
    Returns:
        numpy.ndarray: Refined labels with preserved boundaries
    """
    h, w = image.shape[:2]
    labels_2d = labels.reshape((h, w))
    
    # Detect edges in original image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create boundary regions
    kernel = np.ones((3, 3), np.uint8)
    boundary_mask = cv2.dilate(edges, kernel, iterations=1) > 0
    
    # For pixels near boundaries, choose cluster based on local neighborhood
    refined_labels = labels_2d.copy()
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if boundary_mask[y, x]:
                # Get 3x3 neighborhood
                neighborhood = image[y-1:y+2, x-1:x+2].reshape(-1, 3)
                current_pixel = image[y, x]
                
                # Find the cluster center closest to current pixel
                distances = np.linalg.norm(centers - current_pixel, axis=1)
                best_cluster = np.argmin(distances)
                
                refined_labels[y, x] = best_cluster
    
    return refined_labels.flatten()


def region_aware_clustering(image, n_colors=8, region_coherence=0.5):
    """
    Clustering that maintains coherence within similar regions
    
    Args:
        image: Input RGB image (uint8)
        n_colors: Number of color clusters
        region_coherence: Strength of region coherence (0.0-1.0)
    
    Returns:
        numpy.ndarray: Region-aware clustered image
    """
    # Apply slight smoothing to reduce noise while preserving edges
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Perform initial clustering on smoothed image
    initial_clustered = spatial_color_clustering(
        smoothed, n_colors, spatial_weight=0.2, preserve_boundaries=True
    )
    
    if region_coherence > 0:
        # Apply region growing to improve coherence
        initial_clustered = apply_region_growing(image, initial_clustered, region_coherence)
    
    return initial_clustered


def apply_region_growing(original, clustered, coherence_strength):
    """
    Apply region growing to improve color coherence within similar areas
    
    Args:
        original: Original image
        clustered: Initially clustered image
        coherence_strength: Strength of coherence enforcement
    
    Returns:
        numpy.ndarray: Image with improved region coherence
    """
    h, w = original.shape[:2]
    result = clustered.copy()
    
    # Convert to LAB for better perceptual distance
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    clustered_lab = cv2.cvtColor(clustered, cv2.COLOR_RGB2LAB)
    
    # Apply median filter to reduce isolated pixels
    for channel in range(3):
        result[:, :, channel] = cv2.medianBlur(result[:, :, channel], 3)
    
    return result


def enhanced_color_clustering(image, method='adaptive_spatial', **kwargs):
    """
    Enhanced color clustering with multiple algorithms
    
    Args:
        image: Input RGB image (uint8)
        method: Clustering method ('adaptive_spatial', 'region_aware', 'gmm', 'dbscan')
        **kwargs: Method-specific parameters
    
    Returns:
        numpy.ndarray: Enhanced clustered image
    """
    if method == 'adaptive_spatial':
        # Adaptively determine number of colors
        optimal_colors = extract_dominant_colors_adaptive(image)
        print(f"   🎨 Detected {optimal_colors} significant colors")
        
        return spatial_color_clustering(
            image, 
            n_colors=optimal_colors,
            spatial_weight=kwargs.get('spatial_weight', 0.3),
            preserve_boundaries=kwargs.get('preserve_boundaries', True)
        )
    
    elif method == 'region_aware':
        n_colors = kwargs.get('n_colors', 8)
        return region_aware_clustering(
            image,
            n_colors=n_colors,
            region_coherence=kwargs.get('region_coherence', 0.5)
        )
    
    elif method == 'gmm':
        # Gaussian Mixture Model clustering
        return gmm_color_clustering(image, **kwargs)
    
    elif method == 'dbscan':
        # Density-based clustering
        return dbscan_color_clustering(image, **kwargs)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def gmm_color_clustering(image, n_components=8, covariance_type='full'):
    """
    Gaussian Mixture Model clustering for more flexible cluster shapes
    
    Args:
        image: Input RGB image (uint8)
        n_components: Number of Gaussian components
        covariance_type: Type of covariance parameters
    
    Returns:
        numpy.ndarray: GMM clustered image
    """
    # Reshape image data
    data = image.reshape((-1, 3)).astype(np.float32)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        max_iter=100
    )
    
    labels = gmm.fit_predict(data)
    
    # Get cluster centers (means)
    centers = gmm.means_
    centers = np.uint8(np.clip(centers, 0, 255))
    
    # Create output image
    clustered_data = centers[labels]
    clustered_image = clustered_data.reshape(image.shape)
    
    return clustered_image


def dbscan_color_clustering(image, eps=10, min_samples=50, post_kmeans=True):
    """
    DBSCAN clustering to find natural color groupings
    
    Args:
        image: Input RGB image (uint8)
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum samples in a neighborhood for a core point
        post_kmeans: Whether to apply K-means to DBSCAN results
    
    Returns:
        numpy.ndarray: DBSCAN clustered image
    """
    # Reshape and sample data (DBSCAN can be slow on large datasets)
    data = image.reshape((-1, 3)).astype(np.float32)
    
    # Sample data if image is large
    if len(data) > 50000:
        indices = np.random.choice(len(data), 50000, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data
        indices = np.arange(len(data))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    sample_labels = dbscan.fit_predict(sample_data)
    
    # Handle noise points (label = -1)
    n_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
    
    if n_clusters < 2:
        # Fall back to K-means if DBSCAN doesn't find good clusters
        print(f"   ⚠️  DBSCAN found {n_clusters} clusters, falling back to K-means")
        return spatial_color_clustering(image, n_colors=8)
    
    # Calculate cluster centers
    centers = []
    for cluster_id in range(n_clusters):
        cluster_mask = sample_labels == cluster_id
        if np.any(cluster_mask):
            center = np.mean(sample_data[cluster_mask], axis=0)
            centers.append(center)
    
    centers = np.array(centers)
    
    if post_kmeans and len(centers) > 2:
        # Use DBSCAN centers as initial centers for K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, final_centers = cv2.kmeans(
            data, len(centers), None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS
        )
        centers = final_centers
    else:
        # Assign all pixels to nearest center
        distances = cdist(data, centers)
        labels = np.argmin(distances, axis=1)
    
    # Create output image
    centers = np.uint8(np.clip(centers, 0, 255))
    clustered_data = centers[labels]
    clustered_image = clustered_data.reshape(image.shape)
    
    return clustered_image


def visualize_clustering_comparison(image, methods=['original', 'basic_kmeans', 'adaptive_spatial', 'region_aware']):
    """
    Compare different clustering methods side by side
    
    Args:
        image: Input RGB image (uint8)
        methods: List of methods to compare
    
    Returns:
        None (displays plot)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    results = {}
    
    for i, method in enumerate(methods):
        if method == 'original':
            results[method] = image
        elif method == 'basic_kmeans':
            # Original basic K-means
            data = image.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            clustered_data = centers[labels.flatten()]
            results[method] = clustered_data.reshape(image.shape)
        else:
            results[method] = enhanced_color_clustering(image, method=method)
        
        axes[i].imshow(results[method])
        axes[i].set_title(method.replace('_', ' ').title())
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Test the enhanced clustering
    import sys
    sys.path.append('..')
    
    # Load test image
    test_image = cv2.imread("../examples/tower.jpg")
    if test_image is not None:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Compare methods
        print("Comparing clustering methods...")
        visualize_clustering_comparison(test_image)
    else:
        print("Test image not found")
