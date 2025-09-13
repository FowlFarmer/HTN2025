import cv2
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path

try:
    import sknw
    SKNW_AVAILABLE = True
except ImportError:
    SKNW_AVAILABLE = False
    print("Warning: sknw not available. Install with: pip install sknw")

def find_endpoints(skeleton):
    """
    Find endpoints in skeleton image

    Args:
        skeleton: Binary skeleton image

    Returns:
        list: List of endpoint coordinates
    """
    # Morphological operation to find endpoints
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]], dtype=np.uint8)

    result = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints_mask = (result == 11) & (skeleton > 0)

    endpoints = np.where(endpoints_mask)
    return list(zip(endpoints[1], endpoints[0]))  # (x, y) format

def repair_fragmented_strokes(skeleton, max_gap=6):
    """
    Connect nearby endpoints to repair broken strokes

    Args:
        skeleton: Binary skeleton image
        max_gap: Maximum distance to connect endpoints

    Returns:
        numpy.ndarray: Repaired skeleton
    """
    skeleton_copy = skeleton.copy()

    # Find endpoints
    endpoints = find_endpoints(skeleton_copy)

    # Connect nearby endpoints
    connected_pairs = set()
    for i, ep1 in enumerate(endpoints):
        for j, ep2 in enumerate(endpoints[i+1:], i+1):
            if (i, j) in connected_pairs or (j, i) in connected_pairs:
                continue

            distance = np.linalg.norm(np.array(ep1) - np.array(ep2))
            if distance <= max_gap:
                # Draw line between endpoints
                cv2.line(skeleton_copy, ep1, ep2, 1, 1)
                connected_pairs.add((i, j))

    return skeleton_copy.astype(bool)

def is_eulerian_graph(G):
    """
    Check if graph has an Eulerian path or cycle

    Args:
        G: NetworkX graph

    Returns:
        dict: Eulerian properties
    """
    if len(G.nodes()) == 0:
        return {'is_eulerian': False, 'type': 'empty'}

    # Check connectivity (ignoring isolated vertices)
    non_isolated = [n for n in G.nodes() if G.degree(n) > 0]
    if len(non_isolated) == 0:
        return {'is_eulerian': False, 'type': 'no_edges'}

    subgraph = G.subgraph(non_isolated)
    if not nx.is_connected(subgraph):
        return {'is_eulerian': False, 'type': 'disconnected'}

    # Count nodes with odd degree
    odd_degree_nodes = [n for n in G.nodes() if G.degree(n) % 2 == 1]
    num_odd = len(odd_degree_nodes)

    if num_odd == 0:
        return {'is_eulerian': True, 'type': 'cycle', 'start_nodes': list(G.nodes())}
    elif num_odd == 2:
        return {'is_eulerian': True, 'type': 'path', 'start_nodes': odd_degree_nodes}
    else:
        return {'is_eulerian': False, 'type': 'neither', 'odd_nodes': odd_degree_nodes}

def find_eulerian_path(G):
    """
    Find Eulerian path using Hierholzer's algorithm

    Args:
        G: NetworkX graph

    Returns:
        list: Sequence of nodes forming Eulerian path, or None
    """
    eulerian_info = is_eulerian_graph(G)
    if not eulerian_info['is_eulerian']:
        return None

    # Create mutable copy
    G_copy = G.copy()

    # Choose starting node
    if eulerian_info['type'] == 'path':
        start_node = eulerian_info['start_nodes'][0]
    else:
        start_node = next(iter(G_copy.nodes()))

    # Hierholzer's algorithm
    path = []
    stack = [start_node]

    while stack:
        current = stack[-1]
        if G_copy.degree(current) == 0:
            path.append(stack.pop())
        else:
            neighbors = list(G_copy.neighbors(current))
            if neighbors:
                next_node = neighbors[0]
                G_copy.remove_edge(current, next_node)
                stack.append(next_node)

    return path[::-1]  # Reverse to get correct order

def find_nearest_unvisited_edge(G, current_node, visited_edges):
    """
    Find nearest node with unvisited edges

    Args:
        G: NetworkX graph
        current_node: Current position
        visited_edges: Set of visited edges

    Returns:
        Node with unvisited edges
    """
    nodes_with_unvisited = []

    for node in G.nodes():
        for neighbor in G.neighbors(node):
            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in visited_edges:
                nodes_with_unvisited.append(node)
                break

    if not nodes_with_unvisited:
        return None

    # Find nearest node (using node coordinates if available)
    if hasattr(G.nodes[current_node], 'pos'):
        current_pos = G.nodes[current_node]['pos']
        distances = []
        for node in nodes_with_unvisited:
            if hasattr(G.nodes[node], 'pos'):
                node_pos = G.nodes[node]['pos']
                dist = np.linalg.norm(np.array(current_pos) - np.array(node_pos))
                distances.append((dist, node))

        if distances:
            return min(distances)[1]

    return nodes_with_unvisited[0]

def greedy_traversal(G, start_node=None):
    """
    Greedy traversal minimizing pen lifts

    Args:
        G: NetworkX graph
        start_node: Starting node (if None, chooses optimal start)

    Returns:
        dict: Traversal result with path and pen lifts
    """
    if G.number_of_edges() == 0:
        return {'path': [], 'pen_lifts': 0, 'segments': []}

    # Choose starting node
    if start_node is None:
        # Prefer endpoints (degree 1) or nodes with odd degree
        candidates = [n for n in G.nodes() if G.degree(n) == 1]
        if not candidates:
            candidates = [n for n in G.nodes() if G.degree(n) % 2 == 1]
        if not candidates:
            candidates = list(G.nodes())
        start_node = candidates[0]

    path_segments = []
    current_segment = []
    current = start_node
    visited_edges = set()
    pen_lifts = 0

    while len(visited_edges) < G.number_of_edges():
        current_segment.append(current)

        # Find unvisited edge from current node
        unvisited_neighbors = []
        for neighbor in G.neighbors(current):
            edge = (min(current, neighbor), max(current, neighbor))
            if edge not in visited_edges:
                unvisited_neighbors.append(neighbor)

        if unvisited_neighbors:
            # Continue current stroke
            next_node = unvisited_neighbors[0]
            edge = (min(current, next_node), max(current, next_node))
            visited_edges.add(edge)
            current = next_node
        else:
            # End current segment and start new one
            if current_segment:
                path_segments.append(current_segment.copy())
                current_segment = []

            # Find nearest unvisited edge
            next_start = find_nearest_unvisited_edge(G, current, visited_edges)
            if next_start is not None:
                current = next_start
                pen_lifts += 1
            else:
                break

    # Add final segment
    if current_segment:
        path_segments.append(current_segment)

    # Flatten path
    full_path = []
    for segment in path_segments:
        full_path.extend(segment)

    return {
        'path': full_path,
        'pen_lifts': pen_lifts,
        'segments': path_segments
    }

class StrokeGraph:
    """
    Graph representation of skeleton for stroke analysis
    """

    def __init__(self, skeleton, repair_gaps=True, max_gap=6):
        """
        Initialize stroke graph from skeleton

        Args:
            skeleton: Binary skeleton image (boolean array)
            repair_gaps: Whether to repair fragmented strokes
            max_gap: Maximum gap to repair
        """
        self.skeleton = skeleton.astype(np.uint8)
        self.original_skeleton = skeleton.copy()

        if repair_gaps:
            self.skeleton = repair_fragmented_strokes(self.skeleton, max_gap).astype(np.uint8)

        # Build graph
        if SKNW_AVAILABLE:
            self.G = sknw.build_sknw(self.skeleton)
        else:
            # Fallback: create simple graph
            self.G = self._build_simple_graph()

        # Analyze graph structure
        self._analyze_structure()

    def _build_simple_graph(self):
        """
        Simple fallback graph construction when sknw is not available
        """
        G = nx.Graph()

        # Find connected components
        num_labels, labels = cv2.connectedComponents(self.skeleton)

        for label in range(1, num_labels):
            component = (labels == label).astype(np.uint8)

            # Find endpoints and add as nodes
            endpoints = find_endpoints(component)

            if len(endpoints) >= 2:
                # Add nodes
                for i, ep in enumerate(endpoints):
                    node_id = f"{label}_{i}"
                    G.add_node(node_id, pos=ep)

                # Connect first and last endpoints (simple approximation)
                if len(endpoints) == 2:
                    G.add_edge(f"{label}_0", f"{label}_1")

        return G

    def _analyze_structure(self):
        """
        Analyze graph structure and compute properties
        """
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges(data=True))

        # Identify node types
        self.endpoints = [n for n in self.nodes if self.G.degree(n) == 1]
        self.junctions = [n for n in self.nodes if self.G.degree(n) > 2]
        self.regular_nodes = [n for n in self.nodes if self.G.degree(n) == 2]

        # Analyze Eulerian properties
        self.eulerian_info = is_eulerian_graph(self.G)

        # Compute basic statistics
        self.stats = {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_endpoints': len(self.endpoints),
            'num_junctions': len(self.junctions),
            'is_eulerian': self.eulerian_info['is_eulerian'],
            'eulerian_type': self.eulerian_info.get('type', 'unknown')
        }

        # Compute semantic drawing order
        self.semantic_order = self.get_semantic_drawing_order()

    def get_edge_points(self, edge):
        """
        Get pixel coordinates for an edge

        Args:
            edge: Edge tuple (node1, node2)

        Returns:
            numpy.ndarray: Array of (x, y) coordinates along edge
        """
        if SKNW_AVAILABLE and 'pts' in self.G[edge[0]][edge[1]]:
            return self.G[edge[0]][edge[1]]['pts']
        else:
            # Fallback: simple line between nodes
            if 'pos' in self.G.nodes[edge[0]] and 'pos' in self.G.nodes[edge[1]]:
                start = self.G.nodes[edge[0]]['pos']
                end = self.G.nodes[edge[1]]['pos']

                # Generate points along line
                num_points = int(np.linalg.norm(np.array(end) - np.array(start))) + 1
                if num_points > 1:
                    x_coords = np.linspace(start[0], end[0], num_points)
                    y_coords = np.linspace(start[1], end[1], num_points)
                    return np.column_stack([x_coords, y_coords])

            return np.array([])

    def find_optimal_traversal(self):
        """
        Find optimal traversal path for drawing

        Returns:
            dict: Traversal information
        """
        if self.eulerian_info['is_eulerian']:
            # Use Eulerian path
            eulerian_path = find_eulerian_path(self.G)
            if eulerian_path:
                return {
                    'method': 'eulerian',
                    'path': eulerian_path,
                    'pen_lifts': 0,
                    'segments': [eulerian_path]
                }

        # Fall back to greedy traversal
        greedy_result = greedy_traversal(self.G)
        greedy_result['method'] = 'greedy'
        return greedy_result

    def get_stroke_sequences(self):
        """
        Get sequences of points for each stroke segment

        Returns:
            list: List of stroke segments, each containing point sequences
        """
        traversal = self.find_optimal_traversal()
        stroke_sequences = []

        for segment in traversal['segments']:
            segment_points = []

            # Convert node sequence to point sequence
            for i in range(len(segment) - 1):
                edge = (segment[i], segment[i + 1])
                if edge in self.G.edges():
                    points = self.get_edge_points(edge)
                else:
                    # Try reversed edge
                    edge = (segment[i + 1], segment[i])
                    points = self.get_edge_points(edge)
                    if len(points) > 0:
                        points = points[::-1]  # Reverse direction

                if len(points) > 0:
                    if not segment_points:  # First edge in segment
                        segment_points.extend(points)
                    else:  # Subsequent edges - avoid duplicating junction
                        segment_points.extend(points[1:])

            if segment_points:
                stroke_sequences.append(np.array(segment_points))

        return stroke_sequences

    def get_semantic_drawing_order(self, mask_contour=None):
        """
        Get semantically meaningful drawing order that preserves visual hierarchy

        Args:
            mask_contour: Optional contour of the mask for boundary detection

        Returns:
            dict: Semantic drawing order with classified strokes
        """
        from scipy.spatial.distance import cdist
        from scipy.ndimage import binary_fill_holes

        if len(self.edges) == 0:
            return {
                'drawing_order': [],
                'stroke_classification': {},
                'drawing_phases': {'boundary': [], 'internal': [], 'detail': []},
                'total_strokes': 0
            }

        # Classify strokes by type
        stroke_classification = self._classify_strokes(mask_contour)

        # Create hierarchical drawing order
        drawing_phases = self._create_hierarchical_order(stroke_classification)

        # Flatten into single drawing order while preserving hierarchy
        drawing_order = []
        drawing_order.extend(drawing_phases['boundary'])
        drawing_order.extend(drawing_phases['internal'])
        drawing_order.extend(drawing_phases['detail'])

        return {
            'drawing_order': drawing_order,
            'stroke_classification': stroke_classification,
            'drawing_phases': drawing_phases,
            'total_strokes': len(drawing_order)
        }

    def _classify_strokes(self, mask_contour=None):
        """
        Classify strokes into boundary, internal, and detail categories

        Args:
            mask_contour: Optional contour for boundary detection

        Returns:
            dict: Classification of each edge
        """
        classification = {}

        # Convert graph edges to stroke segments
        stroke_segments = []
        for edge_data in self.edges:
            # Handle edge format: (node1, node2, data_dict)
            if len(edge_data) == 3:
                edge = (edge_data[0], edge_data[1])  # Extract just the node pair
            else:
                edge = edge_data

            points = self.get_edge_points(edge)
            if len(points) > 0:
                stroke_segments.append({
                    'edge': edge,
                    'points': points,
                    'length': len(points),
                    'endpoints': [points[0], points[-1]]
                })

        if not stroke_segments:
            return classification

        # Calculate stroke properties for classification
        stroke_lengths = [seg['length'] for seg in stroke_segments]
        avg_length = np.mean(stroke_lengths)
        length_threshold_detail = avg_length * 0.3  # Short strokes = details
        length_threshold_major = avg_length * 0.8   # Long strokes = major features

        # Analyze spatial distribution
        all_points = np.vstack([seg['points'] for seg in stroke_segments])
        centroid = np.mean(all_points, axis=0)

        # Calculate distances from image boundaries
        h, w = self.skeleton.shape
        boundary_distance_threshold = min(h, w) * 0.1

        for i, segment in enumerate(stroke_segments):
            edge = segment['edge']
            points = segment['points']
            length = segment['length']

            # Calculate features for classification
            features = self._calculate_stroke_features(segment, centroid, (h, w))

            # Classification logic
            if features['is_boundary'] or features['distance_to_edge'] < boundary_distance_threshold:
                classification[edge] = 'boundary'
            elif length < length_threshold_detail or features['is_isolated_detail']:
                classification[edge] = 'detail'
            else:
                classification[edge] = 'internal'

        return classification

    def _calculate_stroke_features(self, segment, image_centroid, image_shape):
        """
        Calculate features for stroke classification

        Args:
            segment: Stroke segment dictionary
            image_centroid: Center point of all strokes
            image_shape: (height, width) of skeleton image

        Returns:
            dict: Feature values for classification
        """
        points = segment['points']
        length = segment['length']
        h, w = image_shape

        # Distance to image boundaries
        min_distances_to_edges = []
        for point in points:
            y, x = point
            dist_to_edge = min(x, y, w - x, h - y)
            min_distances_to_edges.append(dist_to_edge)

        avg_distance_to_edge = np.mean(min_distances_to_edges)
        min_distance_to_edge = min(min_distances_to_edges)

        # Distance to image centroid
        segment_centroid = np.mean(points, axis=0)
        distance_to_center = np.linalg.norm(segment_centroid - image_centroid)

        # Stroke shape analysis
        if len(points) > 2:
            # Calculate straightness (deviation from straight line)
            start_point = points[0]
            end_point = points[-1]
            direct_distance = np.linalg.norm(end_point - start_point)
            path_length = length
            straightness = direct_distance / path_length if path_length > 0 else 0
        else:
            straightness = 1.0

        # Boundary detection heuristics
        boundary_threshold = min(h, w) * 0.15
        is_boundary = (min_distance_to_edge < boundary_threshold and
                      length > np.mean([h, w]) * 0.05)  # Long enough to be structural

        # Detail detection heuristics
        detail_length_threshold = np.mean([h, w]) * 0.02
        is_isolated_detail = (length < detail_length_threshold or
                            (straightness < 0.3 and length < detail_length_threshold * 2))

        return {
            'distance_to_edge': avg_distance_to_edge,
            'min_distance_to_edge': min_distance_to_edge,
            'distance_to_center': distance_to_center,
            'straightness': straightness,
            'is_boundary': is_boundary,
            'is_isolated_detail': is_isolated_detail,
            'length': length
        }

    def _create_hierarchical_order(self, stroke_classification):
        """
        Create hierarchical drawing order based on stroke classification

        Args:
            stroke_classification: Dict mapping edges to categories

        Returns:
            dict: Drawing phases with ordered strokes
        """
        phases = {'boundary': [], 'internal': [], 'detail': []}

        # Group strokes by classification
        for edge, category in stroke_classification.items():
            if category in phases:
                phases[category].append(edge)

        # Sort each phase by drawing priority
        phases['boundary'] = self._sort_boundary_strokes(phases['boundary'])
        phases['internal'] = self._sort_internal_strokes(phases['internal'])
        phases['detail'] = self._sort_detail_strokes(phases['detail'])

        return phases

    def _sort_boundary_strokes(self, boundary_strokes):
        """
        Sort boundary strokes by drawing priority (larger/outer first)

        Args:
            boundary_strokes: List of boundary stroke edges

        Returns:
            list: Sorted boundary strokes
        """
        if not boundary_strokes:
            return []

        # Sort by length (longer boundaries first)
        stroke_data = []
        for edge in boundary_strokes:
            points = self.get_edge_points(edge)
            if len(points) > 0:
                length = len(points)
                # Calculate "enclosure" - how far from center the stroke is
                centroid = np.mean(points, axis=0)
                h, w = self.skeleton.shape
                image_center = np.array([h/2, w/2])
                distance_from_center = np.linalg.norm(centroid - image_center)

                stroke_data.append({
                    'edge': edge,
                    'length': length,
                    'distance_from_center': distance_from_center,
                    'priority': distance_from_center + length * 0.1  # Favor outer + long strokes
                })

        # Sort by priority (higher = draw first)
        stroke_data.sort(key=lambda x: x['priority'], reverse=True)
        return [item['edge'] for item in stroke_data]

    def _sort_internal_strokes(self, internal_strokes):
        """
        Sort internal strokes by drawing priority (major features first)

        Args:
            internal_strokes: List of internal stroke edges

        Returns:
            list: Sorted internal strokes
        """
        if not internal_strokes:
            return []

        # Sort by length and centrality (longer, more central first)
        stroke_data = []
        h, w = self.skeleton.shape
        image_center = np.array([h/2, w/2])

        for edge in internal_strokes:
            points = self.get_edge_points(edge)
            if len(points) > 0:
                length = len(points)
                centroid = np.mean(points, axis=0)
                distance_from_center = np.linalg.norm(centroid - image_center)
                centrality = 1.0 / (1.0 + distance_from_center)  # Higher = more central

                stroke_data.append({
                    'edge': edge,
                    'length': length,
                    'centrality': centrality,
                    'priority': length * 0.5 + centrality * 100  # Balance length and centrality
                })

        # Sort by priority (higher = draw first)
        stroke_data.sort(key=lambda x: x['priority'], reverse=True)
        return [item['edge'] for item in stroke_data]

    def _sort_detail_strokes(self, detail_strokes):
        """
        Sort detail strokes by drawing priority (connected to major features first)

        Args:
            detail_strokes: List of detail stroke edges

        Returns:
            list: Sorted detail strokes
        """
        if not detail_strokes:
            return []

        # Sort details by connectivity to major strokes
        stroke_data = []

        for edge in detail_strokes:
            points = self.get_edge_points(edge)
            if len(points) > 0:
                length = len(points)

                # Calculate connectivity (number of connected nodes)
                connectivity = self.G.degree(edge[0]) + self.G.degree(edge[1])

                stroke_data.append({
                    'edge': edge,
                    'length': length,
                    'connectivity': connectivity,
                    'priority': connectivity * 10 + length  # Favor connected details
                })

        # Sort by priority (higher = draw first)
        stroke_data.sort(key=lambda x: x['priority'], reverse=True)
        return [item['edge'] for item in stroke_data]

def build_stroke_graph(skeleton, repair_gaps=True, max_gap=6):
    """
    Build stroke graph from skeleton

    Args:
        skeleton: Binary skeleton image
        repair_gaps: Whether to repair fragmented strokes
        max_gap: Maximum gap distance to repair

    Returns:
        StrokeGraph: Graph representation of skeleton
    """
    return StrokeGraph(skeleton, repair_gaps, max_gap)

def process_all_stroke_graphs(edge_results):
    """
    Process all edge extraction results into stroke graphs

    Args:
        edge_results: List of edge extraction results from Step 3

    Returns:
        list: List of StrokeGraph objects
    """
    stroke_graphs = []

    for i, result in enumerate(edge_results):
        if result is not None:
            print(f"   🔗 Building graph for mask {i+1}...")

            skeleton = result['skeleton']
            stroke_graph = build_stroke_graph(skeleton)

            # Add metadata
            stroke_graph.mask_id = i
            stroke_graph.extraction_stats = result['stats']
            stroke_graph.offset = result['offset']

            stroke_graphs.append(stroke_graph)

            # Print graph statistics
            stats = stroke_graph.stats
            semantic = stroke_graph.semantic_order
            print(f"      📊 Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
            print(f"      🎯 Endpoints: {stats['num_endpoints']}, Junctions: {stats['num_junctions']}")
            print(f"      ✨ Eulerian: {stats['is_eulerian']} ({stats['eulerian_type']})")
            print(f"      🎨 Semantic: {len(semantic['drawing_phases']['boundary'])} boundary, "
                  f"{len(semantic['drawing_phases']['internal'])} internal, "
                  f"{len(semantic['drawing_phases']['detail'])} detail strokes")
        else:
            print(f"   ⚠️  Skipping empty result for mask {i+1}")
            stroke_graphs.append(None)

    return stroke_graphs

def save_graph_visualization(stroke_graphs, output_dir="examples"):
    """
    Save stroke graph visualization

    Args:
        stroke_graphs: List of StrokeGraph objects
        output_dir: Directory to save results

    Returns:
        str: Path to saved visualization
    """
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    valid_graphs = [g for g in stroke_graphs if g is not None]

    for i, graph in enumerate(valid_graphs[:6]):
        if i >= 6:
            break

        ax = axes[i]

        # Get semantic drawing order
        semantic = graph.semantic_order

        # Draw graph structure
        if len(graph.nodes) > 0:
            # Draw skeleton as background
            ax.imshow(graph.skeleton, cmap='gray', alpha=0.2)

            # Draw nodes
            if SKNW_AVAILABLE:
                for node in graph.nodes:
                    if hasattr(graph.G.nodes[node], 'o'):
                        y, x = graph.G.nodes[node]['o']  # sknw uses (y,x) order
                        if graph.G.degree(node) == 1:
                            ax.plot(x, y, 'ro', markersize=6, alpha=0.7)
                        elif graph.G.degree(node) > 2:
                            ax.plot(x, y, 'bs', markersize=6, alpha=0.7)
                        else:
                            ax.plot(x, y, 'go', markersize=3, alpha=0.5)

            # Draw edges with semantic classification
            # Color scheme: Red=Boundary, Blue=Internal, Green=Detail
            colors = {'boundary': 'red', 'internal': 'blue', 'detail': 'green'}
            alphas = {'boundary': 0.9, 'internal': 0.7, 'detail': 0.5}
            widths = {'boundary': 3, 'internal': 2, 'detail': 1}

            for edge, classification in semantic['stroke_classification'].items():
                points = graph.get_edge_points(edge)
                if len(points) > 0:
                    # Convert points to x,y coordinates for plotting
                    y_coords = [p[0] for p in points]
                    x_coords = [p[1] for p in points]

                    color = colors.get(classification, 'gray')
                    alpha = alphas.get(classification, 0.5)
                    width = widths.get(classification, 1)

                    ax.plot(x_coords, y_coords, color=color, alpha=alpha,
                           linewidth=width, solid_capstyle='round')

            # Add drawing order numbers for boundary strokes (most important)
            boundary_strokes = semantic['drawing_phases']['boundary'][:10]  # Show first 10
            for order_idx, edge in enumerate(boundary_strokes):
                points = graph.get_edge_points(edge)
                if len(points) > 0:
                    # Place number at midpoint of stroke
                    mid_idx = len(points) // 2
                    y, x = points[mid_idx]
                    ax.text(x, y, str(order_idx + 1), fontsize=8, fontweight='bold',
                           color='white', ha='center', va='center',
                           bbox=dict(boxstyle='circle,pad=0.1', facecolor='red', alpha=0.8))

        # Display title with semantic information
        boundary_count = len(semantic['drawing_phases']['boundary'])
        internal_count = len(semantic['drawing_phases']['internal'])
        detail_count = len(semantic['drawing_phases']['detail'])

        ax.set_title(f'Graph {i+1}: Semantic Order\n'
                    f'{graph.stats["num_nodes"]} nodes, {graph.stats["num_edges"]} edges\n'
                    f'{boundary_count} boundary, {internal_count} internal, {detail_count} detail',
                    fontsize=10)
        ax.axis('off')
        ax.invert_yaxis()  # Invert y-axis to match image coordinates

    # Hide unused subplots
    for i in range(len(valid_graphs), 6):
        axes[i].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, label='Boundary strokes (drawn first)'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Internal strokes (drawn second)'),
        plt.Line2D([0], [0], color='green', lw=1, label='Detail strokes (drawn last)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    plt.suptitle(f'Semantic Drawing Order Analysis - {timestamp}', fontsize=16)
    plt.tight_layout()

    # Save with timestamp
    output_path = f"{output_dir}/stroke_graphs_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   💾 Saved graph visualization: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test stroke graph construction
    import sys
    sys.path.append('..')

    from step1_preprocessing.image_preprocessing import preprocess_image
    from step2_segmentation.segmentation import segment_painting
    from step3_edge_extraction.edge_extraction import process_all_masks

    print("🔗 Testing Stroke Graph Construction")
    print("=" * 40)

    # Process test image
    image_path = "../examples/monet.jpeg"

    # Get edge results
    processed_img, _, _ = preprocess_image(image_path)
    masks, scores, _ = segment_painting(image_path)
    edge_results = process_all_masks(processed_img, masks)

    # Build stroke graphs
    stroke_graphs = process_all_stroke_graphs(edge_results)

    # Save visualization
    output_path = save_graph_visualization(stroke_graphs)

    print("✅ Stroke graph construction complete!")
    print(f"📊 Results: {len([g for g in stroke_graphs if g is not None])} graphs generated")