import numpy as np
import networkx as nx
from matplotlib import cm
from typing import List, Sequence, Literal
from collections import defaultdict
from plyfile import PlyData, PlyElement
from random import shuffle
from itertools import product as itertools_product
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase import Atoms

def face_normal(verts, face):
    """Compute normal of quad face (area-weighted sum of 2 triangle normals)."""
    v0, v1, v2, v3 = verts[face]
    n1 = np.cross(v1 - v0, v2 - v0)
    n2 = np.cross(v2 - v0, v3 - v0)
    return n1 + n2

def most_upward_face(faces, vertices):
    """Find the face whose normal points most in +Z and is flat w.r.t. XY plane."""
    max_dot = -np.inf
    best_face_idx = 0
    for i, f in enumerate(faces):
        n = face_normal(vertices, f)
        n /= np.linalg.norm(n) + 1e-8
        dot = n[2]  # alignment with +Z
        if dot > max_dot:
            max_dot = dot
            best_face_idx = i
    return best_face_idx

def reorient_faces_from_seed(faces, vertices):
    """
    Orient quad faces consistently across a mesh.
    Starts from the upward-facing face, propagates via adjacency.
    """

    faces = faces.copy()
    face_count = len(faces)

    # Build edge-to-face map
    edge_map = {}
    for i, f in enumerate(faces):
        for j in range(4):
            a = f[j]
            b = f[(j + 1) % 4]
            key = tuple(sorted((a, b)))
            edge_map.setdefault(key, []).append(i)

    # Build adjacency graph
    graph = nx.Graph()
    graph.add_nodes_from(range(face_count))
    for edge, fs in edge_map.items():
        if len(fs) == 2:
            a, b = fs
            graph.add_edge(a, b)

    visited = np.zeros(face_count, dtype=bool)
    oriented_faces = faces.copy()

    # Start with best upward-facing face
    seed_idx = most_upward_face(faces, vertices)
    visited[seed_idx] = True
    queue = [seed_idx]

    while queue:
        current = queue.pop()
        f0 = oriented_faces[current]

        for neighbor in graph.neighbors(current):
            if visited[neighbor]:
                continue

            f1 = oriented_faces[neighbor]

            # Find shared edge
            shared = None
            for i in range(4):
                a0, b0 = f0[i], f0[(i + 1) % 4]
                for j in range(4):
                    a1, b1 = f1[j], f1[(j + 1) % 4]
                    if {a0, b0} == {a1, b1}:
                        shared = (a0, b0)
                        break
                if shared:
                    break

            if not shared:
                continue

            # Compare direction of shared edge
            edge_f0 = (f0.tolist().index(shared[0]), f0.tolist().index(shared[1]))
            edge_f1 = (f1.tolist().index(shared[0]), f1.tolist().index(shared[1]))

            # If both faces use edge in same direction, neighbor face must flip
            idx0 = f0.tolist().index(shared[0])
            idx1 = f1.tolist().index(shared[0])
            next0 = f0[(idx0 + 1) % 4]
            next1 = f1[(idx1 + 1) % 4]
            same_dir = next0 == shared[1] and next1 == shared[1]

            if same_dir:
                oriented_faces[neighbor] = f1[::-1]  # flip neighbor

            visited[neighbor] = True
            queue.append(neighbor)

    return oriented_faces

def compute_outward_vertex_normals_quads(vertices, faces, mode=Literal['particle', 'slab']):
    """
    Computes per-vertex normals from a quad mesh (faces.shape = (M, 4)).
    Ensures normals point outward for a closed surface.
    """

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    vnormals = np.zeros_like(vertices, dtype=np.float32)

    for face in faces:
        v0, v1, v2, v3 = vertices[face]

        # Split quad into 2 triangles: [v0,v1,v2], [v0,v2,v3]
        n1 = np.cross(v1 - v0, v2 - v0)
        n2 = np.cross(v2 - v0, v3 - v0)
        fnorm = n1 + n2  # total quad face normal (area-weighted)

        for idx in face:
            vnormals[idx] += fnorm

    # Normalize
    vnormals /= np.linalg.norm(vnormals, axis=1, keepdims=True)

    # Ensure outward normals (based on center of mesh)
    if mode == 'particle':
        center = vertices.mean(axis=0)
        outward = vertices - center
    elif mode == 'slab':
        outward = np.array([(0,0,1) for _ in vertices])
    else:
        raise ValueError(f'mode: {mode} not supported.')
    
    dot = np.einsum("ij,ij->i", vnormals, outward)
    if np.mean(dot) < 0:
        vnormals *= -1

    return vnormals

def save_ply_quads(filename, vertices, faces, normals=None, vertex_colors=None):
    """
    Save a quad mesh to PLY with vertex normals and vertex colors (Blender-compatible).
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    if normals is None:
        normals = np.zeros_like(vertices)

    if vertex_colors is None:
        vertex_colors = np.random.randint(0, 256, size=(len(vertices), 3), dtype=np.uint8)
    else:
        vertex_colors = np.asarray(vertex_colors) #.astype(np.uint8)
        if vertex_colors.shape[0] != len(vertices):
            raise ValueError("vertex_colors must match number of vertices")

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, n, c in zip(vertices, normals, vertex_colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]} {c[0]} {c[1]} {c[2]}\n")

        for face in faces:
            f.write(f"4 {face[0]} {face[1]} {face[2]} {face[3]}\n")


def values_to_colors(target_value, values, reference_value=None, palette_nam="plasma"):
    """
    Map a target value to an RGB color using the plasma colormap.
    
    Args:
        target_value (float): The value to map to a color.
        values (list or array): The list of values that define the min and max range.
        reference_value (float, optional): A reference value for normalization. 
                                           If None, only min/max scaling is used.
    
    Returns:
        tuple: (R, G, B) values in the range [0, 255].
    """
    values = np.array(values)
    vmin, vmax = np.min(values), np.max(values)

    # Normalize the value
    if reference_value is not None:
        # Use symmetric normalization around reference_value
        max_dev = max(reference_value - vmin, vmax - reference_value)
        norm_value = (target_value - reference_value) / (2 * max_dev) + 0.5
    else:
        # Simple min-max normalization
        norm_value = (target_value - vmin) / (vmax - vmin)
    
    # Clip to [0, 1] range
    norm_value = np.clip(norm_value, 0, 1)

    # Get color from plasma colormap
    cmap = cm.get_cmap(palette_nam)
    rgba = cmap(norm_value)

    # Convert to RGB (0–255)
    rgb = tuple(int(255 * c) for c in rgba[:3])
    return rgb


def compute_vertex_normals(vertices, faces, normals):
    """
    Compute vertex normals by averaging neighboring face normals.
    Each face normal is normalized before averaging.

    Parameters
    ----------
    vertices : list of [x, y, z]
        List of vertex positions.
    faces : list of list[int]
        List of faces, each given as indices into `vertices`.
    normals : list of [nx, ny, nz]
        List of per-face normals (must match length of faces).

    Returns
    -------
    list of [nx, ny, nz]
        Averaged and normalized vertex normals.
    """
    vertices = np.array(vertices, dtype=float)
    normals = np.array(normals, dtype=float)

    # normalize each face normal first
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths[norm_lengths == 0] = 1.0
    normals = normals / norm_lengths

    vnormals = np.zeros_like(vertices)

    for face, fnormal in zip(faces, normals):
        for idx in face:
            vnormals[idx] += fnormal

    # normalize each vertex normal
    for i in range(len(vnormals)):
        norm = np.linalg.norm(vnormals[i])
        if norm > 0:
            vnormals[i] /= norm

    return vnormals.tolist()


def compute_vertex_gradients(vertices, faces, values):
    """
    Compute gradient at each vertex using distance-weighted contributions from first neighbors.
    
    Parameters:
    -----------
    vertices : array-like, shape (n_vertices, 3)
        3D coordinates of vertices
    faces : array-like, shape (n_faces, 3) 
        Triangular faces defined by vertex indices
    values : array-like, shape (n_vertices,)
        Scalar values at each vertex
    
    Returns:
    --------
    gradients : ndarray, shape (n_vertices, 3)
        Gradient vectors at each vertex
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    values = np.asarray(values)
    
    n_vertices = len(vertices)
    gradients = np.zeros((n_vertices, 3))
    
    # Build adjacency graph from faces
    adjacency = defaultdict(set)
    
    # Each face contributes edges between all pairs of vertices
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i]].add(face[j])
    
    # Compute gradient at each vertex
    for vertex_idx in range(n_vertices):
        neighbors = list(adjacency[vertex_idx])
        
        if len(neighbors) == 0:
            continue
            
        vertex_pos = vertices[vertex_idx]
        vertex_value = values[vertex_idx]
        
        # Compute weighted gradient contributions
        weighted_gradient = np.zeros(3)
        total_weight = 0.0
        
        for neighbor_idx in neighbors:
            neighbor_pos = vertices[neighbor_idx]
            neighbor_value = values[neighbor_idx]
            
            # Vector from current vertex to neighbor
            edge_vector = neighbor_pos - vertex_pos
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-12:  # Avoid division by zero
                # Normalized edge direction
                edge_direction = edge_vector / edge_length
                
                # Value difference along edge
                value_diff = neighbor_value - vertex_value
                
                # Weight inversely proportional to distance
                weight = 1.0 / edge_length
                
                # Gradient contribution: (value_diff / distance) * direction * weight
                gradient_contribution = (value_diff / edge_length) * edge_direction * weight
                
                weighted_gradient += gradient_contribution
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 1e-12:
            gradients[vertex_idx] = weighted_gradient / total_weight
    return gradients


def compute_gradients_per_column(energies, grid, faces):
    """
    Compute vertex gradients for energies, column by column if 2D.

    Args:
        energies : np.ndarray
            Shape (n_positions,) or (n_positions, n_conformers)
        grid : np.ndarray
            Vertex coordinates
        faces : np.ndarray
            Mesh faces

    Returns:
        grads : np.ndarray
            Shape (n_positions, n_conformers, 3)
        grad_norms : np.ndarray
            Shape (n_positions, n_conformers)
    """
    energies = np.atleast_2d(energies)  # shape -> (n_positions, n_conformers)
    n_positions, n_confs = energies.shape

    grads = np.zeros((n_positions, n_confs, 3))
    grad_norms = np.zeros((n_positions, n_confs))

    for j in range(n_confs):
        grad_j = compute_vertex_gradients(vertices=grid, faces=faces, values=energies[:, j])
        # grads[:, j, :] = grad_j
        grad_norms[:, j] = np.linalg.norm(grad_j, axis=1)

    return grads, grad_norms



def compute_vertex_gradients_least_squares(vertices, faces, values):
    """
    Alternative implementation using least squares fitting to compute gradients.
    Often more robust for irregular meshes.
    
    Parameters:
    -----------
    vertices : array-like, shape (n_vertices, 3)
        3D coordinates of vertices
    faces : array-like, shape (n_faces, 3)
        Triangular faces defined by vertex indices
    values : array-like, shape (n_vertices,)
        Scalar values at each vertex
    
    Returns:
    --------
    gradients : ndarray, shape (n_vertices, 3)
        Gradient vectors at each vertex
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    values = np.asarray(values)
    
    n_vertices = len(vertices)
    gradients = np.zeros((n_vertices, 3))
    
    # Build adjacency graph
    adjacency = defaultdict(set)
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i]].add(face[j])
    
    # Compute gradient at each vertex using least squares
    for vertex_idx in range(n_vertices):
        neighbors = list(adjacency[vertex_idx])
        
        if len(neighbors) < 3:  # Need at least 3 neighbors for 3D gradient
            continue
            
        vertex_pos = vertices[vertex_idx]
        vertex_value = values[vertex_idx]
        
        # Set up least squares problem: Ax = b
        # where A contains relative positions, x is gradient, b contains value differences
        A = []
        b = []
        weights = []
        
        for neighbor_idx in neighbors:
            neighbor_pos = vertices[neighbor_idx]
            neighbor_value = values[neighbor_idx]
            
            # Relative position vector
            rel_pos = neighbor_pos - vertex_pos
            distance = np.linalg.norm(rel_pos)
            
            if distance > 1e-12:
                # Weight inversely proportional to distance
                weight = 1.0 / distance
                
                A.append(rel_pos)
                b.append(neighbor_value - vertex_value)
                weights.append(weight)
        
        if len(A) >= 3:
            A = np.array(A)
            b = np.array(b)
            weights = np.array(weights)
            
            # Apply weights
            W = np.diag(weights)
            A_weighted = W @ A
            b_weighted = W @ b
            
            # Solve weighted least squares: (A^T W A) x = A^T W b
            try:
                gradient = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)[0]
                gradients[vertex_idx] = gradient
            except np.linalg.LinAlgError:
                pass  # Keep zero gradient if solve fails
    
    return gradients


import numpy as np
from scipy.spatial import cKDTree

def select_non_interacting_vertices(vertices, radius=3.0, decay=1.5, randomize=True):
    """
    Selects the maximum number of vertices from a mesh such that
    no two selected vertices have significant interaction.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Coordinates of mesh vertices.
    radius : float, optional (default=3.0)
        Base interaction radius.
    decay : float, optional (default=1.5)
        Distance over which interaction decays to near zero.

    Returns
    -------
    selected : list[int]
        Indices of selected vertices.
    """
    vertices = np.asarray(vertices)
    tree = cKDTree(vertices)

    # Define cutoff distance where interaction is considered "significant"
    cutoff = radius + decay  

    N = len(vertices)
    selected = []
    excluded = np.zeros(N, dtype=bool)

    inds = [n for n in range(N)]
    if randomize:
        shuffle(inds)
        
    for i in inds:
        if excluded[i]:
            continue

        # Accept this vertex
        selected.append(i)

        # Exclude all neighbors within cutoff distance
        neighbors = tree.query_ball_point(vertices[i], cutoff)
        excluded[neighbors] = True

    return selected


def estimate_radius_decay(atoms_list, rmax=None):
    """
    Estimate interaction radius and decay based on xy-projection
    of atomic positions across a list of Atoms.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        List of ASE Atoms objects.
    rmax : float or None
        Optional cutoff for maximum pair distance to consider (default: no cutoff).

    Returns
    -------
    radius : float
        Estimated radius (interaction stays constant inside).
    decay : float
        Estimated decay length (interaction falls off).
    """
    xy_coords = []
    for atoms in atoms_list:
        pos = atoms.get_positions()[:, :2]  # project to xy-plane
        xy_coords.append(pos)
    xy_coords = np.vstack(xy_coords)

    # pairwise distances in 2D
    diff = xy_coords[:, None, :] - xy_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))

    # take upper triangle (avoid zeros/self-distances)
    dists = dists[np.triu_indices(len(xy_coords), k=1)]

    if rmax is not None:
        dists = dists[dists <= rmax]

    if len(dists) == 0:
        raise ValueError("No valid pairwise distances found.")

    # estimate: short-range cutoff and spread
    radius = np.percentile(dists, 5)   # "core" distance
    decay = np.percentile(dists, 95) - radius

    return float(radius), float(decay+1.)

def compute_vertex_areas(vertices, faces):
    """
    Compute per-vertex surface area contribution for a mesh with arbitrary polygonal faces.

    Parameters
    ----------
    vertices : (N, 3) array
        Vertex coordinates.
    faces : list of lists or array-like
        Each element is a list of vertex indices forming a face (any length >= 3).

    Returns
    -------
    vertex_area : (N,) array
        Surface area contribution for each vertex.
    """
    vertices = np.asarray(vertices, dtype=float)
    n_vertices = len(vertices)
    vertex_area = np.zeros(n_vertices)

    for f in faces:
        f = np.asarray(f, dtype=int)
        if len(f) < 3:
            continue  # skip degenerate faces

        # fan triangulation: pick vertex 0, form triangles (0, i, i+1)
        face_area = 0.0
        v0 = vertices[f[0]]

        for i in range(1, len(f) - 1):
            v1, v2 = vertices[f[i]], vertices[f[i + 1]]
            tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            face_area += tri_area

        # distribute area equally among the vertices of this face
        vertex_area[f] += face_area / len(f)

    return vertex_area


def fit_reaction_plane(images, fixed_indices=None):
    """
    Fit a plane to moving atoms across NEB images.

    Parameters
    ----------
    images : list of ase.Atoms
        List of NEB images.
    fixed_indices : list of int, optional
        Atom indices to ignore (fixed atoms). Default: None.

    Returns
    -------
    plane_func : function
        Function z = f(x, y) giving plane height for given (x, y).
    normal : np.array
        Normal vector of the plane.
    point : np.array
        Point on the plane (centroid of moving atoms).
    """

    # collect positions of moving atoms
    all_pos = []
    for img in images:
        pos = img.get_positions()
        if fixed_indices is not None:
            mask = np.ones(len(pos), dtype=bool)
            mask[fixed_indices] = False
            pos = pos[mask]
        all_pos.append(pos)
    
    all_pos = np.vstack(all_pos)  # shape (N_total, 3)

    # compute centroid
    centroid = np.mean(all_pos, axis=0)

    # subtract centroid
    X = all_pos - centroid

    # SVD to find normal (smallest singular value)
    _, _, vh = np.linalg.svd(X)
    normal = vh[-1, :]  # normal vector of best-fit plane

    # plane equation: (r - centroid) . normal = 0
    # return a function z = f(x, y)
    def plane_func(x, y):
        a, b, c = normal
        if abs(c) < 1e-8:
            raise ValueError("Plane normal nearly vertical, cannot express z as function of x,y")
        return (-a*(x - centroid[0]) - b*(y - centroid[1])) / c + centroid[2]

    return plane_func, normal, centroid


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)

    # choose a vector not parallel to the normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(normal, e1)
    return e1, e2

def project_to_plane(points, centroid, e1, e2):
    rel = points - centroid
    u = rel @ e1
    v = rel @ e2
    return u, v


def generate_plane_mesh(images, resolution=0.1, gutter=2.0):
    """
    Generate a square quad mesh lying in the best-fit reaction plane.

    Parameters
    ----------
    images : list of ase.Atoms
        NEB images
    resolution : float
        Grid spacing in plane coordinates
    gutter : float
        Padding added around projected atom bounds

    Returns
    -------
    verts : (N, 3) ndarray
        Vertex positions
    edges : list of (i, j)
        Edge index pairs
    faces : list of [i0, i1, i2, i3]
        Quad faces
    normals : (N, 3) ndarray
        Per-vertex normals
    """

    # ---- collect fixed atom indices (FixAtoms only) ----
    fixed_indices = set()
    for img in images:
        for c in img.constraints or []:
            if isinstance(c, FixAtoms):
                fixed_indices.update(c.index)

    fixed_indices = list(fixed_indices)

    # ---- fit reaction plane ----
    _, normal, centroid = fit_reaction_plane(
        images, fixed_indices=fixed_indices
    )
    normal = normal / np.linalg.norm(normal)

    # ---- collect moving atom positions ----
    moving_pos = []
    for img in images:
        pos = img.get_positions()
        if fixed_indices:
            mask = np.ones(len(pos), dtype=bool)
            mask[fixed_indices] = False
            pos = pos[mask]
        moving_pos.append(pos)
    moving_pos = np.vstack(moving_pos)

    # ---- build plane basis ----
    e1, e2 = plane_basis(normal)

    # ---- project atoms into plane coordinates ----
    rel = moving_pos - centroid
    u = rel @ e1
    v = rel @ e2

    umin, umax = u.min() - gutter, u.max() + gutter
    vmin, vmax = v.min() - gutter, v.max() + gutter

    # ---- generate grid ----
    us = np.arange(umin, umax + resolution, resolution)
    vs = np.arange(vmin, vmax + resolution, resolution)

    Nu = len(us)
    Nv = len(vs)

    # ---- generate vertices ----
    verts = np.empty((Nu * Nv, 3))
    for i, uu in enumerate(us):
        for j, vv in enumerate(vs):
            idx = i * Nv + j
            verts[idx] = centroid + uu * e1 + vv * e2

    # ---- generate quad faces & edges ----
    faces = []
    edges = set()

    def vid(i, j):
        return i * Nv + j

    for i in range(Nu - 1):
        for j in range(Nv - 1):
            v0 = vid(i,     j)
            v1 = vid(i,     j + 1)
            v2 = vid(i + 1, j)
            v3 = vid(i + 1, j + 1)

            faces.append([v0, v1, v3, v2])

            edges.update({
                (v0, v1),
                (v1, v3),
                (v3, v2),
                (v2, v0),
            })

    edges = list(edges)

    # ---- per-vertex normals ----
    normals = np.tile(normal, (len(verts), 1))

    return verts, edges, faces, normals


def get_mesh_islands(points, faces, selected_indices):
    """
    Extract disconnected islands from a mesh subset.
    
    Parameters:
    -----------
    points : np.ndarray, shape (N, 3)
        Vertex positions
    faces : np.ndarray, shape (M, 4)
        Quad faces as vertex indices
    selected_indices : np.ndarray
        Indices of vertices to keep
    
    Returns:
    --------
    islands : list of dict
        Each dict contains {'points': ndarray, 'faces': ndarray, 'original_indices': ndarray}
    """
    # Get faces that use only selected vertices
    selected_set = set(selected_indices)
    valid_faces = []
    
    for face in faces:
        if all(v in selected_set for v in face):
            valid_faces.append(face)
    
    if len(valid_faces) == 0:
        return []
    
    valid_faces = np.array(valid_faces)
    
    # Build adjacency graph (vertices connected by edges)
    n_points = len(points)
    edges = []
    
    for face in valid_faces:
        # For quads, edges are: (0,1), (1,2), (2,3), (3,0)
        for i in range(4):
            v1, v2 = face[i], face[(i+1) % 4]
            edges.append([v1, v2])
            edges.append([v2, v1])  # undirected
    
    edges = np.array(edges)
    
    # Create sparse adjacency matrix
    data = np.ones(len(edges), dtype=int)
    adjacency = csr_matrix((data, (edges[:, 0], edges[:, 1])), 
                           shape=(n_points, n_points))
    
    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)
    
    # Extract each island
    islands = []
    
    for island_id in range(n_components):
        # Get vertices in this island
        island_verts = np.where(labels == island_id)[0]
        
        # Filter to only selected vertices
        island_verts = np.array([v for v in island_verts if v in selected_set])
        
        if len(island_verts) == 0:
            continue
        
        # Create mapping from old to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(island_verts)}
        
        # Get faces for this island
        island_faces = []
        for face in valid_faces:
            if all(v in old_to_new for v in face):
                new_face = [old_to_new[v] for v in face]
                island_faces.append(new_face)
        
        if len(island_faces) == 0:
            continue
        
        island_faces = np.array(island_faces)
        island_points = points[island_verts]
        
        islands.append({
            'points': island_points,
            'faces': island_faces,
            'original_indices': island_verts
        })
    
    return islands

from collections import defaultdict

def get_manifold_minima(faces, vals, tol=1e-03):
    adj = [set() for _ in vals]
    for f in faces:
        for i in f:
            adj[i] |= set(f) - {i}

    return [
        i for i in range(len(vals))
        if all(vals[i] < vals[j] - tol for j in adj[i])
    ]


# ---------------------------------------------------------------------------
# Linear scaling relation utilities
# ---------------------------------------------------------------------------

def fit_line_and_distances(x, y):
    """
    Fit a line y = m*x + b and return the signed perpendicular distance of
    each point from the fitted line.

    Parameters
    ----------
    x, y : array-like
        Input data arrays.

    Returns
    -------
    distances : np.ndarray
        Signed perpendicular distances from the fitted line.
        Positive values are above the line, negative below.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m, b = np.polyfit(x, y, 1)
    distances = (m * x - y + b) / np.sqrt(m ** 2 + 1)
    return distances


# ---------------------------------------------------------------------------
# ASE atom-selection utilities
# ---------------------------------------------------------------------------

def slice_atoms_near_point(atoms: Atoms, point: np.ndarray, d: float) -> Atoms:
    """
    Return a subset of *atoms* whose positions lie within distance *d* of *point*.

    Parameters
    ----------
    atoms : ase.Atoms
    point : array-like, shape (3,)
    d : float
        Distance cutoff in Å.

    Returns
    -------
    ase.Atoms
    """
    positions = atoms.get_positions()
    distances = np.linalg.norm(positions - np.asarray(point), axis=1)
    return atoms[distances < d]


# ---------------------------------------------------------------------------
# Geometric projection / rotation utilities
# ---------------------------------------------------------------------------

def furthest_projected_pairs(p0, normal, points, tol=0.1):
    """
    Find pairs of point indices that are furthest apart when projected into
    the plane orthogonal to *normal* at *p0*.

    Returns
    -------
    list of (i, j) index pairs
    """
    p0 = np.asarray(p0)
    n = np.asarray(normal, dtype=float)
    pts = np.asarray(points)

    n /= np.linalg.norm(n)

    v = pts - p0
    v_plane = v - np.outer(v @ n, n)

    _, _, vh = np.linalg.svd(v_plane, full_matrices=False)
    direction = vh[0]
    proj = v_plane @ direction

    pmin, pmax = proj.min(), proj.max()
    imin = np.where(np.abs(proj - pmin) < tol)[0]
    imax = np.where(np.abs(proj - pmax) < tol)[0]

    return list(itertools_product(imin, imax))


def points_close_to_rotating_line(p0, normal, points, angles, tol=1e-3):
    """
    For a line rotating around *normal* at anchor *p0*, return for each angle
    the indices of *points* close to the line, split by side (+/-).

    Parameters
    ----------
    p0 : array-like, shape (3,)
    normal : array-like, shape (3,)
    points : array-like, shape (N, 3)
    angles : sequence of float
        Rotation angles in radians.
    tol : float
        Distance tolerance.

    Returns
    -------
    list of (pos_indices, neg_indices) per angle
    """
    p0 = np.asarray(p0)
    n = np.asarray(normal, dtype=float)
    pts = np.asarray(points)

    n /= np.linalg.norm(n)

    tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    results = []
    for theta in angles:
        d = np.cos(theta) * u + np.sin(theta) * v
        d_perp = np.cross(n, d)
        rel = pts - p0
        signed_dist = rel @ d_perp
        close = np.abs(signed_dist) < tol
        pos = np.where((signed_dist > 0) & close)[0]
        neg = np.where((signed_dist < 0) & close)[0]
        results.append((pos, neg))

    return results


# ---------------------------------------------------------------------------
# Mesh curvature
# ---------------------------------------------------------------------------

def _normalize_vec(v):
    """Normalize a vector; returns zero vector if norm is near zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _compute_incident_faces(faces):
    """Build vertex → incident face list mapping."""
    inc = defaultdict(list)
    for fi, f in enumerate(faces):
        for vert in f:
            inc[vert].append(fi)
    return inc


def _signed_dihedral_at_vertex(vertices, f1, f2, n1, n2, v):
    """Signed dihedral angle between two quad faces sharing vertex *v*."""
    shared = list(set(f1) & set(f2))
    if len(shared) < 2:
        return 0.0
    u = shared[1] if shared[0] == v else shared[0]
    e = _normalize_vec(vertices[u] - vertices[v])
    sign_val = np.dot(n1, np.cross(n2, e))
    dot_val = np.clip(np.dot(n1, n2), -1, 1)
    return np.arccos(dot_val) * np.sign(sign_val)


def curvature_deformed_cube(vertices, faces, normals):
    """
    Approximate vertex-level signed curvature on a quad mesh by summing
    signed dihedral angles around each vertex.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
    faces : array-like, each face is a sequence of 4 vertex indices
    normals : array-like, shape (N_faces, 3)
        Per-face normals.

    Returns
    -------
    curvature_signed : np.ndarray, shape (N,)
        Positive → convex, negative → concave.
    curvature_magnitude : np.ndarray, shape (N,)
        Absolute curvature.
    """
    vertices = np.array(vertices)
    inc = _compute_incident_faces(faces)

    curvature_signed = np.zeros(len(vertices))
    curvature_magnitude = np.zeros(len(vertices))

    for v in range(len(vertices)):
        fids = inc[v]
        if len(fids) != 4:
            continue
        total = 0.0
        for i in range(4):
            total += _signed_dihedral_at_vertex(
                vertices,
                faces[fids[i]], faces[fids[(i + 1) % 4]],
                normals[fids[i]], normals[fids[(i + 1) % 4]],
                v,
            )
        curvature_signed[v] = total
        curvature_magnitude[v] = abs(total)

    return curvature_signed, curvature_magnitude
