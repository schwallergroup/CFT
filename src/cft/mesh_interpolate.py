from collections import defaultdict
import numpy as np


def subdivide_ply_smooth_color(infile, outfile, levels=1):
    V, C, F = read_ply_quad_mesh(infile)

    # subdivide
    V2, C2, F2 = subdivide_catmull_clark(V, C, F, levels=levels)

    # write
    write_ply_quad_mesh(outfile, V2, C2, F2)

def catmull_clark_step(verts, colors, faces):
    
    V = np.asarray(verts)
    C = np.asarray(colors)

    # --- adjacency ---
    face_points = []
    face_color = []
    face_of_edge = defaultdict(list)
    faces_of_vert = defaultdict(list)
    edges_of_vert = defaultdict(set)

    # --- face points ---
    for fi, f in enumerate(faces):
        face_points.append(V[f].mean(axis=0))
        face_color.append(C[f].mean(axis=0))

        for i in range(4):
            v0, v1 = f[i], f[(i+1) % 4]
            key = tuple(sorted((v0, v1)))
            face_of_edge[key].append(fi)
            edges_of_vert[v0].add(key)
            faces_of_vert[v0].append(fi)

    face_points = np.array(face_points)
    face_color = np.array(face_color)

    # --- edge points ---
    edge_point = {}
    edge_color = {}

    for (i, j), fs in face_of_edge.items():
        if len(fs) == 2:
            fp = (V[i] + V[j] +
                  face_points[fs[0]] +
                  face_points[fs[1]]) / 4
            fc = (C[i] + C[j] +
                  face_color[fs[0]] +
                  face_color[fs[1]]) / 4
        else:  # boundary
            fp = 0.5 * (V[i] + V[j])
            fc = 0.5 * (C[i] + C[j])

        edge_point[(i, j)] = fp
        edge_color[(i, j)] = fc

    # --- new vertex positions ---
    new_verts = []
    new_colors = []

    for i in range(len(V)):
        n = len(faces_of_vert[i])
        F = face_points[faces_of_vert[i]].mean(axis=0)
        C_F = face_color[faces_of_vert[i]].mean(axis=0)

        E = np.array([
            edge_point[e] for e in edges_of_vert[i]
        ]).mean(axis=0)
        C_E = np.array([
            edge_color[e] for e in edges_of_vert[i]
        ]).mean(axis=0)

        v_new = (F + 2 * E + (n - 3) * V[i]) / n
        c_new = (C_F + 2 * C_E + (n - 3) * C[i]) / n

        new_verts.append(v_new)
        new_colors.append(c_new)

    # --- assemble final mesh ---
    Vout = new_verts[:]
    Cout = new_colors[:]
    edge_idx = {}
    face_idx = []

    def add(v, c):
        idx = len(Vout)
        Vout.append(v)
        Cout.append(c)
        return idx

    for (i, j), p in edge_point.items():
        edge_idx[(i, j)] = add(p, edge_color[(i, j)])

    for p, c in zip(face_points, face_color):
        face_idx.append(add(p, c))

    new_faces = []
    for fi, f in enumerate(faces):
        i0, i1, i2, i3 = f
        fpi = face_idx[fi]

        e01 = edge_idx[tuple(sorted((i0, i1)))]
        e12 = edge_idx[tuple(sorted((i1, i2)))]
        e23 = edge_idx[tuple(sorted((i2, i3)))]
        e30 = edge_idx[tuple(sorted((i3, i0)))]

        new_faces += [
            [i0, e01, fpi, e30],
            [e01, i1, e12, fpi],
            [fpi, e12, i2, e23],
            [e30, fpi, e23, i3],
        ]

    return np.array(Vout), np.array(Cout), new_faces


def subdivide_catmull_clark(
    verts, colors, faces, levels=1
):
    for _ in range(levels):
        verts, colors, faces = catmull_clark_step(
            verts, colors, faces
        )
    return verts, colors, faces

from plyfile import PlyElement, PlyData

def write_ply_quad_mesh(filename, verts, colors, faces, binary=False):
    verts = np.asarray(verts)
    colors = np.clip(np.asarray(colors), 0, 1)

    vertex_data = np.empty(len(verts),
        dtype=[
            ('x','f4'), ('y','f4'), ('z','f4'),
            ('red','u1'), ('green','u1'), ('blue','u1')
        ]
    )

    vertex_data['x'] = verts[:,0]
    vertex_data['y'] = verts[:,1]
    vertex_data['z'] = verts[:,2]
    vertex_data['red']   = (colors[:,0] * 255).astype(np.uint8)
    vertex_data['green'] = (colors[:,1] * 255).astype(np.uint8)
    vertex_data['blue']  = (colors[:,2] * 255).astype(np.uint8)

    face_data = np.array(
        [(tuple(f),) for f in faces],
        dtype=[('vertex_indices', 'i4', (4,))]
    )

    ply = PlyData(
        [
            PlyElement.describe(vertex_data, 'vertex'),
            PlyElement.describe(face_data, 'face')
        ],
        text=not binary
    )

    ply.write(filename)


import numpy as np
from plyfile import PlyData

def read_ply_quad_mesh(filename):
    ply = PlyData.read(filename)

    v = ply['vertex'].data
    verts = np.column_stack([v['x'], v['y'], v['z']]).astype(float)

    if {'red','green','blue'}.issubset(v.dtype.names):
        colors = np.column_stack([v['red'], v['green'], v['blue']]).astype(float) / 255.0
    else:
        colors = np.ones_like(verts)

    faces = []
    for f in ply['face'].data:
        idx = f['vertex_indices']
        if len(idx) != 4:
            raise ValueError("Non-quad face found — Catmull–Clark requires quads")
        faces.append(list(idx))

    return verts, colors, faces
