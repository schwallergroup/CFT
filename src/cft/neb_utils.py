"""
cft.neb_utils
~~~~~~~~~~~~~

Utilities for NEB (Nudged Elastic Band) band fitting, plotting, and probe
construction from NEB images.
"""

from collections import namedtuple

import numpy as np
from ase.geometry import find_mic


# ---------------------------------------------------------------------------
# NEB band fitting
# ---------------------------------------------------------------------------

def fit_raw(energies, forces, positions, cell=None, pbc=None, energy_reference=0.0):
    """
    Calculate parameters for fitting NEB images to a smooth band.

    Parameters
    ----------
    energies : sequence of float
        Potential energies of each image.
    forces : sequence of array-like, shape (N_atoms, 3)
        Forces on atoms for each image.
    positions : sequence of array-like, shape (N_atoms, 3)
        Atomic positions for each image.
    cell : array-like, optional
        Unit cell (used for minimum image convention).
    pbc : array-like of bool, optional
        Periodic boundary conditions.
    energy_reference : float, optional
        Value subtracted from all energies before fitting (e.g.
        ``e_particle + isolated_ref['O'] + isolated_ref['C']``).
        Default is 0.0 (no shift).

    Returns
    -------
    ForceFit
    """
    energies = np.array(energies) - energy_reference
    n_images = len(energies)
    fit_energies = np.empty((n_images - 1) * 20 + 1)
    fit_path = np.empty((n_images - 1) * 20 + 1)

    path = [0]
    for i in range(n_images - 1):
        dR = positions[i + 1] - positions[i]
        if cell is not None and pbc is not None:
            dR, _ = find_mic(dR, cell, pbc)
        path.append(path[i] + np.sqrt((dR ** 2).sum()))

    lines = []
    lastslope = None
    for i in range(n_images):
        if i == 0:
            direction = positions[i + 1] - positions[i]
            dpath = 0.5 * path[1]
        elif i == n_images - 1:
            direction = positions[-1] - positions[-2]
            dpath = 0.5 * (path[-1] - path[-2])
        else:
            direction = positions[i + 1] - positions[i - 1]
            dpath = 0.25 * (path[i + 1] - path[i - 1])

        direction /= np.linalg.norm(direction)
        slope = -(forces[i] * direction).sum()
        x = np.linspace(path[i] - dpath, path[i] + dpath, 3)
        y = energies[i] + slope * (x - path[i])
        lines.append((x, y))

        if i > 0:
            s0 = path[i - 1]
            s1 = path[i]
            x = np.linspace(s0, s1, 20, endpoint=False)
            c = np.linalg.solve(
                np.array(
                    [
                        (1, s0, s0 ** 2, s0 ** 3),
                        (1, s1, s1 ** 2, s1 ** 3),
                        (0, 1, 2 * s0, 3 * s0 ** 2),
                        (0, 1, 2 * s1, 3 * s1 ** 2),
                    ]
                ),
                np.array([energies[i - 1], energies[i], lastslope, slope]),
            )
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            fit_path[(i - 1) * 20: i * 20] = x
            fit_energies[(i - 1) * 20: i * 20] = y

        lastslope = slope

    fit_path[-1] = path[-1]
    fit_energies[-1] = energies[-1]
    return ForceFit(path, energies, fit_path, fit_energies, lines)


class ForceFit(
    namedtuple("ForceFit", ["path", "energies", "fit_path", "fit_energies", "lines"])
):
    """Data container for NEB band fitting parameters."""

    def plot(
        self,
        ax=None,
        color_points="#ff7f0e",
        color_fit="#5F5F5F",
        color_lines="#ff7f0e",
        linewidth_fit=1.0,
        linewidth_lines=0.5,
        markersize=12,
    ):
        """
        Plot the NEB force curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        color_points : str
            Colour for raw data scatter points.
        color_fit : str
            Colour for the fitted spline.
        color_lines : str
            Colour for tangent line segments.
        linewidth_fit : float
        linewidth_lines : float
        markersize : float

        Returns
        -------
        ax
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            _, ax = plt.subplots()

        for x, y in self.lines:
            ax.plot(x, y, "-", color=color_lines, lw=linewidth_lines)

        ax.plot(
            self.fit_path, self.fit_energies, "-",
            color=color_fit, lw=linewidth_fit, label="NEB Fit",
        )

        sns.scatterplot(
            x=self.path,
            y=self.energies,
            ax=ax,
            s=markersize,
            color=color_points,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label="NEB Data",
        )

        ax.set_xlabel(r"path [Å]")
        ax.set_ylabel("energy [eV]")
        ax.legend(fontsize=8)
        ax.grid(False)
        return ax


def fit_images(images, energy_reference=0.0):
    """
    Fit a NEB image list to a smooth band.

    Parameters
    ----------
    images : list of ase.Atoms
        Each image must have a calculator attached.
    energy_reference : float, optional
        Subtracted from all image energies before fitting.

    Returns
    -------
    ForceFit
    """
    R = [atoms.positions for atoms in images]
    E = [atoms.get_potential_energy() for atoms in images]
    F = [atoms.get_forces() for atoms in images]
    A = images[0].cell
    pbc = images[0].pbc
    return fit_raw(E, F, R, A, pbc, energy_reference=energy_reference)


def plot_band(images, ax=None, energy_reference=0.0):
    """
    Plot the NEB band on *ax* (create a new figure if ax is None).

    Parameters
    ----------
    images : list of ase.Atoms
    ax : matplotlib.axes.Axes, optional
    energy_reference : float, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    forcefit = fit_images(images, energy_reference=energy_reference)
    ax = forcefit.plot(ax=ax)
    return ax.figure


# ---------------------------------------------------------------------------
# NEB probe construction
# ---------------------------------------------------------------------------

def get_neb_probe(pmd_image, keep=None, d=None):
    """
    Build an ``autoadsorbate.Fragment`` probe shaped like the CO geometry in
    a NEB transition-state image.

    The probe uses SMILES ``S1S[C][O]1`` as a scaffold; the two S atoms act
    as anchoring dummies and are trimmed according to *keep*.

    Parameters
    ----------
    pmd_image : ase.Atoms
        Atoms object containing at least a C and an O atom (the PMD image
        extracted from a NEB trajectory).
    keep : list of str, optional
        Element symbols to retain in the probe fragment.
        Default: ``['C', 'O']``.

    Returns
    -------
    autoadsorbate.Fragment
    """
    from autoadsorbate import Fragment

    if keep is None:
        keep = ["C", "O"]

    keep = list(keep) + ["S"]

    co_atoms = pmd_image[[
        atom.index for atom in pmd_image if atom.symbol in ["C", "O"]
    ]]
    
    if d is None:
        d = co_atoms.get_distance(0, 1)

    f = Fragment("S1S[C][O]1", to_initialize=1)

    f.conformers[0][0].position = [1, 0.1, 0]
    f.conformers[0][1].position = [-1, -0.1, 0]
    f.conformers[0][-2].position = [d / 2, 0, 1]
    f.conformers[0][-1].position = [-d / 2, 0, 1]

    f.conformers[0] = f.conformers[0][[
        atom.index for atom in f.conformers[0] if atom.symbol in keep
    ]]

    return f


def plot_barrier(ax, E, x=None, k=2, n=200, anchor=0.05):
    """Plot a smoothed energy barrier on *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
    E : array-like
        Energy values at each image.
    x : array-like, optional
        Reaction-coordinate positions.  Defaults to ``[0, 1, ..., len(E)-1]``.
    k : int
        Spline order (default 2).
    n : int
        Number of interpolation points (default 200).
    anchor : float
        Half-width of horizontal tick marks at each image (default 0.05).
    """
    import numpy as np
    from scipy.interpolate import make_interp_spline

    E = np.asarray(E)
    x = np.arange(len(E)) if x is None else np.asarray(x)

    xs = np.linspace(x.min(), x.max(), n)
    ys = make_interp_spline(x, E, k=k)(xs)

    ax.plot(xs, ys, lw=2)
    for xi, Ei in zip(x, E):
        ax.plot([xi - anchor, xi + anchor], [Ei, Ei], lw=3)

    ax.set_xlabel("Reaction coordinate")
    ax.set_ylabel("Energy")


def construct_trajectory(row, manifold, particle, fragment_pmd):
    """
    Build an IS → TS → FS trajectory for a single reaction pathway.

    Constructs three ``ase.Atoms`` objects representing the Initial State (C
    and O placed at their static basin vertices), Transition State (PMD placed
    at the anisotropic minimum for the given rotation angle), and Final State
    (CO placed explicitly at the PMD site with a standard 1.13 Å bond length).

    Parameters
    ----------
    row : pandas.Series
        A row from the reaction-info DataFrame produced by the site-search
        workflow.  Required keys: ``i_c``, ``i_o``, ``i_pmd``, ``angle``.
    manifold : cft.Manifold
        The surface manifold (must have ``.grid`` and ``.normals`` arrays).
    particle : ase.Atoms
        The bare particle / surface atoms (no adsorbates).
    fragment_pmd : ase.Atoms
        Pre-built PMD fragment conformer (e.g. from
        ``get_neb_probe(pmd_image).get_conformer(0)``).

    Returns
    -------
    list of ase.Atoms
        ``[state_IS, state_TS, state_FS]``
    """
    from ase import Atoms
    from autoadsorbate.Surf import attach_fragment

    # IS: C and O placed at their static basin vertices
    state_IS = particle.copy()
    c_idx, o_idx = int(row['i_c']), int(row['i_o'])
    state_IS += Atoms("C", positions=[manifold.grid[c_idx]])
    state_IS += Atoms("O", positions=[manifold.grid[o_idx]])

    # TS: PMD placed dynamically at the anisotropic minimum for this angle
    state_TS = particle.copy()
    pmd_idx = int(row['i_pmd'])
    site_dict_ts = {
        "coordinates": manifold.grid[pmd_idx],
        "n_vector": manifold.normals[pmd_idx],
    }
    attach_fragment(state_TS, site_dict_ts, fragment=fragment_pmd,
                    n_rotation=row['angle'], height=0)

    # FS: CO placed at the PMD site; standard C≡O bond length 1.13 Å
    state_FS = particle.copy()
    n_vec = manifold.normals[pmd_idx]
    pos_C = manifold.grid[pmd_idx]
    pos_O = pos_C + n_vec * 1.13
    state_FS += Atoms("CO", positions=[pos_C, pos_O])

    return [state_IS, state_TS, state_FS]


def get_PMD_structure(point, n_vector, phi, fragment):
    """Build an ASE Atoms object by attaching *fragment* at *point*.

    Parameters
    ----------
    point : array-like, shape (3,)
        Adsorption site coordinates.
    n_vector : array-like, shape (3,)
        Surface normal at the site.
    phi : float
        Rotation angle (degrees) around the normal.
    fragment : ase.Atoms
        Pre-built fragment conformer (e.g. from
        ``get_neb_probe(pmd_image).get_conformer(0)``).

    Returns
    -------
    ase.Atoms
    """
    from ase import Atoms
    from autoadsorbate.Surf import attach_fragment

    x = Atoms()
    attach_fragment(
        x,
        site_dict={"coordinates": point, "n_vector": n_vector},
        fragment=fragment,
        n_rotation=phi,
        height=0,
    )
    return x
