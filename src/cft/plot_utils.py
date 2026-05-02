"""
cft.plot_utils
~~~~~~~~~~~~~~
Reusable matplotlib helpers for CFT figures.
"""
import numpy as np


def add_linear_fits(ax, df, xcol, ycol, groupcol):
    """Overlay per-group linear fits on *ax*.

    For each group in *df* defined by *groupcol*, fits a line to
    (*xcol*, *ycol*) and plots it in faint gray.

    Parameters
    ----------
    ax : matplotlib Axes
    df : pandas.DataFrame
    xcol : str
        Column name for x values.
    ycol : str
        Column name for y values.
    groupcol : str
        Column used to split the data into groups.
    """
    for name, g in df.groupby(groupcol):
        if len(g) < 2:
            continue

        x = g[xcol].values
        y = g[ycol].values
        m, c = np.polyfit(x, y, 1)

        xfit = np.array([x.min(), x.max()])
        yfit = m * xfit + c

        ax.plot(
            xfit, yfit,
            color="#4B4B4B",
            linewidth=0.5,
            linestyle="-",
            zorder=0,
        )

def plot_metal_bep_panels(df, xcol='E_reaction', ycol='vals_PMD', metal_col='first_neighbor', 
                          palette=None, alpha=0.15, s=20, add_fits=True, xlabel=None, ylabel=None):
    """
    Create a multi-panel scatter plot (one panel per metal) showing BEP-like correlations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Reaction pathway dataframe (e.g. df_reaction_info)
    xcol : str
        Column for x-axis (e.g., reaction energy).
    ycol : str
        Column for y-axis (e.g., transition state energy or barrier).
    metal_col : str
        Column containing the anchoring metal symbol.
    palette : dict, optional
        Mapping of metal symbols to specific colors.
    alpha : float
        Transparency of scatter points.
    s : float
        Marker size.
    add_fits : bool
        If True, overlay faint linear fits for the data in each panel.
        
    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    metals = df[metal_col].dropna().unique()
    n_metals = len(metals)

    if xlabel is None:
        xlabel = f'{xcol} / eV'
    if ylabel is None:
        ylabel = f'{ycol} / eV'
    
    if n_metals == 0:
        return plt.subplots()
    
    # Calculate grid size
    n_cols = min(n_metals, 3)
    n_rows = int(np.ceil(n_metals / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols, sharex=True,
        sharey=True, figsize = [182/25.6, 120/25.6],
        dpi = 600, squeeze=False
    )
    axes = axes.flatten()
    
    if palette is None:
        # Default nice qualitative palette if none provided
        colors = sns.color_palette("muted", n_metals)
        palette = {m: colors[i] for i, m in enumerate(metals)}

    for i, metal in enumerate(metals):
        ax = axes[i]
        subset = df[df[metal_col] == metal]
        
        sns.scatterplot(
            data=subset,
            x=xcol,
            y=ycol,
            color=palette.get(metal, 'blue'),
            alpha=alpha,
            s=s,
            linewidth=0,
            ax=ax,
        )
        
        if add_fits and len(subset) > 1:
            x = subset[xcol].values
            y = subset[ycol].values
            m, c = np.polyfit(x, y, 1)
            xfit = np.array([x.min(), x.max()])
            yfit = m * xfit + c
            
            # Compute R^2 for quantitative metrics
            ss_tot = np.sum((y - np.mean(y))**2)
            ss_res = np.sum((y - (m * x + c))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            fit_label = f'Fit: $y = {m:.2f}x {"+" if c >= 0 else "-"} {abs(c):.2f}$\n$R^2 = {r2:.2f}$'
            ax.plot(xfit, yfit, color="#333333", linestyle="--", linewidth=1.5, zorder=4, label=fit_label)

        ax.set_title(f"First neighbor: {metal}")
        ax.set_xlabel(xlabel)
        if i % n_cols == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')
            
        ax.legend(loc='upper right', frameon=False, fontsize='small')
        # sns.despine(ax=ax)
        
    # Hide any unused subplots
    for j in range(n_metals, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    return fig, axes

def plot_metal_distributions(df, val_col='vals_PMD', metal_col='first_neighbor', palette=None, s=3):
    """
    Plot the energetic distributions of reaction pathways grouped strictly by their 
    primary anchoring metal. Uses a clean stripplot colored by the metal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing pathway properties.
    val_col : str
        The energetic property to evaluate (e.g., 'vals_PMD' or 'E_barrier').
    metal_col : str
        The categorical column for the anchoring metal (e.g., 'first_neighbor').
    palette : str or dict
        Mapping of metal symbols to specific colors.
    s : float
        Marker size for the individual pathway points.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    metals = df[metal_col].dropna().unique()
    n_metals = len(metals)
    
    if n_metals == 0:
        return plt.subplots()

    if palette is None:
        colors = sns.color_palette("muted", n_metals)
        palette = {m: colors[i] for i, m in enumerate(metals)}
        
    # Sort the environments by their median stability to create a visual tier-list
    order = df.groupby(metal_col)[val_col].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(6, 0.7 * n_metals + 1))
    
    # Plot the individual pathway scatter points
    sns.stripplot(
        data=df, x=val_col, y=metal_col, order=order,
        hue=metal_col, palette=palette, legend=False,
        alpha=0.6, size=s, jitter=0.25, zorder=2, ax=ax
    )
    
    # Add a bold median marker tick to guide the eye chemically
    sns.pointplot(
        data=df, x=val_col, y=metal_col, order=order,
        estimator='median', errorbar=None, color='black', 
        markers='|', markersize=20, linestyles='', zorder=3, ax=ax
    )
    
    ax.set_title('Transition State Evaluated Energies by Anchoring Element', pad=15)
    ax.set_xlabel(r'$\Delta E_{\text{many-body}}^{\text{PMD}}$ / eV')
    ax.set_ylabel('')
    
    # Clean modern aesthetics
    sns.despine(left=True)
    ax.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    return fig, ax
