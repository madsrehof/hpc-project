import matplotlib.pyplot as plt
import numpy as np

def visualize_colormap(
    array: np.ndarray,
    cmap: str = "viridis",
    title: str = "2D Array Colormap",
    colorbar_label: str = "Value",
    figsize: tuple = (8, 6),
    vmin: float = None,
    vmax: float = None,
    show_grid: bool = False,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize a 2D NumPy array as a colormap heatmap.
 
    Parameters
    ----------
    array : np.ndarray
        A 2D NumPy array to visualize.
    cmap : str
        Matplotlib colormap name (default: 'viridis').
        Other options: 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', etc.
    title : str
        Title of the plot.
    colorbar_label : str
        Label for the colorbar.
    figsize : tuple
        Figure size as (width, height) in inches.
    vmin : float, optional
        Minimum value for colormap scaling. Defaults to array min.
    vmax : float, optional
        Maximum value for colormap scaling. Defaults to array max.
    show_grid : bool
        If True, draw grid lines between cells (useful for small arrays).
    save_path : str, optional
        If provided, saves the figure to this file path.
 
    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
 
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10)
    >>> fig = visualize_colormap(data, cmap="plasma", title="Random Data")
    >>> plt.show()
    """
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
 
    fig, ax = plt.subplots(figsize=figsize)
 
    im = ax.imshow(
        array,
        cmap=cmap,
        aspect="auto",
        vmin=vmin if vmin is not None else array.min(),
        vmax=vmax if vmax is not None else array.max(),
        interpolation="nearest",
    )
 
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=12)
 
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Column Index", fontsize=11)
    ax.set_ylabel("Row Index", fontsize=11)
 
    if show_grid:
        ax.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", size=0)
 
    rows, cols = array.shape
    ax.text(
        0.01, -0.07,
        f"Shape: {rows}×{cols}  |  Min: {array.min():.4g}  |  Max: {array.max():.4g}  |  Mean: {array.mean():.4g}",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )
 
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
 
    return fig