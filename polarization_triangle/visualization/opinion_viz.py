import matplotlib.pyplot as plt


import numpy as np
from matplotlib.colors import LogNorm


def draw_opinion_distribution_heatmap(history, title, filename, bins=50, log_scale=True, 
                                      cmap='viridis', vmin=None, vmax=None, custom_norm=None):
    """
    Draw 3D heatmap showing opinion distribution changes over time.

    Parameters:
    history -- Array containing all agent opinions for each time step, shape (time_steps, n_agents)
    title -- Chart title
    filename -- Save filename
    bins -- Number of bins for opinion values
    log_scale -- Whether to use logarithmic scale for colors, useful for highlighting small peaks
    cmap -- Color mapping scheme ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'hot', 'jet', etc.)
    vmin -- Minimum value for color scale, auto-determined if None
    vmax -- Maximum value for color scale, auto-determined if None
    custom_norm -- Custom color normalization object, overrides log_scale, vmin, vmax if provided
    """
    # Convert to numpy array for compatibility
    history = np.array(history)

    # Get number of time steps and agents
    time_steps, n_agents = history.shape

    # Create bins for opinion values
    opinion_bins = np.linspace(-1, 1, bins + 1)

    # Initialize heatmap data matrix
    heatmap_data = np.zeros((time_steps, bins))

    # Calculate opinion distribution for each time step
    for t in range(time_steps):
        # Calculate the number of agents in each bin
        hist, _ = np.histogram(history[t], bins=opinion_bins, range=(-1, 1))
        heatmap_data[t] = hist

    # Enlarge default font and disable Unicode minus to avoid missing characters in some fonts
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.unicode_minus': False
    })

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 12))

    # Create coordinates
    x = opinion_bins[:-1] + np.diff(opinion_bins) / 2  # opinion values (bin midpoints)
    y = np.arange(time_steps)  # time steps

    # Determine color normalization
    if custom_norm is not None:
        # Use custom normalization
        norm = custom_norm
    elif log_scale:
        # Use logarithmic scale, first replace 0 values with 1 (or the minimum non-zero value) to avoid log(0) error
        min_nonzero = np.min(heatmap_data[heatmap_data > 0]) if np.any(heatmap_data > 0) else -1
        log_data = np.copy(heatmap_data)
        log_data[log_data == 0] = min_nonzero
        
        # Set the range for logarithmic normalization
        log_vmin = vmin if vmin is not None else min_nonzero
        log_vmax = vmax if vmax is not None else np.max(log_data)
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        heatmap_data = log_data
    else:
        # Use linear scale
        linear_vmin = vmin if vmin is not None else np.min(heatmap_data)
        linear_vmax = vmax if vmax is not None else np.max(heatmap_data)
        norm = plt.Normalize(vmin=linear_vmin, vmax=linear_vmax)

    # Draw heatmap
    pcm = ax.pcolormesh(x, y, heatmap_data, norm=norm, cmap=cmap, shading='auto')

    # Add colorbar (using math format to fix display issues with 10^{-1} and minus sign)
    cbar = fig.colorbar(pcm, ax=ax)
    try:
        from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter
        from matplotlib.ticker import LogFormatter
        if custom_norm is None and log_scale:
            # cbar.formatter = LogFormatterMathtext()
            cbar.formatter = LogFormatter(base=10.0, labelOnlyBase=False)
            cbar.update_ticks()
        else:
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-2, 3))
            cbar.formatter = sf
            cbar.update_ticks()
    except Exception:
        pass
    cbar.set_label('Agent Count', fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    
    # If a specific value range is set, you can customize the colorbar ticks
    if vmin is not None and vmax is not None:
        if log_scale and not custom_norm:
            # Ticks for logarithmic scale
            cbar.set_ticks([vmin, vmin*10, vmin*100, vmax])
        else:
            # Ticks for linear scale
            step = (vmax - vmin) / 5
            cbar.set_ticks([vmin + i*step for i in range(6)])

    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=24)
    ax.set_ylabel('Time Step', fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

    # Optimize Y-axis ticks to prevent overcrowding
    max_ticks = 10
    step = max(1, time_steps // max_ticks)
    ax.set_yticks(np.arange(0, time_steps, step))

    # Save chart
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    # Additionally, create a waterfall plot version to provide a different perspective
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Select some time steps to display (to avoid overcrowding)
    step = max(1, time_steps // 20)
    selected_timesteps = np.arange(0, time_steps, step)

    # Prepare data for 3D plot
    X, Y = np.meshgrid(x, selected_timesteps)
    selected_data = heatmap_data[selected_timesteps]

    # Draw 3D surface
    surf = ax.plot_surface(X, Y, selected_data, cmap=cmap, edgecolor='none', alpha=0.8)

    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=18)
    ax.set_ylabel('Time Step', fontsize=18)
    ax.set_zlabel('Agent Count', fontsize=18)
    ax.set_title(f"{title} - 3D View", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)

    # Add colorbar
    cbar3d = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar3d.set_label('Agent Count', fontsize=16)
    cbar3d.ax.tick_params(labelsize=12)

    # Save 3D plot
    waterfall_filename = filename.replace('.png', '_3d.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename, dpi=300)
    plt.close()
