import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import numpy as np


def make_grid(X1, X2=None, grid_resolution=0.01):
    """Define a grid on which to evaluate uncertainty.

    X1 : (n_examples, n_features)
    X2 : (n_examples, n_features)

    X1 (and X2) define(s) the bounds of the grid.
    """
    assert X1.shape[1] == 2, 'Cannot make grid if the number of features is not two.'

    if X2 is not None:
        X = np.concatenate((X1, X2))
    else:
        X = X1

    # Grid on which to evaluate model. The grid is based on the extent of the data
    lb = np.floor(X.min(0))
    ub = np.ceil(X.max(0)) + grid_resolution
    grid = np.mgrid[tuple(slice(i, j, grid_resolution) for i,j in zip(lb, ub))]

    # Some keyword-arguments to the plotting function
    dg = grid_resolution/2
    extent = (lb[0]-dg, ub[0]-dg, lb[1]-dg, ub[1]-dg)
    imshow_kwargs = dict(origin='lower', extent=extent)

    return grid, imshow_kwargs


def plot_grid_dw(uncertainty, similarity, beta, data, grid, imshow_kwargs):
    n1, n2 = grid.shape[1:]

    plot_dict = {
        "Uncertainty": {
            'data': uncertainty.reshape(n1, n2).T,
            'point to mark': 'max'
        },
        "Similarity": {
            'data': similarity.reshape(n1, n2).T,
            'point to mark': 'max'
        },
        "Density Weighted": {
            "data": (uncertainty * similarity**beta).reshape(n1,n2).T,
            "point to mark": 'max'
        }
    }

    fig = plt.figure(figsize=(15, 5))
    imgrid = ImageGrid(fig, 111,
                    nrows_ncols=(1,3),
                    axes_pad=(0.15, 0.35),
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    )
    for ax, name in zip(imgrid, plot_dict):
        pd = plot_dict[name]
        im = ax.imshow(pd['data'], **imshow_kwargs)
        if name == 'Similarity':
            ax.contour(*grid, pd['data'].T, levels=10, colors='k', alpha=0.5)
        ax.cax.colorbar(im)
        ax.scatter(*data['train']['X'].T, c=data['train']['y'], cmap='jet')
        g = ax.scatter(*data['pool']['X'].T, c='k', s=10, marker='x')
        title = name
        # Mark max/min
        if pd['point to mark'] is not None:
            if pd['point to mark'] == 'max':
                point = np.where(pd['data'] == pd['data'].max())
            elif pd['point to mark'] == 'min':
                point = np.where(pd['data'] == pd['data'].min())
            point = grid[:, point[1], point[0]].squeeze()
            ax.scatter(*point, marker='P', color='k', s=80)
            title += f" ({pd['point to mark']} : {point})"
        ax.set_title(title)

def plot_grid(probabilities, least_confident, margin, entropy, data, grid, imshow_kwargs):
    """Plot the probabilities of each class from the logistic regression classifier
    (numpy array with shape [n_points, n_classes, 3]) along with the result of
    applying each sampling strategy (numpy arrays with shape [n_points, n_classes]).

    (Some additional variables are needed for making the plots.)
    """
    n1, n2 = grid.shape[1:]

    # This is just to be able to plot the two class problem as RGB
    # (ignore the blue channel)
    n_classes = probabilities.shape[-1]
    if n_classes == 2:
        probabilities = np.column_stack((probabilities, np.zeros(len(probabilities))))

    plot_dict = {
        'Probabilities': {
            'data': probabilities.reshape(n1, n2, 3).transpose(1,0,2),
            'point to mark': None
        },
        'Least Confident': {
            'data': least_confident.reshape(n1, n2).T,
            'point to mark': 'max'
        },
        'Margin': {
            'data': margin.reshape(n1, n2).T,
            'point to mark': 'max'
        },
        'Entropy': {
            'data': entropy.reshape(n1, n2).T,
            'point to mark': 'max'
        }
    }
    # Make the plot
    # (please ignore the colormap of the 'Probabilities' plot. It should not be
    # there!)
    fig = plt.figure(figsize=(12, 12))
    imgrid = ImageGrid(fig, 111,
                    nrows_ncols=(2,2),
                    axes_pad=(0.15, 0.35),
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    )
    for ax, name in zip(imgrid, plot_dict):
        pd = plot_dict[name]
        im = ax.imshow(pd['data'], **imshow_kwargs)
        ax.cax.colorbar(im)
        ax.scatter(*data['train']['X'].T, c=data['train']['y'], cmap='jet')
        g = ax.scatter(*data['pool']['X'].T, c=data['pool']['y'], cmap='jet')
        g.set_facecolors('none')
        title = name
        # Mark max/min
        if pd['point to mark'] is not None:
            if pd['point to mark'] == 'max':
                point = np.where(pd['data'] == pd['data'].max())
            elif pd['point to mark'] == 'min':
                point = np.where(pd['data'] == pd['data'].min())
            point = grid[:, point[1], point[0]].squeeze()
            ax.scatter(*point, marker='P', color='k', s=80)
            title += f" ({pd['point to mark']} : {point})"
        ax.set_title(title)

def plot_pool(data, idx, uncertainty=None, features=None):
    """Plot the true labels for training data and the data in the pool.
    Also plot the uncertainty estimate of the data in the pool (and mark
    the selected point) together with the training set.

    This function should be called *after* the next point to sample has been
    identified but *before* the training set and pool has actually been
    updated!

    data : the data dictionary (as formatted by prepare_data)
    idx : index into the pool set
    uncertainty : uncertainty for each point in the pool set
    features : list of the features to plot, e.g., [1, 3]
    """
    if features is None:
        features = [1, 3]
    assert len(features) == 2

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 5))

    ax = axes[0]
    ax.scatter(*data['train']['X'][:, features].T, c=data['train']['y'])
    g = ax.scatter(*data['pool']['X'][:, features].T, c=data['pool']['y'])
    g.set_facecolors('none')
    ax.legend(['Training data', 'Data in pool'])
    ax.set_title('True label of data')

    ax = axes[1]
    ax.scatter(*data['train']['X'][:, features].T)
    g = ax.scatter(*data['pool']['X'][:, features].T, marker='.', c=uncertainty)
    ax.scatter(*data['pool']['X'][idx, features], marker='P', color='m', s=80)
    ax.legend(['Training set', 'Pool', 'Point to sample'])
    ax.set_title('Uncertainty Estimate of Pool')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(g, cax=cax)
    fig.tight_layout()

