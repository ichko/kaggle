from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors, rc
rc('animation', html='jshtml')

# data frame rate
FPS = 150.
# size of the arena the flies are enclosed in
ARENA_RADIUS_MM = 26.689

# hard-code indices of keypoints and skeleton edges
keypointidx = np.arange(18, dtype=int)
skeleton_edges = np.array([
    [7,  8],
    [10, 14],
    [11, 12],
    [12, 17],
    [7, 11],
    [9, 10],
    [7,  9],
    [5,  7],
    [2,  3],
    [2,  7],
    [5, 18],
    [6, 13],
    [7, 16],
    [7, 15],
    [2,  4],
    [6,  7],
    [7,  0],
    [7,  1]
])
# keypoints for computing distances between pairs of flies
fidxdist = np.array([2, 7, 8])


"""
d = get_fly_dists(x,tgt=0)
Compute the distance between fly tgt and all other flies. This is defined as the
minimum distance between any pair of the following keypoints:
'antennae','end_notum_x','end_abdomen', hard-coded with fidxdist
at middle frame data.ctrf
Input:
x: ndarray of size maxnflies x nkeypoints x 2 data sample, sequence of data for all flies
tgt: (optional) which fly to compute distances to. Default is 0.
Output:
d: Array of length nflies with the squared distance to the selected target.
"""


def get_fly_dists(x, tgt=0):
    nkpts = len(fidxdist)
    ntgts = x.shape[0]
    ndim = x.shape[2]
    d = np.min(np.sum((x[:, fidxdist, :].reshape(ntgts, 1, nkpts, ndim) -
                       x[tgt, fidxdist, :].reshape(1, nkpts, 1, ndim))**2., axis=3), axis=(1, 2))
    return d


"""
dark3cm = get_Dark3_cmap()
Returns a new matplotlib colormap based on the Dark2 colormap.
I didn't have quite enough unique colors in Dark2, so I made Dark3 which
is Dark2 followed by all Dark2 colors with the same hue and saturation but
half the brightness.
"""


def get_Dark3_cmap():
    dark2 = list(cm.get_cmap('Dark2').colors)
    dark3 = dark2.copy()
    for c in dark2:
        chsv = colors.rgb_to_hsv(c)
        chsv[2] = chsv[2]/2.
        crgb = colors.hsv_to_rgb(chsv)
        dark3.append(crgb)
    dark3cm = colors.ListedColormap(tuple(dark3))
    return dark3cm


"""
isreal = get_real_flies(x)
Returns which flies in the input ndarray x correspond to real data (are not nan).
Input:
x: ndarray of arbitrary dimensions, as long as the last two dimensions are nfeatures x 2,
and correspond to the keypoints and x,y coordinates.
"""


def get_real_flies(x):
    # x is ntgts x nfeatures x 2
    isreal = np.all(np.isnan(x), axis=(-1, -2)) == False
    return isreal


"""
fig,ax,isnewaxis = set_fig_ax(fig=None,ax=None)
Create new figure and/or axes if those are not input.
Returns the handles to those figures and axes.
isnewaxis is whether a new set of axes was created.
"""


def set_fig_ax(fig=None, ax=None):
    isnewaxis = True
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    else:
        isnewaxis = False
    return fig, ax, isnewaxis


"""
hkpt,hedge,fig,ax = plot_fly(pose=None, 
                             fig=None, ax=None, kptcolors=None, color=None, name=None,
                             plotskel=True, plotkpts=True, hedge=None, hkpt=None)
Visualize the single fly position specified by pose
Inputs:
pose: Required. nfeatures x 2 ndarray.
kptidx: Optional. 1-dimensional array specifying which keypoints to plot. If None, 
uses keypointidx. Default: None.
skelidx: Optional. nedges x 2 ndarray specifying which keypoints to connect with edges. 
If None, uses skeleton_edges. Default: None.
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
Default: None
color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
name: Optional. String defining an identifying label for these plots. Default None.
plotskel: Optional. Whether to plot skeleton edges. Default: True.
plotkpts: Optional. Whether to plot key points. Default: True.
hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
"""


def plot_fly(pose=None, kptidx=None, skelidx=None, fig=None, ax=None, kptcolors=None, color=None, name=None,
             plotskel=True, plotkpts=True, hedge=None, hkpt=None):
    # plot_fly(x,fig=None,ax=None,kptcolors=None):
    # x is nfeatures x 2
    assert(pose is not None)
    if kptidx is None:
        kptidx = keypointidx
    if skelidx is None:
        skelidx = skeleton_edges

    isnewaxis = False
    if ((hedge is None) and plotskel) or ((hkpt is None) and plotkpts):
        fig, ax, isnewaxis = set_fig_ax(fig=fig, ax=ax)
    isreal = get_real_flies(pose)

    if plotkpts:
        if isreal:
            xc = pose[kptidx, 0]
            yc = pose[kptidx, 1]
        else:
            xc = []
            yc = []
        if hkpt is None:
            if kptcolors is None:
                kptcolors = 'hsv'
            if (type(kptcolors) == list or type(kptcolors) == np.ndarray) and len(kptcolors) == 3:
                kptname = 'keypoints'
                if name is not None:
                    kptname = name + ' ' + kptname
                hkpt = ax.plot(xc, yc, '.', color=kptcolors,
                               label=kptname, zorder=10)[0]
            else:
                if type(kptcolors) == str:
                    kptcolors = plt.get_cmap(kptcolors)
                hkpt = ax.scatter(xc, yc, c=np.arange(
                    len(kptidx)), marker='.', cmap=kptcolors, zorder=10)
        else:
            if type(hkpt) == matplotlib.lines.Line2D:
                hkpt.set_data(xc, yc)
            else:
                hkpt.set_offsets(np.column_stack((xc, yc)))

    if plotskel:
        nedges = skelidx.shape[0]
        if isreal:
            xc = np.concatenate(
                (pose[skelidx, 0], np.zeros((nedges, 1))+np.nan), axis=1)
            yc = np.concatenate(
                (pose[skelidx, 1], np.zeros((nedges, 1))+np.nan), axis=1)
        else:
            xc = np.array([])
            yc = np.array([])
        if hedge is None:
            edgename = 'skeleton'
            if name is not None:
                edgename = name + ' ' + edgename
            if color is None:
                color = [.6, .6, .6]
            hedge = ax.plot(xc.flatten(), yc.flatten(), '-',
                            color=color, label=edgename, zorder=0)[0]
        else:
            hedge.set_data(xc.flatten(), yc.flatten())

    if isnewaxis:
        ax.axis('equal')

    return hkpt, hedge, fig, ax


"""
hkpt,hedge,fig,ax = plot_flies(poses=None, kptidx=None, skelidx=None,
                               colors=None,kptcolors=None,hedges=None,hkpts=None,
                               **kwargs)
Visualize all flies for a single frame specified by poses.
Inputs:
poses: Required. nflies x nfeatures x 2 ndarray.
colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
colormap I defined in get_Dark3_cmap(). Default: None.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
Default: None
hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
Default: None.
Extra arguments: All other arguments will be passed directly to plot_fly.
"""


def plot_flies(poses=None, fig=None, ax=None, colors=None, kptcolors=None, hedges=None, hkpts=None, **kwargs):

    if hedges is None or hkpts is None:
        fig, ax, isnewaxis = set_fig_ax(fig=fig, ax=ax)
    else:
        isnewaxis = False
    if colors is None:
        colors = get_Dark3_cmap()
    if kptcolors is None:
        kptcolors = [0, 0, 0]
    nflies = poses.shape[0]
    if not (type(colors) == list or type(kptcolors) == np.ndarray):
        if type(colors) == str:
            cmap = cm.get_cmap(colors)
        else:
            cmap = colors
        colors = cmap(np.linspace(0., 1., nflies))

    if hedges is None:
        hedges = [None, ]*nflies
    if hkpts is None:
        hkpts = [None, ]*nflies

    for fly in range(nflies):
        hkpts[fly], hedges[fly], fig, ax = plot_fly(poses[fly, ...], fig=fig, ax=ax, color=colors[fly, ...],
                                                    kptcolors=kptcolors, hedge=hedges[fly], hkpt=hkpts[fly], **kwargs)
    if isnewaxis:
        ax.axis('equal')

    return hkpts, hedges, fig, ax


"""
animate_pose_sequence(seq=None, kptidx=None, skelidx=None,
                      start_frame=0,stop_frame=None,skip=1,
                      fig=None,ax=None,savefile=None,
                      **kwargs)
Visualize all flies for the input sequence of frames seq.
Inputs:
seq: Required. seql x nflies x nfeatures x 2 ndarray.
start_frame: Which frame of the sequence to start plotting at. Default: 0.
stop_frame: Which frame of the sequence to end plotting on. Default: None. If None, the
sequence length (seq.shape[0]) is used.
skip: How many frames to skip between plotting. Default: 1.
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
savefile: Optional. Name of video file to save animation to. If None, animation is displayed
instead of saved.
Extra arguments: All other arguments will be passed directly to plot_flies.
"""


def animate_pose_sequence(seq=None, start_frame=0, stop_frame=None, skip=1,
                          fig=None, ax=None,
                          annotation_sequence=None,
                          savefile=None,
                          **kwargs):

    if stop_frame is None:
        stop_frame = seq.shape[0]
    fig, ax, isnewaxis = set_fig_ax(fig=fig, ax=ax)

    isreal = get_real_flies(seq)
    idxreal = np.where(np.any(isreal, axis=0))[0]
    seq = seq[:, idxreal, ...]

    # plot the arena wall
    theta = np.linspace(0, 2*np.pi, 360)
    ax.plot(ARENA_RADIUS_MM*np.cos(theta),
            ARENA_RADIUS_MM*np.sin(theta), 'k-', zorder=-10)
    minv = -ARENA_RADIUS_MM*1.01
    maxv = ARENA_RADIUS_MM*1.01

    # first frame
    f = start_frame
    h = {}
    h['kpts'], h['edges'], fig, ax = plot_flies(
        poses=seq[f, ...], fig=fig, ax=ax, **kwargs)
    h['frame'] = plt.text(-ARENA_RADIUS_MM*.99, ARENA_RADIUS_MM*.99, 'Frame %d (%.2f s)' % (f, float(f)/FPS),
                          horizontalalignment='left', verticalalignment='top')
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout(pad=0)
    # ax.margins(0)

    def update(f):
        h['kpts'], h['edges'], fig, ax = plot_flies(poses=seq[f, ...],
                                                    hedges=h['edges'], hkpts=h['kpts'], **kwargs)
        h['frame'].set_text('Frame %d (%.2f s)' % (f, float(f)/FPS))
        return h['edges']+h['kpts']

    ani = animation.FuncAnimation(fig, update, frames=np.arange(
        start_frame, stop_frame, skip, dtype=int))
    if savefile is not None:
        print('Saving animation to file %s...' % savefile)
        writer = animation.PillowWriter(fps=30)
        ani.save(savefile, writer=writer)
        print('Finished writing.')
    else:
        pass
    return ani
