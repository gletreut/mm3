############################################################################
## IMPORTS
############################################################################
import os,sys
import cPickle as pkl
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from scipy.stats import iqr
from scipy import signal
from copy import copy

origin = os.path.dirname(os.path.realpath(sys.argv[0]))
mm3dir = os.path.dirname(origin)
sys.path.append(mm3dir)
import mm3_helpers as mm3

# global settings
plt.rcParams['axes.linewidth']=0.5

############################################################################
## FUNCTIONS
############################################################################
def histogram(X,density=True):
    valmax = np.max(X)
    valmin = np.min(X)
    iqrval = iqr(X)
    nbins_fd = (valmax-valmin)*np.float_(len(X))**(1./3)/(2.*iqrval)
    if (nbins_fd < 1.0e4):
        return np.histogram(X,bins='auto',density=density)
    else:
        return np.histogram(X,bins='sturges',density=density)

def get_binned(X,Y,edges):

    nbins = len(edges)-1
    digitized = np.digitize(X,edges)
    Y_binned=[ None for n in range(nbins)]
    for i in range(1,nbins+1):
        Y_binned[i-1] = np.array(Y[digitized == i])

    return Y_binned

def make_binning(x,y,bincount_min, method=np.mean, emethod=np.std):
    """
    Given a graph (x,y), return a graph (x_binned, y_binned) which is a binned version
    of the input graph, along x. Standard deviations per bin for the y direction are also returned.
    """
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    histx, bins = histogram(x)
    digitized = np.digitize(x,bins)
    x_binned=[]
    y_binned=[]
    error_binned=[]
    for i in range(1,len(bins)):
        ypts = y[digitized == i]

        if (len(ypts) < bincount_min):
            continue

        x_binned.append(float(0.5*(bins[i-1] + bins[i])))
        y_binned.append(float(method(ypts)))
        error_binned.append(float(emethod(ypts)))

    res = {}
    res['x'] = np.array(x_binned)
    res['y'] = np.array(y_binned)
    res['err'] = np.array(error_binned)
    return res

def plot_attribute_time(data, fileout, attrdict, attr, time_mode='birth', scatter_max_pts=1000, lw=0.5, ms=2, xformat='{x:.2g}', yformat='{x:.2g}', offsets=None, labels=None, colors=None, bin_width=None, tlo=None, thi=None, aratio=4./3, units_dx=None, units_dy=None, units_dn=None, ylims=None):
    """
    Plot overlay of multiple datasets
    """

    # preliminary checks
    ndata = len(data)
    if ndata == 0:
        sys.exit("Empty data sets!")

    ## offsets
    if offsets is None:
        offsets = [None]*ndata

    if len(offsets) != ndata:
        sys.exit("offsets must have same length as data sets!")

    if labels is None:
        labels = [None]*ndata

    if len(labels) != ndata:
        sys.exit("labels must have same length as data sets!")

    if colors is None:
        colors = [None]*ndata

    if len(colors) != ndata:
        sys.exit("colors must have same length as data sets!")

    # check if attr in attrdict
    if not (attr in attrdict):
        sys.exit("Attr: {:s} not found in attrdict".format(attr))

    ## attributes for birth and division times
    #attr_times = "times"   # frames
    attr_times = "times_min"

    # build the vectors to plot
    Tb_list=[]   # lists of times
    Td_list=[]   # lists of times
    Tm_list=[]   # lists of times
    Y_list=[]   # lists of attribute values
    tmin_old,tmax_old, dt_old = None, None, None
    for n in range(ndata):
        T_birth = []
        T_division = []
        Y = []
        dt = 9.99e99
        cells = data[n]
        for key in cells.keys():
            cell = cells[key]
            try:
                times = np.array(getattr(cell,attr_times), dtype=np.float_)
                tb = times[0]
                td = times[-1]
                u = np.min(np.diff(times))
                if u < dt and u > 0.:
                    dt = u

                y = np.float_(getattr(cell,attr))
                if np.isfinite(y):
                    Y.append(y)
                    T_birth.append(tb)
                    T_division.append(td)
            except (ValueError,AttributeError):
                continue
        # end loop cells

        ## offsets
        offset = offsets[n]
        print offsets, offset
        if offset is None:
            try:
                offset = tmax_old + dt_old  # just an estimate because in reality should be the last time acquisition
                                            # including cells that have been filtered out
            except TypeError:
                offset = 0
        T_birth = np.array(T_birth, dtype=np.float_) + offset
        T_division = np.array(T_division, dtype=np.float_) + offset

        ## prints
        tmin, tmax = np.nanmin(T_birth), np.nanmax(T_division)
        print "dataset {:d}    tmin = {:.1f}    tmax = {:.1f}    dt = {:.1f}    offset = {:.1f}".format(n, tmin, tmax, dt, offset)


        ## rescale
        try:
            scale = attrdict[attr]['scale']
        except (KeyError):
            scale = None
        if not scale is None:
            Y = np.array(Y)*scale

        try:
            scale = attrdict[attr_times]['scale']
        except (KeyError):
            scale = None
        if not scale is None:
            T_birth = T_birth*scale
            T_division = T_division*scale

        ## update
        Tb_list.append(T_birth)
        Td_list.append(T_division)
        Tm_list.append(0.5*(T_birth+T_division))
        Y_list.append(np.array(Y, dtype=np.float_))
        tmin_old = tmin
        tmax_old = tmax
        dt_old = dt

    # end loop datasets

    # time mode
    if time_mode == 'birth':
        T_list = Tb_list
    elif time_mode == 'division':
        T_list = Td_list
    else:
        T_list = Tm_list

    # make the binning
    if tlo is None:
        tlo = np.min(np.concatenate(Tb_list))
    if thi is None:
        thi = np.max(np.concatenate(Td_list))

    ## build bins
    if bin_width is None:
        hist, bins = histogram(np.concatenate(T_list))
        bin_width = np.min(np.diff(bins))

    print "bin_width = {:.1f}".format(bin_width)
    tlo = int(tlo/bin_width)*bin_width
    thi = int(thi/bin_width+0.5)*bin_width
    nbins = int((thi-tlo)/bin_width)        # t_i = t_0 (t_lo) + n \Delta
    thi = tlo + nbins*bin_width             # t_N = t_0 + N \Delta
    edges = np.arange(nbins+1, dtype=np.float_)*bin_width + tlo
    Xbinned = 0.5*(edges[:-1]+edges[1:])

    ## make subsets
    Ydata_binned_list = []
    for n in range(ndata):
        X = T_list[n]
        Y = Y_list[n]
        Ydata_binned = get_binned(X,Y,edges)  # matrix where rows are bin index and columns y-values within
        Ydata_binned_list.append(Ydata_binned)

    # make the plot
    ## general plot parameters
    mm3.information("aratio = {:.2f}".format(aratio))
    ax_height = 3
    ax_width = aratio*ax_height
    figsize = (ax_width, 2*ax_height)
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    gs = gridspec.GridSpec(5,1)
    axes =[]
    ax = fig.add_subplot(gs[0:4,0])
    axes.append(ax)
    ax = fig.add_subplot(gs[4,0], sharex=axes[0])
    axes.append(ax)

    ## plot per dataset
    Ns = np.zeros(nbins, dtype=np.int_)
    haslabel=False
    for n in range(ndata):
        ### general parameters
        label = labels[n]
        color = colors[n]
        if not label is None:
            haslabel=True

        ### dataset
        T = T_list[n]
        Y = Y_list[n]
        Ydata_binned = Ydata_binned_list[n]
        Ybinned_mean = np.array([np.nanmean(Ydata) for Ydata in Ydata_binned])
        Ybinned_median = np.array([np.nanmedian(Ydata) for Ydata in Ydata_binned])
        Nbinned = np.array([float(len(Ydata)) for Ydata in Ydata_binned])
        Ybinned_std = np.array([np.nanstd(Ydata) for Ydata in Ydata_binned])
        idx = np.isfinite(Ybinned_std) & (Nbinned > 0)
        Ybinned_err = np.copy(Ybinned_std)
        Ybinned_err[idx] = Ybinned_err[idx] / np.sqrt(Nbinned[idx])
        Ns += np.int_(Nbinned)

        ### plot all cells
        axes[0].plot(T,Y,'o', ms=ms, mfc=color, mec='none', alpha=0.7)

        ### plot mean/median
        #axes[0].boxplot(Ydata_binned, positions=Xbinned, showfliers=False, manage_xticks=False, widths=0.33*bin_width)
        axes[0].plot(Xbinned, Ybinned_median, 's', ms=4*ms, mfc='none', mec=color)
        axes[0].errorbar(Xbinned, Ybinned_mean, yerr=Ybinned_err, linestyle='-', marker='o', ms=4*ms, c=color, ecolor=color, elinewidth=lw, lw=lw, label=label)

    ## plot number of cells per bin
    axes[1].bar(edges[:-1], Ns, bin_width, color='lightgray', edgecolor='k', align='edge')

    ## legend
    if haslabel:
        axes[0].legend(loc='best', frameon=False, fontsize='medium')

    ## axes adjustments
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize='medium', length=4)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].tick_params(bottom=False, labelbottom=False)
    axes[0].spines['left'].set_smart_bounds(True)
    axes[1].spines['left'].set_smart_bounds(True)
    axes[1].spines['bottom'].set_smart_bounds(True)

    if not (units_dx is None):
        axes[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx))

    if not (units_dy is None):
        axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dy))

    if not (units_dn is None):
        axes[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dn))

    axes[0].set_xlim(tlo, thi)
    axes[1].set_ylim(0., None)

    try:
        axes[0].set_ylabel(attrdict[attr]['label'], fontsize='medium')
    except KeyError:
        axes[0].set_ylabel(attrY, fontsize='medium')
    axes[1].set_ylabel("# cells", fontsize='medium')
    axes[1].set_xlabel(attrdict[attr_times]['label'], fontsize='medium')

    if not ylims is None:
        axes[0].set_ylim(ylims)


    ## last configurations
    rect = [0.,0.,1.,1.]
    gs.tight_layout(fig, rect=rect, pad=0.5)
    fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=300)
    mm3.information("{:<20s}{:<s}".format('fileout',fileout))

    return


############################################################################
## MAIN
############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plots of single-cell measurements (overlay).")
    parser.add_argument('pklfile', type=file, nargs='+', help='Pickle files containing the cell dictionary.')
    parser.add_argument('-f', '--paramfile',  type=file, required=True, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, default='.', help='The output directory for figures.')
#    parser.add_argument('--distributions',  action='store_true', help='Plot the distributions of cell variables.')
#    parser.add_argument('--crosscorrelations',  action='store_true', help='Plot the cross-correlation of cell variables.')
#    parser.add_argument('--autocorrelations',  action='store_true', help='Plot the autocorrelation of cell variables.')
    parser.add_argument('--attribute_time',  action='store_true', help='Plot the scatter plots specified in the Yaml file.')
#    parser.add_argument('-l', '--lineagesfile',  type=file, help='Pickle file containing the list of lineages.')

    # load arguments
    namespace = parser.parse_args(sys.argv[1:])
    ## paramfiles
    paramfile = namespace.paramfile.name
    params = yaml.load(namespace.paramfile)

    ## outputdirectory
    outputdir = namespace.outputdir
    outputdir = os.path.relpath(outputdir, os.getcwd())
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
        mm3.information("Making output directory: {:s}".format(outputdir))

    ## pkl files
    ndata = len(namespace.pklfile)
    mm3.information("ndata = {:d}".format(ndata))

    data = []
    for n in range(ndata):
        cells = pkl.load(namespace.pklfile[n])
        data.append(cells)

    # plots
    ## attribute vs time plots
    if namespace.attribute_time:
        mm3.information ('Plotting attribute/time plots.')
        plotdir = os.path.join(outputdir,'attribute_time')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
            mm3.information("Making directory: {:s}".format(plotdir))
        try:
            attrdict = params['plot_attribute_time_args']['attributes']
            filedict = params['plot_attribute_time_args']['plots']
            colors = params['plot_attribute_time_args']['colors']
            labels = params['plot_attribute_time_args']['labels']
            offsets = params['plot_attribute_time_args']['offsets']
        except KeyError:
            sys.exit("Missing options for scatter plots!")

        for filename in filedict.keys():
            print filename
            fileout = os.path.join(plotdir,filename)
            plot_attribute_time(data, fileout=fileout, attrdict=attrdict, colors=colors, labels=labels, offsets=offsets, **params['plot_attribute_time_args']['plots'][filename])
# lineages
    # exit
    sys.exit()

    ###END###
