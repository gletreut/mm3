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
import matplotlib.patches as mpatches
from scipy.stats import iqr
from scipy import signal
from copy import copy

origin = os.path.dirname(os.path.realpath(sys.argv[0]))
mm3dir = os.path.dirname(origin)
sys.path.append(mm3dir)
import mm3_helpers as mm3

# global settings
plt.rcParams['axes.linewidth']=0.5
dpi=300

wv_list = [280, 403., 492.]

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
    """
    X and Y are vectors of scalars with the same length.
    Binning is performed along X.
    Sub-data sets of Y are returned, corresponding to Y values falling in each bin.
    """

    nbins = len(edges)-1
    digitized = np.digitize(X,edges)
    Y_binned=[ None for n in range(nbins)]
    for i in range(1,nbins+1):
        Y_binned[i-1] = np.array(Y[digitized == i])

    return Y_binned

def plot_attribute_distribution(data, fileout, attrdict, attr, lw=0.5, ms=1, labels=None, colors=None, bin_width=None, xlo=None, xhi=None, aratio=4./3, units_dx=None, xformat='{:<.2f}'):
    """
    Plot overlay of distributions from multiple datasets
    """

    # preliminary checks
    ndata = len(data)
    if ndata == 0:
        sys.exit("Empty data sets!")

    ## offsets
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

    # build the vectors
    X_list=[]   # lists of attribute values
    for n in range(ndata):
        X = []
        cells = data[n]
        for key in cells.keys():
            cell = cells[key]
            try:
                x = np.float_(getattr(cell,attr))
                if np.isfinite(x):
                    X.append(x)
            except (ValueError,AttributeError):
                continue
        # end loop cells
        ncell = len(X)
        if ncell == 0:
            # the attribute is not found in this data set
            X_list.append([])
            continue

        ## rescale
        try:
            scale = attrdict[attr]['scale']
        except (KeyError):
            scale = None
        if not scale is None:
            X = np.array(X)*scale

        ## update
        X_list.append(np.array(X, dtype=np.float_))
    # end loop datasets

    # make the binning
    if xlo is None:
        xlo = np.min(np.concatenate(X_list))
    if xhi is None:
        xhi = np.max(np.concatenate(X_list))

    ## build bins
    if bin_width is None:
        hist, bins = histogram(np.concatenate(X_list))
        bin_width = np.min(np.diff(bins))

    print "bin_width = {:.1g}".format(bin_width)
    xlo = int(xlo/bin_width)*bin_width
    xhi = int(xhi/bin_width+0.5)*bin_width
    nbins = int((xhi-xlo)/bin_width)        # x_i = x_0 (x_lo) + n \Delta
    xhi = xlo + nbins*bin_width             # x_N = x_0 + N \Delta
    edges = np.arange(nbins+1, dtype=np.float_)*bin_width + xlo
    Xbinned = 0.5*(edges[:-1]+edges[1:])

    ## make subsets
    Xdata_binned_list = []
    for n in range(ndata):
        X = X_list[n]
        if (len(X) == 0):
            Xdata_binned_list.append([[]]*nbins)
            continue
        Xdata_binned = get_binned(X,X,edges)  # matrix where rows are bin index and columns y-values within
        Xdata_binned_list.append(Xdata_binned)

    # make the plot
    ## general plot parameters
    mm3.information("aratio = {:.2f}".format(aratio))
    ax_height = 3
    ax_width = aratio*ax_height
    figsize = (ax_width, ax_height)
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    ax=fig.gca()

    ## plot per dataset
    legend_label = "{:<s}\n{:<20s}{:<20s}{:<20s}"
    for n in range(ndata):
        ### general parameters
        label = labels[n]
        color = colors[n]
        if label is None:
            label=""

        ### dataset
        X = X_list[n]
        if len(X) == 0:
            continue

        Xdata_binned = Xdata_binned_list[n]
        Nbinned = np.array([float(len(Xdata)) for Xdata in Xdata_binned])
        Ncells = np.sum(Nbinned)
        F = Nbinned / (Ncells*bin_width)

        ### label
        mu = np.nanmean(X)
        std = np.nanstd(X)
        cv = std/mu
        med = np.nanmedian(X)
        mean_label = ("mean = "+ xformat).format(mu)
        median_label = ("med = "+ xformat).format(med)
        CV_label = ("CV = {:.0f} %").format(cv*100)
        title_label = label
        label = legend_label.format(title_label, mean_label, CV_label, median_label)

        ### plot mean/median
        ax.plot(Xbinned, F, '-o', ms=ms, color=color, mew=lw, lw=lw, label=label)
        #ax.plot(Xbinned, F, '-o', ms=ms, color=color, mfc='none', mew=lw, lw=lw, label=label)

    ## legend
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize='xx-small')

    ## axes adjustments
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.tick_params(axis='both', labelsize='medium', length=4, left=False, labelleft=False)

    if not (units_dx is None):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx))

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0., None)

    try:
        ax.set_xlabel(attrdict[attr]['label'], fontsize='medium')
    except KeyError:
        axes.set_xlabel(attr, fontsize='medium')

    ## last configurations
    fig.tight_layout()
    fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
    mm3.information("{:<20s}{:<s}".format('fileout',fileout))
    plt.close('all')

    return

def plot_attribute_time(data, fileout, attrdict, attr, time_mode=None, scatter_max_pts=1000, lw=0.5, ms=2, xformat='{x:.2g}', yformat='{x:.2g}', offsets=None, labels=None, colors=None, bin_width=None, tlo=None, thi=None, aratio=4./3, units_dt=None, units_dy=None, units_dn=None, ylims=None, translate_label=None):
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
    #attr_times = "abs_time"

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
        ncells = len(Y)
        if (ncells == 0):
            Tb_list.append([])
            Td_list.append([])
            Tm_list.append([])
            Y_list.append([])
            print "dataset {:d} does not contain this attribute".format(n)
            continue

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
    if time_mode is None:
        time_mode = "middle"
    if time_mode == 'birth':
        T_list = Tb_list
    elif time_mode == 'division':
        T_list = Td_list
    else:
        time_mode = 'middle'
        T_list = Tm_list
    print "{:<20s}{:<s}".format('time_mode', time_mode)

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
        if (len(Y) == 0):
            Ydata_binned_list.append([[]]* nbins)
            continue
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
        print n
        ### general parameters
        label = labels[n]
        color = colors[n]
        if not label is None:
            haslabel=True
            if translate_label:
                label = label.translate(translate_label.upper())

        ### dataset
        T = T_list[n]
        Y = Y_list[n]
        if (len(Y) == 0):
            continue
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

    if not (units_dt is None):
        axes[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dt))

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
    fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
    mm3.information("{:<20s}{:<s}".format('fileout',fileout))
    plt.close('all')

    return

def plot_queen_distribution(data, outputdir='.', attrdict={'fl_px_med_med_405ex':{},'fl_px_med_med_488ex':{}}, attrs=['fl_px_med_med_405ex','fl_px_med_med_488ex'], backgrounds=None, lw=0.5, ms=1, labels=None, colors=None, bin_widths=None, xlos=None, xhis=None, aratio=4./4, units_dx=None, xformat='{:<.2f}'):
    """
    Plot overlay of QUEEN signal distributions

    """

    # preliminary checks
    ndata = len(data)
    if ndata == 0:
        sys.exit("Empty data sets!")

    ## other checks
    if labels is None:
        labels = [None]*ndata

    if len(labels) != ndata:
        sys.exit("labels must have same length as data sets!")

    if colors is None:
        colors = [None]*ndata

    if len(colors) != ndata:
        sys.exit("colors must have same length as data sets!")

    if bin_widths is None:
        bin_widths = [[None]*ndata for i in range(3)]
    elif np.isscalar(bin_widths):
        bin_widths = [[bin_widths]*ndata for i in range(3)]
    elif len(bin_widths) == 3:
        bin_widths_old = bin_widths
        bin_widths = []
        for i in range(3):
            bwidths = bin_widths_old[i]
            if np.isscalar(bwidths):
                bin_widths.append([bwidths]*ndata)
            elif len(bwidths) == ndata:
                bin_widths.append(bwidths)
            else:
                bin_widths.append([None]*ndata)

    else:
        raise ValueError()

    if np.array(bin_widths).shape != (3,ndata):
        raise ValueError("wrong shape for bin_widths!")

    if xlos is None:
        xlos = [None]*3

    if xhis is None:
        xhis = [None]*3

    if units_dx is None:
        units_dx = [None]*3

    # check attrs
    if len(attrs) != 2:
        sys.exit("Input attributes must be of length 2!")
    for attr in attrs:
        if not (attr in attrdict):
            sys.exit("Attr: {:s} not found in attrdict".format(attr))

    # check backgrounds
    if backgrounds is None:
        backgrounds = [0.,0.]
    if len(backgrounds) != 2:
        backgrounds = [0.,0.]
        #sys.exit("Input backgrounds must be of length 2!")

    # build the vectors
    X_list=[None for n in range(ndata)]   # lists of attribute values
    Y_list=[None for n in range(ndata)]   # lists of attribute values
    Q_list=[None for n in range(ndata)]   # lists of attribute values
    for n in range(ndata):
        X = []
        Y = []
        cells = data[n]
        for key in cells.keys():
            cell = cells[key]
            try:
                x = np.float_(getattr(cell,attrs[0]))
                if np.isfinite(x):
                    X.append(x)
            except (ValueError,AttributeError):
                pass
            try:
                y = np.float_(getattr(cell,attrs[1]))
                if np.isfinite(y):
                    Y.append(y)
            except (ValueError,AttributeError):
                pass

        # end loop cells

        ## update
        X = np.array(X, dtype=np.float_)
        Y = np.array(Y, dtype=np.float_)
        X_list[n]=X
        Y_list[n]=Y
        if len(X) != 0 and len(X) == len(Y):
            dx = X - backgrounds[0]
            idx = dx > 0.
            dy = Y - backgrounds[1]
            idx = idx & (dy > 0.)
            idx = ~idx
            Q = dx/dy
            Q[idx]=None
            if not np.any(np.isfinite):
                Q = []
        else:
            Q = []
        Q_list[n]=Q
    # end loop datasets
    V_list = [X_list, Y_list, Q_list] # n,3 shape

    ## make subsets
    ### n-lists
    Xdata_binned_list = []
    Ydata_binned_list = []
    Qdata_binned_list = []
    ### nx3-matrix
    Vbins = [ [None]*ndata for i in range(3)]   # 3,n shape
    Vdata_binned_list = [ [None]*ndata for i in range(3)]   # 3,n shape
    for i in range(3):
        if i < 2:
            attr=attrs[i]
        else:
            attr='queen'
        print "attr {:s}".format(attr)

        ### extremes
        if xlos[i] is None:
            xlos[i] = np.nanmin(np.concatenate(V_list[i]))
        if xhis[i] is None:
            xhis[i] = np.nanmax(np.concatenate(V_list[i]))

        for n in range(ndata):
            print "{:<2s}dataset {:d}".format("", n)
            V = V_list[i][n]
            if (len(V) == 0):
                print "{:<2s}empty attribute list!".format("")
                continue

            ### build bins

            bin_width = bin_widths[i][n]
            if bin_width is None:
                hist, bins = histogram(V)
                bin_widths[i][n] = bin_width = np.min(np.diff(bins))

            ### make the binning
            xlo = xlos[i]
            xhi = xhis[i]
            xlo = int(xlo/bin_width)*bin_width
            xhi = int(xhi/bin_width+0.5)*bin_width
            nbins = int((xhi-xlo)/bin_width)        # x_i = x_0 (x_lo) + n \Delta
            xhi = xlo + nbins*bin_width             # x_N = x_0 + N \Delta
            edges = np.arange(nbins+1, dtype=np.float_)*bin_width + xlo
            Vbins[i][n] = 0.5*(edges[:-1]+edges[1:])

            print "{:<4s}bin_width = {:.1g}    nbins = {:d}    xlo = {:.1g}    xhi = {:.1g}".format("",bin_width, nbins, xlo, xhi)

            ### build subsets list
            Vdata_binned_list[i][n] = get_binned(V,V,edges)  # matrix where rows are bin index and columns y-values within

    # make the plot
    print "MAKING PLOT"
    ## general plot parameters
    mm3.information("aratio = {:.2f}".format(aratio))
    ax_width = 4
    ax_height = ax_width/aratio
    figsize = (ax_width*3, ax_height)
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    gs = gridspec.GridSpec(1,3)
    axes =[]
    for i in range(3):
        ax = fig.add_subplot(gs[0,i])
        axes.append(ax)

    ## plot per dataset
    legend_label = "{:<s}{:<20s}{:<20s}{:<20s}"
    for i in range(3):
        if i < 2:
            attr=attrs[i]
        else:
            attr='queen'
        print "attr {:d}: {:s}".format(i, attr)
        ax = axes[i]
        for n in range(ndata):
            print "{:<2s}dataset {:d}".format("", n)
            ### general parameters
            label = labels[n]
            color = colors[n]
            if label is None:
                label=""

            ### dataset
            V = V_list[i][n]
            if len(V) == 0:
                continue

            Fbins = Vbins[i][n]
            Vdata_binned = Vdata_binned_list[i][n]
            Nbinned = np.array([float(len(Vdata)) for Vdata in Vdata_binned])
            Ncells = np.sum(Nbinned)
            bin_width = bin_widths[i][n]
            F = Nbinned / (Ncells*bin_width)    # frequencies
            print "{:<4s}sum(freqs) = {:.2f}".format("", np.sum(F)*bin_width)

            ### label
            mu = np.nanmean(V)
            std = np.nanstd(V)
            cv = std/mu
            med = np.nanmedian(V)
            mean_label = ("mean = "+ xformat).format(mu)
            median_label = ("med = "+ xformat).format(med)
            CV_label = ("CV = {:.0f} %").format(cv*100)
#            if (i == 0):
#                title_label = label + "\n"
#            else:
#                title_label = ""
            label = legend_label.format("", mean_label, CV_label, median_label)

            ### plot mean/median
            ax.plot(Fbins, F, '-o', ms=ms, color=color, mew=lw, lw=lw, label=label)
            #ax.plot(Xbinned, F, '-o', ms=ms, color=color, mfc='none', mew=lw, lw=lw, label=label)

        ### plot backgrounds
        if i < 2:
            bg = backgrounds[i]
            ax.axvline(x=bg, linestyle='--', lw=lw, color='k', label="bg = " + xformat.format(bg))

        ## axes adjustments
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.tick_params(axis='both', labelsize='medium', length=4, left=False, labelleft=False)

        if not (units_dx[i] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[i]))

        ax.set_xlim(xlos[i], xhis[i])
        ax.set_ylim(0., None)

        try:
            ax.set_xlabel(attrdict[attr]['label'], fontsize='medium')
        except KeyError:
            ax.set_xlabel(attr, fontsize='medium')

        ## legend
        ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.3), fontsize='xx-small', borderaxespad=0, borderpad=0)
        #ax.legend(loc='best', frameon=False, fontsize='xx-small')


    # legend
    patches=[]
    for n in range(ndata):
        label=labels[n]
        color=colors[n]
        if label is None:
            label = "{:d}".format(n)

        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    #axes[2].add_artist(plt.legend(handles=patches, loc='upper right'))
    plt.figlegend(handles=patches, fontsize='xx-small', mode='expand', ncol=ndata,borderaxespad=0, borderpad=0, loc='lower center', frameon=False)

    ## last configurations
    rect = [0.,0.,1.,1.]
    gs.tight_layout(fig, rect=rect, w_pad=0.0, h_pad=0.0)
    filename = "queen_distributions"
    for ext in ['.png', '.pdf']:
        fileout = os.path.join(outputdir, filename + ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
        mm3.information("{:<20s}{:<s}".format('fileout',fileout))
    plt.close('all')

    return

def plot_queen_time(data, offsets=None, outputdir='.', attrdict={'fl_px_med_med_405ex':{},'fl_px_med_med_488ex':{}}, attrs=['fl_px_med_med_405ex','fl_px_med_med_488ex'], backgrounds=None, time_mode=None, scatter_max_pts=1000, lw=0.5, ms=1, labels=None, colors=None, bin_width=None, xlos=None, xhis=None, tlo=None, thi=None, aratio=4./3, units_dx=None, unit_dt=None, unit_dn=None, xformat='{:<.2f}', tformat='{:<.2f}', xlims=None):
    """
    Plot overlay of QUEEN signal time traces

    """

    # preliminary checks
    ndata = len(data)
    if ndata == 0:
        sys.exit("Empty data sets!")

    ## other checks
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

    if xlos is None:
        xlos = [None]*3

    if xhis is None:
        xhis = [None]*3

    if units_dx is None:
        units_dx = [None]*3

    if xlims is None:
        xlims = [[None,None]]*3

    # check attrs
    if len(attrs) != 2:
        sys.exit("Input attributes must be of length 2!")
    for attr in attrs:
        if not (attr in attrdict):
            sys.exit("Attr: {:s} not found in attrdict".format(attr))

    # check backgrounds
    if backgrounds is None:
        backgrounds = [0.,0.]
    if len(backgrounds) != 2:
        backgrounds = [0.,0.]
        #sys.exit("Input backgrounds must be of length 2!")

    #attr_times = "abs_times"
    attr_times = "times_min"

    # build the vectors
    T_list=[[None for n in range(ndata)] for i in range(3)]
    V_list=[[None for n in range(ndata)] for i in range(3)]

    tmin_old,tmax_old, dt_old = None, None, None
    Tbis_list = [None for n in range(ndata)]
    for n in range(ndata):
        Tbis = []
        T = [[] for i in range(3)] # prepare 3 copies of time values in case attributes can be null sometimes
        V = [[] for i in range(3)] # 3 attributes
        dt = 9.99e99
        cells = data[n]
        for key in cells.keys():
            cell = cells[key]
            try:
                times = np.array(getattr(cell,attr_times), dtype=np.float_)
                tb = times[0]
                td = times[-1]
                if 'time_mode' == 'birth':
                    t = tb
                elif 'time_mode' == 'division':
                    t = td
                else:
                    t = 0.5*(tb+td)
                Tbis.append(t)
                u = np.min(np.diff(times))
                if u < dt and u > 0.:
                    dt = u
            except (ValueError,AttributeError):
                print "cell {:s}: cannot find time attribute {:s}".format(getattr(cell,'id'),attr_times)
                continue    # if problem with times, then skip this cell
            try:
                x = np.float_(getattr(cell,attrs[0]))
                if np.isfinite(x):
                    T[0].append(t)
                    V[0].append(x)
            except (ValueError,AttributeError):
                pass
            try:
                y = np.float_(getattr(cell,attrs[1]))
                if np.isfinite(y):
                    T[1].append(t)
                    V[1].append(y)
            except (ValueError,AttributeError):
                pass
        # end loop cells
        ## offsets
        offset = offsets[n]
        if offset is None:
            try:
                offset = tmax_old + dt_old  # just an estimate because in reality should be the last time acquisition
                                            # including cells that have been filtered out
            except TypeError:
                offset = 0

        Tbis = np.array(Tbis) + offset
        Tbis_list[n] = Tbis
        tmin, tmax = np.nanmin(Tbis), np.nanmax(Tbis)
        tmin_old = tmin
        tmax_old = tmax
        dt_old = dt
        print "dataset {:d}    tmin = {:.1f}    tmax = {:.1f}    dt = {:.1f}    offset = {:.1f}".format(n, tmin, tmax, dt, offset)


        ## update
        for i in range(2):
            T[i] = np.array(T[i], dtype=np.float_) + offset
            V[i] = np.array(V[i], dtype=np.float_)

        if np.all(T[0] == T[1]):
            dx = V[0] - backgrounds[0]
            idx = dx > 0.
            dy = V[1] - backgrounds[1]
            idx = idx & (dy > 0.)
            TQ = np.copy(T[0])[idx]
            Q = dx[idx]/dy[idx]
        else:
            Q = np.array([])
            TQ = np.array([])
        T[2] = TQ
        V[2] = Q

        for i in range(3):
            T_list[i][n]=T[i]
            V_list[i][n]=V[i]

    # end loop datasets

    # rescaling
    try:
        scale = attrdict[attr_times]['scale']
    except (KeyError):
        scale = None
    if not scale is None:
        Tbis_list = [T*scale for T in Tbis_list]

    for i in range(3):
        if i < 2:
            attr=attrs[i]
        else:
            attr='queen'
        print "rescaling attr {:s}".format(attr)


        ### maybe rescale
        try:
            scale = attrdict[attr_times]['scale']
        except (KeyError):
            scale = None
        if not scale is None:
            T_list[i] = [T*scale for T in T_list[i]]

        try:
            scale = attrdict[attr]['scale']
        except (KeyError):
            scale = None
        if not scale is None:
            V_list[i] = [V*scale for V in V_list[i]]

    ### extremes
    Tall = np.concatenate([np.concatenate(T_list[i]) for i in range(3)])
    if tlo is None:
        tlo = np.nanmin(Tall)
    if thi is None:
        thi = np.nanmax(Tall)

    if bin_width is None:
        hist, bins = histogram(Tall)
        bin_width = np.min(np.diff(bins))

    ### make the binning
    tlo = int(tlo/bin_width)*bin_width
    thi = int(thi/bin_width+0.5)*bin_width
    nbins = int((thi-tlo)/bin_width)        # x_i = x_0 (x_lo) + n \Delta
    thi = tlo + nbins*bin_width             # x_N = x_0 + N \Delta
    edges = np.arange(nbins+1, dtype=np.float_)*bin_width + tlo
    Tbins = 0.5*(edges[:-1]+edges[1:])
    print "bin_width = {:.1g}    nbins = {:d}    tlo = {:.1g}    thi = {:.1g}".format(bin_width, nbins, tlo, thi)


    ## make subsets
    Vdata_binned_list = [ [None]*ndata for i in range(3)]   # 3,n shape
    for i in range(3):
        if i < 2:
            attr=attrs[i]
        else:
            attr='queen'
        print "build subsets attr {:s}".format(attr)

        for n in range(ndata):
            print "{:<2s}dataset {:d}".format("", n)
            T = T_list[i][n]
            V = V_list[i][n]
            if (len(T) == 0):
                print "{:<2s}empty attribute list!".format("")
                continue

            ### build subsets list
            Vdata_binned_list[i][n] = get_binned(T,V,edges)  # matrix where rows are bin index and columns y-values within

    Tbis_all = np.concatenate(Tbis_list)
    print Tbis_all
    Tbis_binned = get_binned(Tbis_all, Tbis_all, edges)
    Ns = np.array([float(len(Tbis_set)) for Tbis_set in Tbis_binned])

    # make the plot
    print "MAKING PLOT"
    ## general plot parameters
    mm3.information("aratio = {:.2f}".format(aratio))
    ax_height = 3
    ax_width = ax_height*aratio
    figsize = (ax_width, ax_height*4)
    fig = plt.figure(num='none', facecolor='w', figsize=figsize)
    nrows=7
    gs = gridspec.GridSpec(nrows,1)
    axes =[]
    ax = fig.add_subplot(gs[0:2,0])
    axes.append(ax)
    for i in range(1,4):
        ax = fig.add_subplot(gs[2*i:min(2*(i+1),nrows),0], sharex=axes[0])
        axes.append(ax)

    ## plot per dataset
    for i in range(3):
        if i < 2:
            attr=attrs[i]
        else:
            attr='queen'
        print "plotting attr {:d}: {:s}".format(i, attr)
        ax = axes[i]
        for n in range(ndata):
            print "{:<2s}dataset {:d}".format("", n)
            ### general parameters
            #label = labels[n]
            color = colors[n]

            ### dataset
            T = T_list[i][n]
            V = V_list[i][n]
            if (len(T) == 0):
                print "{:<4s}empty attribute list!".format("")
                continue

            Vdata_binned = Vdata_binned_list[i][n]
            Vbinned_mean = np.array([np.nanmean(Vdata) for Vdata in Vdata_binned])
            Vbinned_median = np.array([np.nanmedian(Vdata) for Vdata in Vdata_binned])
            Vbinned_std = np.array([np.nanstd(Vdata) for Vdata in Vdata_binned])
            Nbinned = np.array([float(len(Vdata)) for Vdata in Vdata_binned])
            idx = np.isfinite(Vbinned_std) & (Nbinned > 0)
            Vbinned_err = np.copy(Vbinned_std)
            Vbinned_err[idx] = Vbinned_err[idx] / np.sqrt(Nbinned[idx])
            Vbinned_err[~idx] = None

            ### plot all cells
            ax.plot(T,V,'o', ms=ms, mfc=color, mec='none', alpha=0.7)

            ### plot mean/median
            ax.plot(Tbins, Vbinned_median, 's', ms=4*ms, mfc='none', mec=color)
            ax.errorbar(Tbins, Vbinned_mean, yerr=Vbinned_err, linestyle='-', marker='o', ms=4*ms, c=color, ecolor=color, elinewidth=lw, lw=lw)

            ### plot backgrounds
            if i < 2:
                bg = backgrounds[i]
                ax.axhline(y=bg, linestyle='--', lw=lw, color='k', label="bg = " + xformat.format(bg))
        # end loop on data set

        if not (units_dx[i] is None):
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[i]))

        ax.set_ylim(xlos[i], xhis[i])

        try:
            ax.set_ylabel(attrdict[attr]['label'], fontsize='medium')
        except KeyError:
            ax.set_ylabel(attr, fontsize='medium')

        ## axes adjustments
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.tick_params(axis='both', labelsize='medium', length=4, bottom=False, labelbottom=False)
    # end loop on axes

    # cell count axes
    ax=axes[-1]
    ax.bar(edges[:-1], Ns, bin_width, color='lightgray', edgecolor='k', align='edge')
    if not (unit_dt is None):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=unit_dt))

    if not (unit_dn is None):
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=unit_dn))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    ax.tick_params(axis='both', labelsize='medium', length=4)

    ax.set_xlim(tlo, thi)
    ax.set_ylim(0., None)

    ax.set_ylabel("# cells", fontsize='medium')
    ax.set_xlabel(attrdict[attr_times]['label'], fontsize='medium')


    # legend
    patches=[]
    for n in range(ndata):
        label=labels[n]
        color=colors[n]
        if label is None:
            label = "{:d}".format(n)

        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    #axes[2].add_artist(plt.legend(handles=patches, loc='upper right'))
    plt.figlegend(handles=patches, fontsize='x-small', mode='expand', ncol=ndata,borderaxespad=0, borderpad=0, loc='lower center', frameon=False)

    ## last configurations
    rect = [0.,0.,1.,1.]
    gs.tight_layout(fig, rect=rect, w_pad=0.0, h_pad=0.0)
    filename = "queen_time"
    for ext in ['.png', '.pdf']:
        fileout = os.path.join(outputdir, filename + ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
        mm3.information("{:<20s}{:<s}".format('fileout',fileout))
    plt.close('all')

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
    parser.add_argument('--show_attributes',  action='store_true', help='Show the attributes of the first cell of each dat set.')
    parser.add_argument('--attribute_time',  action='store_true', help='Plot the time-binned values.')
    parser.add_argument('--attribute_distribution',  action='store_true', help='Plot the distributions of the attributes specified in the Yaml file.')
    parser.add_argument('--queen_time',  action='store_true', help='Queen analysis specific. Plot the time-binned values.')
    parser.add_argument('--queen_distribution',  action='store_true', help='Queen analysis specific. Plot the distributions.')

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

    # show attributes
    if namespace.show_attributes:
        for n in range(ndata):
            cells = data[n]
            key = cells.keys()[0]
            cell = cells[key]
            attrs = vars(cell).keys()
            attrs.sort()
            print "-"*76
            print "dataset {:d}".format(n)
            for attr in attrs:
                print "{:<20s}".format(attr)

    # plots
    ## general
    try:
        colors = params['colors']
        labels = params['labels']
        attrdict = params['attributes']
    except KeyError:
        sys.exit("Missing options in param file!")

    ## attribute distributions
    if namespace.attribute_distribution:
        mm3.information ('Plotting attribute distributions.')
        plotdir = os.path.join(outputdir,'attribute_distribution')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
            mm3.information("Making directory: {:s}".format(plotdir))
        try:
            filedict = params['plot_attribute_distribution_args']['plots']
        except KeyError:
            sys.exit("Missing options for plot_attribute_distribution!")

        for filename in filedict.keys():
            fileout = os.path.join(plotdir,filename)
            argdict = {}
            for mydict in [params['plot_attribute_distribution_args']['plots'][filename], params['plot_attribute_distribution_args']['general']]:
                for key in mydict:
                    argdict[key]=mydict[key]
            plot_attribute_distribution(data, fileout=fileout, attrdict=attrdict, colors=colors, labels=labels, **argdict)

    ## attribute vs time plots
    if namespace.attribute_time:
        mm3.information ('Plotting attribute/time plots.')
        plotdir = os.path.join(outputdir,'attribute_time')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
            mm3.information("Making directory: {:s}".format(plotdir))
        try:
            filedict = params['plot_attribute_time_args']['plots']
            offsets = params['plot_attribute_time_args']['offsets']
        except KeyError:
            sys.exit("Missing options for plot_attribute_time!")

        for filename in filedict.keys():
            fileout = os.path.join(plotdir,filename)
            argdict = {}
            for mydict in [params['plot_attribute_time_args']['plots'][filename], params['plot_attribute_time_args']['general']]:
                for key in mydict:
                    argdict[key]=mydict[key]
            plot_attribute_time(data, fileout=fileout, attrdict=attrdict, colors=colors, labels=labels, offsets=offsets, **argdict)

    ## QUEEN analysis distributions
    if namespace.queen_distribution:
        mm3.information ('Plotting QUEEN analysis distributions.')
        plotdir = os.path.join(outputdir,'queen_analysis')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
            mm3.information("Making directory: {:s}".format(plotdir))

        argdict=params['plot_queen_distribution_args']

        plot_queen_distribution(data, outputdir=plotdir, attrdict=attrdict, colors=colors, labels=labels, **argdict)

    ## QUEEN analysis time-binned values
    if namespace.queen_time:
        mm3.information ('Plotting QUEEN analysis time plots.')
        plotdir = os.path.join(outputdir,'queen_analysis')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
            mm3.information("Making directory: {:s}".format(plotdir))

        argdict=params['plot_queen_time_args']

        plot_queen_time(data, outputdir=plotdir, attrdict=attrdict, colors=colors, labels=labels, **argdict)
    # exit

