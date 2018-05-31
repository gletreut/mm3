import os,sys,glob
import cPickle as pkl
import argparse
import yaml
import numpy as np
import time
import shutil
import scipy.io as spio
from scipy.stats import iqr

import mm3_helpers as mm3

# yaml formats
npfloat_representer = lambda dumper,value: dumper.represent_float(float(value))
nparray_representer = lambda dumper,value: dumper.represent_list(value.tolist())
float_representer = lambda dumper,value: dumper.represent_scalar(u'tag:yaml.org,2002:float', "{:<.6g}".format(value))
yaml.add_representer(float,float_representer)
yaml.add_representer(np.float_,npfloat_representer)
yaml.add_representer(np.ndarray,nparray_representer)

################################################
# main
################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Access information on MM3 pickle file.")
    parser.add_argument('pklfile', metavar='cell pickle file', type=file, help='Pickle file containing the cell dictionary.')
    parser.add_argument('--show_attributes',  action='store_true', help='Show attributes of cells in this pickle file.')
    parser.add_argument('--show_values',  action='store_true', help='Show attributes and values of cells in this pickle file.')
    namespace = parser.parse_args(sys.argv[1:])

    cells = pkl.load(namespace.pklfile)
    fname = os.path.splitext(os.path.basename(namespace.pklfile.name))[0]
    ddir = os.path.dirname(namespace.pklfile.name)
    print "{:<20s}{:<s}".format('data dir', ddir)

    ncells = len(cells)
    print "ncells = {:d}".format(ncells)

    if (namespace.show_attributes):
        cell = cells.values()[0]
        for attr in sorted(vars(cell).keys()):
            print "{:s}".format(attr)

    if (namespace.show_values):
        cell = cells.values()[0]
        for attr in sorted(vars(cell).keys()):
            print "{:s}".format(attr)
            print getattr(cell,attr)
            print ""

