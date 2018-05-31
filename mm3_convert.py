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

class Cell():
    '''
    The Cell class is one cell that has been born. It is not neccesarily a cell that
    has divided.
    '''

    # initialize (birth) the cell
    def __init__(self):
        """
        dummy class
        """

        self.id = None
        self.fov = None
        self.peak = None
        self.birth_label = None
        self.parent = None
        self.daughters = None
        # birth and division time
        self.birth_time = None
        self.division_time = None # filled out if cell divides
        self.times = None
        self.abs_times = None
        self.labels = None
        self.bboxes = None
        self.areas = None
        # calculating cell length and width by using Feret Diamter. These values are in pixels
        self.lengths = None
        self.widths = None

        # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
        self.volumes = None

        # angle of the fit elipsoid and centroid location
        self.orientations = None
        self.centroids = None

        # these are special datatype, as they include information from the daugthers for division
        # computed upon division
        self.times_w_div = None
        self.lengths_w_div = None
        self.widths_w_div = None

        # this information is the "production" information that
        # we want to extract at the end. Some of this is for convenience.
        # This is only filled out if a cell divides.
        self.sb = None # in um
        self.sd = None # this should be combined lengths of daughters, in um
        self.delta = None
        self.tau = None
        self.elong_rate = None
        self.septum_position = None

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

################################################
# main
################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Conversion Matlab to Pickle.")
    parser.add_argument('matfile', metavar='matlab file', type=str, help='Matlab file containing the cell dictionary.')
    namespace = parser.parse_args(sys.argv[1:])

    matfilepath = namespace.matfile
    if not os.path.isfile(matfilepath):
        sys.exit('File {} does not exist'.format(matfilepath))
    matdata = loadmat(matfilepath)['complete_cells_cell_cycle']

    cells = {}
    for key in matdata:
        cell=Cell()
        for attr in matdata[key].keys():
            value = matdata[key][attr]
            if value == []:
                value = None
            setattr(cell,attr,value)
        cells[key]=cell

    pkldata = os.path.splitext(matfilepath)[0] + '.pkl'
    with open(pkldata,'w') as fout:
        pkl.dump(cells,fout)
    print "{:<20s}{:<s}".format("fileout",pkldata)

