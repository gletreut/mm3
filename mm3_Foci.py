#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import inspect
<<<<<<< HEAD
import getopt
import numpy as np
=======
import argparse
import yaml
>>>>>>> master
try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as sio

# user modules
# realpath() will make your script run, even if you symlink it
cmd_folder = os.path.realpath(os.path.abspath(
                              os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# This makes python look for modules in ./external_lib
cmd_subfolder = os.path.realpath(os.path.abspath(
                                 os.path.join(os.path.split(inspect.getfile(
                                 inspect.currentframe()))[0], "external_lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import mm3_helpers as mm3
import mm3_plots as mm3_plots

# when using this script as a function and not as a library the following will execute
if __name__ == "__main__":
    '''
    This script adds foci location information onto an existing dictionary of cells.
    '''

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_Foci.py',
                                     description='Finds foci.')
    parser.add_argument('-f', '--paramfile', type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-c', '--cellfile', type=file,
                        required=False, help='Path to Cell object dicionary to analyze. Defaults to complete_cells.pkl.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile.name:
        param_file_path = namespace.paramfile.name
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'
    p = mm3.init_mm3_helpers(param_file_path) # initialized the helper library

    # load cell file
    mm3.information('Loading cell data.')
    if namespace.cellfile:
        cell_file_path = namespace.cellfile.name
    else:
        mm3.warning('No cell file specified. Using complete_cells.pkl.')
        cell_file_path = os.path.join(p['cell_dir'], 'complete_cells.pkl')

    with open(cell_file_path, 'r') as cell_file:
        Cells = pickle.load(cell_file)

    # load specs file
    try:
        with open(os.path.join(p['ana_dir'],'specs.yaml'), 'r') as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        mm3.warning('Could not load specs file.')
        raise ValueError

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    ### foci analysis
    mm3.information("Starting foci analysis.")

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = mm3_plots.organize_cells_by_channel(Cells, specs)

    # time table
    time_table_path = os.path.join(p['ana_dir'], 'time_table.pkl')
    with open(time_table_path, 'r') as fin:
        time_table = pickle.load(fin)
    times_all = np.array(np.sort(time_table[fov_id].keys()), np.int_)
    t0 = times_all[0] # first time index
    tN = times_all[-1] # last time index
    print ("t0 = {:d}    tN = {:d}".format(t0,tN))

    # range of time
    if (p['image_start'] < 1):
        p['image_start']=1
    p['image_end'] = np.min([p['image_end'],np.max([cell.division_time for cell in Cells.values()])])
    if p['image_end'] < p['image_start']:
        p['image_end'] = p['image_start']
    extraction_range = range(p['image_start'], p['image_end']+1)

    # for each set of cells in one fov/peak, find the foci
    #for fov_id in [10]:
    for fov_id in fov_id_list:
        if not fov_id in Cells_by_peak:
            continue

        for peak_id, Cells_of_peak in Cells_by_peak[fov_id].items():
            # test
            # print ('Peak no',peak_id)
            # print ('Cells_of_peak')
            # print (Cells_of_peak)
            if (len(Cells_of_peak) == 0):
                continue
            if p['debug_foci']:
                image_data_seg = mm3.load_stack(fov_id, peak_id, color='seg')
                image_data_FL = mm3.load_stack(fov_id, peak_id, color='sub_{}'.format(p['foci_plane']))
                for t in extraction_range:
                    print ("t = {:d}".format(t))
                    mm3.foci_lap_peak(image_data_seg[t-t0], image_data_FL[t-t0], fov_id=fov_id, peak_id=peak_id, t=t)
            else:
                mm3.foci_analysis(fov_id, peak_id, Cells_of_peak)

            # test
            # sys.exit()

    # Output data to both dictionary and the .mat format used by the GUI
    fileout = os.path.join(p['cell_dir'], os.path.splitext(cell_filename)[0] + '_foci.pkl')
    with open(fileout, 'wb') as cell_file:
        mm3.information("Writing file: {}".format(fileout))
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    for cell in Cells.values():
        for attr in vars(cell).keys():
            if (getattr(cell,attr) is None):
                setattr(cell,attr,[])
    fileout = os.path.join(p['cell_dir'], os.path.splitext(cell_filename)[0] + '_foci.mat')
    with open(fileout, 'wb') as cell_file:
        mm3.information("Writing file: {}".format(fileout))
        sio.savemat(cell_file, Cells)

    mm3.information("Finished foci analysis.")
