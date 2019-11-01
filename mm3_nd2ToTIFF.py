#!/usr/bin/python
from __future__ import print_function

# import modules
import sys
import os
import time
import inspect
import argparse
import yaml
import glob
import math
import copy
import json
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import pims_nd2
import warnings

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

# supress the warning this always gives
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tiff

# this is the mm3 module with all the useful functions and classes
import mm3_helpers as mm3

### Main script
if __name__ == "__main__":
    '''
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point. Multiple color planes are stacked in each time point to make a multipage TIFF.
    '''

    # set switches and parameters
    parser = argparse.ArgumentParser(prog='python mm3_nd2ToTIFF.py', description='Converter .nd2 -> TIFFs.')
    parser.add_argument('nd2files',  type=str, nargs='+',
                        help='.nd2 files')
    parser.add_argument('-f', '--paramfile',  type=file,
                        required=True, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str,
                        required=False, default='.', help='Yaml file containing parameters.')
    namespace = parser.parse_args()

    # Load the project parameters file
    mm3.information('Loading experiment parameters.')
    if namespace.paramfile.name:
        param_file_path = namespace.paramfile.name
    else:
        mm3.warning('No param file specified. Using 100X template.')
        param_file_path = 'yaml_templates/params_SJ110_100X.yaml'

    # output directory
    outputdir=namespace.outputdir
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
        print("Making directory {:s}".format(outputdir))

    # init parameters
    p = mm3.init_mm3_helpers(param_file_path, experiment_directory=outputdir) # initialized the helper library

    # fovs
    fovs = None
    if ('fovs' in p):
        fovs = p['fovs']
    if (fovs is None):
        fovs = []
    user_spec_fovs = [int(val) for val in fovs]

    # number of rows of channels. Used for cropping.
    number_of_rows = p['nd2ToTIFF']['number_of_rows']

    # cropping
    try:
        vertical_crop = p['nd2ToTIFF']['crop_y']
        if len(np.array(vertical_crop).shape) == 1:
            number_of_rows = 1
        else:
            number_of_rows = 2
    except:
        sys.exit("Wrong argument for \"crop_y\"")

    # number between 0 and 9, 0 is no compression, 9 is most compression.
    tif_compress = p['nd2ToTIFF']['tiff_compress']

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(p['TIFF_dir']):
        os.makedirs(p['TIFF_dir'])

    # Load ND2 files into a list for processing
    nd2files = namespace.nd2files
    for nd2_file in nd2files:
        file_prefix = os.path.splitext(os.path.basename(nd2_file))[0]
        mm3.information('Extracting %s ...' % file_prefix)

        # load the .nd2 file object has lots of information thanks to pims
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            try:
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3.information('Starttime got from nd2 metadata.')
            except ValueError:
                # problem with the date
                jdn = mm3.julian_day_number()
                nd2f._lim_metadata_desc.dTimeStart = jdn
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3.information('Start time found from lim.')

            # get the color names out. Kinda roundabout way.
            planes = [nd2f.metadata[md]['name'] for md in nd2f.metadata if md[0:6] == u'plane_' and not md == u'plane_count']

            # this insures all colors will be saved when saving tiff
            if len(planes) > 1:
                nd2f.bundle_axes = [u'c', u'y', u'x']

            # extraction range is the time points that will be taken out. Note the indexing,
            # it is zero indexed to grab from nd2, but TIFF naming starts at 1.
            # if there is more than one FOV (len(nd2f) != 1), make sure the user input
            # last time index is before the actual time index. Ignore it.
            if (p['nd2ToTIFF']['image_start'] < 1):
                p['nd2ToTIFF']['image_start'] = 1
            if p['nd2ToTIFF']['image_end']:
                if (len(nd2f) > 0) and (len(nd2f) < p['nd2ToTIFF']['image_end']):
                    p['nd2ToTIFF']['image_end'] = len(nd2f)
            else:
                p['nd2ToTIFF']['image_end'] = len(nd2f)
            extraction_range = range(p['nd2ToTIFF']['image_start'],
                                     p['nd2ToTIFF']['image_end']+1)

            nt = nd2f.sizes['t']
            fmt_t="t{{t:0{:d}d}}".format(int(np.log10(nt))+1)
            nm = nd2f.sizes['m']
            fmt_xy="xy{{fov:0{:d}d}}".format(int(np.log10(nm))+1)
            #fname_suf = "_t%04dxy%02d.tif"
            fname_suf = "_{:s}{:s}.tif".format(fmt_t, fmt_xy)
            # loop through time points

            if len(user_spec_fovs) == 0:    # analyze all FOVs
                user_spec_fovs=range(1,nm+1)

            for t in extraction_range:
                # timepoint output name (1 indexed rather than 0 indexed)
                t_id = t - 1
                # set counter for FOV output name
                #fov = fov_naming_start

                for fov_id in range(0, nd2f.sizes[u'm']): # for every FOV
                    # fov_id is the fov index according to elements, fov is the output fov ID
                    fov = fov_id + 1

                    # skip FOVs as specified above
                    if not (fov in user_spec_fovs):
                        continue

                    # set the FOV we are working on in the nd2 file object
                    nd2f.default_coords[u'm'] = fov_id

                    # get time picture was taken
                    seconds = copy.deepcopy(nd2f[t_id].metadata['t_ms']) / 1000.
                    minutes = seconds / 60.
                    hours = minutes / 60.
                    days = hours / 24.
                    acq_time = starttime + days

                    # get physical location FOV on stage
                    x_um = nd2f[t_id].metadata['x_um']
                    y_um = nd2f[t_id].metadata['y_um']

                    # make dictionary which will be the metdata for this TIFF
                    metadata_t = { 'fov': fov,
                                   't' : t,
                                   'jd': acq_time,
                                   'x': x_um,
                                   'y': y_um,
                                   'planes': planes}
                    metadata_json = json.dumps(metadata_t)

                    # get the pixel information
                    image_data = nd2f[t_id]

                    # crop tiff if specified. Lots of flags for if there are double rows or  multiple colors
                    if vertical_crop:
                        # add extra axis to make below slicing simpler.
                        if len(image_data.shape) < 3:
                            image_data = np.expand_dims(image_data, axis=0)

                        # for just a simple crop
                        if number_of_rows == 1:

                            nc, H, W = image_data.shape
                            ylo = int(vertical_crop[0])
                            yhi = int(vertical_crop[1])
                            image_data = image_data[:, ylo:(yhi+1), :]

                            # save the tiff
                            tif_filename = file_prefix + fname_suf.format(t=t, fov=fov)
                            mm3.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                        # for dealing with two rows of channel
                        elif number_of_rows == 2:
                            # cut and save top row
                            ylo_1 = int(vertical_crop[0][0])
                            yhi_1 = int(vertical_crop[0][1])
                            ylo_2 = int(vertical_crop[1][0])
                            yhi_2 = int(vertical_crop[1][1])
                            fov_1 = 2*fov-1
                            fov_2 = 2*fov

                            image_data_one = image_data[:,ylo_1:(yhi_1+1),:]
                            metadata_t['fov']=fov_1
                            metadata_json = json.dumps(metadata_t)
                            tif_filename = file_prefix + fname_suf.format(t=t, fov=fov_1)
                            information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(TIFF_dir, tif_filename), image_data_one, description=metadata_json, compress=tif_compress, photometric='minisblack')

                            # cut and save bottom row
                            image_data_two = image_data[:,ylo_2:(yhi_2+1),:]
                            metadata_t['fov']=fov_2
                            metadata_json = json.dumps(metadata_t)
                            tif_filename = file_prefix + fname_suf.format(t=t, fov=fov_2)
                            information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(TIFF_dir, tif_filename), image_data_two, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    else: # just save the image if no cropping was done.
                        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                        mm3.information('Saving %s.' % tif_filename)
                        tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    # increase FOV counter
                    #fov += 1
