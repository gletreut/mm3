#################### imports ####################
import os,sys
import numpy as np
from scipy import integrate
import yaml
import argparse

#################### parameter ####################
ttol=1  # tolerance on peak difference for mapping

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Compare 2 spec files.")
    parser.add_argument('specfile_in',  type=str, help='Yaml file containing peaks per FOV.')
    parser.add_argument('specfile_ref',  type=str, help='Yaml file containing peaks per FOV (reference).')
    parser.add_argument('-o',  '--specfile_out', type=str, required=False, help='Yaml file containing fixed peaks per FOV')

    # load arguments
    namespace = parser.parse_args(sys.argv[1:])
    filein=namespace.specfile_in
    fileref=namespace.specfile_ref
    fileout=namespace.specfile_out

    if not (os.path.isfile(filein) and os.path.isfile(fileref) ):
        sys.exit("Both files must exist!")

    # load the yaml files
    with open(filein,'r') as fin:
        spec_in = yaml.load(fin, Loader=yaml.SafeLoader)


    with open(fileref,'r') as fin:
        spec_ref = yaml.load(fin, Loader=yaml.SafeLoader)

    specs_out = {}

    # FOVS
    fov_in = set(spec_in)
    fov_ref = set(spec_ref)

    fov_missing = fov_ref - fov_in
    print("MISSING FOVs:\n", list(fov_missing))

    fov_new = fov_in - fov_ref
    print("NEW FOVs:\n", list(fov_new))

    spec_out = {fov: specs_in[fov] for fov in fov_new} # keep new FOVs

    # check peak differences
    fov_list = list(fov_in - fov_new)
    fov_list.sort()
    for fov in fov_list:
        spec_out[fov]={}
        text=""
        peaks_in = set(spec_in[fov])
        peaks_ref = set(spec_ref[fov])

        peaks_missing = list(peaks_ref - peaks_in)
        peaks_missing.sort()
        peaks_new = list(peaks_in - peaks_ref)
        peaks_new.sort()

        # copy unproblematic peaks
        peaks_conserved=list(peaks_in - set(peaks_new))
        peaks_conserved.sort()
        for p in peaks_conserved:
            spec_out[fov][p]=spec_ref[fov][p]

        # make peak correspondence for new peaks
        for p in peaks_new:
            p_candidates = list(range(p-ttol,p))+list(range(p+1,p+ttol+1))
            for pc in p_candidates:
                if pc in peaks_missing: # we found a match and stop
                    spec_out[fov][p]=spec_ref[fov][pc]
                    break

        # printing
        if (len(peaks_missing) != 0):
            text=text+"\n"+"{:2s}MISSING PEAKS:{:2s}".format("","")+", ".join(["{:d}".format(p) for p in peaks_missing])
        if (len(peaks_new) != 0):
            text=text+"\n"+"{:2s}NEW PEAKS:{:2s}".format("","")+", ".join(["{:d}".format(p) for p in peaks_new])

        if (text != ""):
            text="FOV {:d}".format(fov) + text + "\n"
            print(text)

    # write
    if not (fileout is None):
        with open(fileout,'w') as fout:
            yaml.dump(spec_out, fout, Dumper=yaml.SafeDumper)
        print("fileout: {:s}".format(fileout))
