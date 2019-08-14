EXPNAME := SJ1798_MOPSgly17aa_arab02pc_001
#TIFFS := ./TIFF/$(EXPNAME)*.tif
PARAMPROCESSING := roles/params_processing.yaml
ND2 := ../data/$(EXPNAME).nd2
NPROC := 2

PARAMPOSTPROCESSING := roles/params_postprocessing.yaml
ALLCELLS := analysis/cell_data/all_cells.pkl
FILTEREDCELLS := analysis/cell_data/all_cells_processed_complete_filtered_labels1_fovpeak_continuouslineages.pkl
LINEAGESFILTEREDCELLS := analysis/cell_data/all_cells_lineages_selection_min3.pkl

# variables
TIFF := target_tiff
COMPILE := target_compile
PICKING := target_picking
PICKED := target_picked
SUBTRACT := target_subtract
SEGMENT := target_segment
POSTPROCESS := target_postprocess
PLOTS := target_plots
OBJ_COMPILE := analysis/TIFF_metadata.pkl analysis/channel_masks.pkl analysis/time_table.pkl
OBJ_PICKING := specs_picking.yaml analysis/crosscorrs.pkl
OBJ_PICKED := analysis/specs.yaml

default: $(PICKING)

##############################################################################
# IMAGE PROCESSING TARGETS
##############################################################################

# mm3_Segment.py
$(SEGMENT): $(SUBTRACT) $(PARAMPROCESSING)
	python mm3/mm3_Segment.py -f $(PARAMPROCESSING) -j $(NPROC)
	touch $(SEGMENT)

# mm3_Subtract.py
$(SUBTRACT): $(PICKED) $(PARAMPROCESSING)
	python mm3/mm3_Subtract.py -f $(PARAMPROCESSING) -j $(NPROC)
	touch $(SUBTRACT)

# mm3_ChannelPicker.py -- 2
$(PICKED): specs_picking.yaml $(PICKING) $(PARAMPROCESSING)
	python mm3/mm3_ChannelPicker.py -f $(PARAMPROCESSING) -i -c -s specs_picking.yaml -j $(NPROC)
	touch $(PICKED)

# mm3_ChannelPicker.py -- 1
$(PICKING): $(COMPILE) $(PARAMPROCESSING)
	python mm3/mm3_ChannelPicker.py -f $(PARAMPROCESSING) -i -j $(NPROC)
	cp analysis/specs.yaml specs_picking.yaml
	touch $(PICKING)

# mm3_Compile.py
$(COMPILE): $(TIFF) $(PARAMPROCESSING)
	python mm3/mm3_Compile.py -f $(PARAMPROCESSING) -j $(NPROC)
	touch $(COMPILE)

# mm3_nd2ToTIFF.py
$(TIFF): $(ND2) $(PARAMPROCESSING)
	python mm3/mm3_nd2ToTIFF.py -f $(PARAMPROCESSING) $(ND2)
	touch $(TIFF)

##############################################################################
# POSTPROCESSING TARGETS
##############################################################################

# mm3_postprocessing.py
$(POSTPROCESS): $(PARAMPOSTPROCESSING)
	python mm3/mm3_postprocessing.py -f $(PARAMPOSTPROCESSING) $(ALLCELLS)
	touch $(POSTPROCESS)

# mm3_plots_alternative.py
$(PLOTS): $(POSTPROCESS) $(PARAMPOSTPROCESSING)
	python mm3/mm3_plots_alternative.py -f $(PARAMPOSTPROCESSING) $(FILTEREDCELLS) -l $(LINEAGESFILTEREDCELLS) --distributions --crosscorrelations --scatter
	touch $(PLOTS)

##############################################################################
# UTILS
##############################################################################
# dummy targets to prevent further processing
dummy:
	touch $(TIFF)
	touch $(COMPILE)
	touch $(PICKING)
	touch $(PICKED)
	touch $(SUBTRACT)
	touch $(SEGMENT)
	touch $(POSTPROCESS)

##############################################################################
# HELP AND DOC ON MAKEFILES
##############################################################################
# https://www.gnu.org/software/make/manual/make.html
# https://makefiletutorial.com/

