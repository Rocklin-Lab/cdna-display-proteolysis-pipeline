# cdna-display-proteolysis-pipeline

## Description
- This repo includes only scripts related to a manuscript titled "Mega-scale experimental analysis of protein folding stability in biology and protein design" 
- URL: https://www.biorxiv.org/content/10.1101/2022.12.06.519132v3
- You can find the following all tables required for the pipelines above here
- URL: https://doi.org/10.5281/zenodo.7844779

## Environment and Installation
We tested the following pipeline using Python 3.8 or 3.9 on Ubuntu Ubuntu 16.04.7 LTS and Red Hat Enterprise Linux Server 7.9 (Maipo). GPU is not essential to run the pipeline but highly recommended to accelerate the process.

To install the requirements, you can use the env file:
	conda env create -f env/SE3nv.yml
	conda activate SE3nv
It usually takes ~30min to install all the requirements using Conda.

## Scripts on github
### env (.yml file for requirement installation)
	protease-pipeline.yml
### Pipeline_K50_dG (a main pipeline to generate K50/dG [folding stability] from NGS data)
	STEP1_module.ipynb
	STEP1_run.ipynb
	STEP2_run.ipynb
	STEP3_run.ipynb
	STEP4_module.ipynb
	STEP4_run.ipynb
	STEP5_module.ipynb
	STEP5_run.ipynb
	Raw_NGS_counts_overlapped_seqs_STEP1_lib1_lib2.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib2_lib3.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib1_lib4.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib2_lib4.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib3_lib4.csv
	STEP1_out_protease_concentration_trypsin
	STEP1_out_protease_concentration_chymotrypsin

Using GPU, it will take ~30min for STEP1, ~1 hr for STEP2, ~10 hrs for STEP3, ~30min for STEP4, and ~1 hr for STEP5.

### Pipeline_figure_model
	Burial_side_chain_contact_Fig3_Fig6.ipynb
	Additive_model_Fig4.ipynb
	Classification_model_Fig5.ipynb
	Data_quality_filtering_script.ipynb
	  
### Pipeline_qPCR_data: a pipeline to analyze qPCR data (related to Fig. S1)
	Raw_qPCR_data_FigS1.csv
	Process_qPCR_data.ipynb


## Files on Zenodo

### Raw_NGS_count_tables.zip
	NGS_count_lib1.csv
  	NGS_count_lib2.csv
	NGS_count_lib3.csv
  	NGS_count_lib4.csv
### K50_dG_tables.zip
	K50_dG_lib1.csv
	K50_dG_lib2.csv
	K50_dG_lib3.csv
	K50_dG_lib4.csv

### Processed_K50_dG_datasets.zip
	K50_dG_Dataset1_Dataset2.csv
	K50_Dataset3.csv
	Single_DMS_list.csv
	Double_DMS_list.csv	
	Triple_DMS_list.csv
	Heat_maps_single_DMS.pdf
	Heat_maps_double_DMS.pdf

### Data_tables_for_figs.zip
	dG_extdG_data_Fig1.csv
	dG_site_feature_Fig3.csv
	dG_for_double_mutants_Fig4.csv
	dG_non_redundant_natural_Fig5.csv
	dG_GEMME_non_redundant_natural_Fig6.csv

### Pipeline_qPCR_data.zip
	Raw_qPCR_data_FigS1.csv
	Process_qPCR_data.ipynb

### Pipeline_K50_dG.zip
	STEP1_module.ipynb
	STEP1_run.ipynb
	STEP2_run.ipynb
	STEP3_run.ipynb
	STEP4_module.ipynb
	STEP4_run.ipynb
	STEP5_module.ipynb
	STEP5_run.ipynb
	Raw_NGS_counts_overlapped_seqs_STEP1_lib1_lib2.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib2_lib3.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib1_lib4.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib2_lib4.csv
	Raw_NGS_counts_overlapped_seqs_STEP1_lib3_lib4.csv
	K50_scrambles_for_STEP3.csv
	STEP1_out_protease_concentration_trypsin
	STEP1_out_protease_concentration_chymotrypsin
	STEP3_unfolded_model_params

### Pipeline_figure_model.zip
	Burial_side_chain_contact_Fig3_Fig6.ipynb
	Additive_model_Fig4.ipynb
	Classification_model_Fig5.ipynb

### AlphaFold_model_PDBs.zip

### Blueprints_for_EEHH.zip
	eehh_EA_GBB_AGBB.bp
	eehh_GG_GBB_AGBB.bp
	eehh_XX_XXX_XXXX.bp