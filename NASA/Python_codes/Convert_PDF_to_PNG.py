# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pdf2image
from pdf2image import convert_from_path
import os
# conda install -c conda-forge poppler
# pip install poppler-utils # did not run. Above line worked

# %%
# curr_path = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/06_snapshot_plots_exactEval/alfalfa_grass_hay/"
# f_name = "36848_WSDA_SF_2018_46.5590659_-120.74566841_corrected.pdf"
# image = convert_from_path(curr_path + f_name, dpi=50, poppler_path=None)

# output_dir = "/Users/hn/Documents/Adobe/"
# image[0].save(f'/{pdf_path}{f_name[0:-4]}.png')

# %%
main = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/"

main_in = main + "06_snapshot_plots_exactEval/"
main_out = main + "06_snapshot_plots_exactEval_PNG/"

# %%
sub_folders = []
for dir, sub_dirs, files in os.walk(main_in):
    sub_folders.extend(sub_dirs)

for a_subdir in sub_folders:
    curr_in = main_in + a_subdir + "/"
    curr_out = main_out + a_subdir + "/"
    os.makedirs(curr_out, exist_ok=True)
    
    PDF_fiels = [x for x in os.listdir(curr_in) if x.endswith(".pdf")]
    for a_pdf_file in PDF_fiels:
        if "corrected" in a_pdf_file or "TOA" in a_pdf_file:
            image = convert_from_path(curr_in + a_pdf_file, dpi=50)
            image[0].save(f'/{curr_out}{a_pdf_file[0:-4]}.png')
        else:
            image = convert_from_path(curr_in + a_pdf_file, dpi=100)
            image[0].save(f'/{curr_out}{a_pdf_file[0:-4]}.png')

# %%

# %%
# import numpy as np
# len(np.sort(os.listdir(main_in)))

# %%
# from glob import glob
# sub_dirs = glob(main_in + "*/", recursive = True)
# sub_dirs

# %%
# print (np.sort(sub_folders))
# print (np.sort(sub_dirs))

# %% [markdown]
# # Flat

# %%
main_in = main + "06_snapshot_flat/"
main_out = main + "06_snapshot_flat_PNG/"

# %%
PDF_fiels = [x for x in os.listdir(main_in) if x.endswith(".pdf")]
for a_pdf_file in PDF_fiels:
    if "corrected" in a_pdf_file or "TOA" in a_pdf_file:
        image = convert_from_path(main_in + a_pdf_file, dpi=50)
        image[0].save(f'/{main_out}{a_pdf_file[0:-4]}.png')
    else:
        image = convert_from_path(main_in + a_pdf_file, dpi=100)
        image[0].save(f'/{main_out}{a_pdf_file[0:-4]}.png')
    

# %%

# %%

# %%
