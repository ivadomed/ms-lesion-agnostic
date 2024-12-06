"""
This file prints the fsleyes command to visualize the files in the exclude.yml
"""
import os
from pathlib import Path
import yaml
import argparse

path_exclude = "/home/plbenveniste/net/ms-lesion-agnostic/build_msd/ms-lesion-agnostic/dataset_aggregation/lesion_outside_sc.yml"

# load the exclude.yml
with open(path_exclude, 'r') as stream:
    exclude = yaml.safe_load(stream)


for elem in exclude["Lesions outside the SC"]:
    image = elem.replace("derivatives/labels/", "").replace("_lesion-manual", "").replace("_desc-rater3_label-lesion_seg", "").replace("_desc-rater1_label-lesion_seg","")
    print(f"fsleyes {image} {elem} &")
