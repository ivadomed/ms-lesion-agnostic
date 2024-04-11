# Lesion segmentation protocol:

The following details the protocol for Multiple Sclerosis (MS) lesion segmentation in the spinal cord.
Imaging the spinal cord is often essential to confirm the diagnosis of MS. That is because the lesions of the spinal cord are included in the McDonald diagnostic criteria, which considers dissemination in space and in time [(Thompson et al. 2018)](https://pubmed.ncbi.nlm.nih.gov/29275977/). While the MAGNIMS-CMSC-NAIMS working group recommends to use at least two sagittal images for MS diagnosis, still, axial imaging is mentioned as optional in international imaging guidelines [(Wattjes et al. 2021)](https://pubmed.ncbi.nlm.nih.gov/34139157/).
For detecting MS lesions in the spinal cord, two main contrasts emerge: PSIR and STIR contrasts. New studies [(Peters et al. 2024)](https://pubmed.ncbi.nlm.nih.gov/38289376/)[(Fechner et al. 2019)](https://pubmed.ncbi.nlm.nih.gov/30679225/) showed that using PSIR contrasts improved MS lesion detection in the spinal cord. [(Fechner et al. 2019)](https://pubmed.ncbi.nlm.nih.gov/30679225/) showed that the PSIR contrast showed a higher signal-to-noise (SNR) ratio compared to the STIR contrast. 

## Criteria to segment MS lesions in the spinal cord:

- Do not segment lesions in images with too many artifacts (such as this [example](https://github.com/ivadomed/canproco/issues/53#issue-1938136790)). Preferably, add the image to the exclude file so that it isn’t used for model training…
- When segmenting lesions on thick slices, always look at the above/below slices to build the volume of the lesion (this can minimize partial volume effect).
- Do not segment lesions above the first vertebrae (because here we focus only on MS lesions in the spinal cord). 
- For lesions segmentations which you are not 100% sure, flag the subject and report it for external validation of the segmentation.

## How to manually segment lesions:

- MS spinal cord lesions can be manually corrected from the prediction of a segmentation model or manually segmented from scratch. In the first case, make sure to build the json file associated with the segmentation prediction such as the following :

```json
{
        "GeneratedBy": [
            {
                "Name": "2D nnUNet model model_ms_seg_sc-lesion_regionBased.zip",
                "Version": "https://github.com/ivadomed/canproco/releases/tag/r20240125",
                "Date": "2024-01-26"
            }
        ]
    }
```

- For manual correction of the segmentation file use the manual-correction (https://github.com/spinalcordtoolbox/manual-correction) repository. The command can be inspired from this: 

```console
python manual_correction.py -path-img ~/data/canproco -config ~/config_seg.yml  -path-label ~/data/canproco/derivatives/labels  -suffix-files-lesion _lesion-manual -fsleyes-dr="-40,70"
``` 

- A Quality Control (QC) report should be produced using SCT, and added to a Github issue for further validation by other investigators. Using SCT, you can review lesion segmentation in the axial or sagittal plane :

```console
sct_qc -i {image_file} -d {lesion_seg_file} -s {sc_seg_file} -p sct_deepseg_lesion -plane {sagittal, axial} -qc {canproco_qc_folder}
```  

- If you are not sure of a subject, it should be flagged on Github for a more open discussion: here are some examples [(1)](https://github.com/ivadomed/ms-lesion-agnostic/issues/4#issuecomment-1947326493) and [(2)](https://github.com/ivadomed/ms-lesion-agnostic/issues/4#issuecomment-1947338624)

## Step 1: Get familiar with FSLeyes and SCT:
It is common practice to use FSLeyes at NeuroPoly for visual inspection of MRI images and manual segmentation of MS lesions. Therefore, naturally, the first step of the lesion segmentation process is to complete the FSLeyes tutorial ([FSLeyes documentation](https://open.win.ox.ac.uk/pages/fsl/fsleyes/fsleyes/userdoc/) and [video tutorial](https://www.youtube.com/playlist?list=PLIQIswOrUH69qFMNg8KYkEGkvCNEwlnfT)). Trainees are encouraged to learn keyboard shortcuts (ctrl+F to toggle an image, shift+↑ to scroll through volumes, ...).

Furthermore, it is recommended to get familiar with SCT for creating QCs and for manual correction ([SCT tutorial](https://spinalcordtoolbox.com/user_section/tutorials.html)). 

## More details:
The following section details the different types of errors which occur during lesion segmentation. It is based on the condensed Nascimento Taxonomy:

<img width="522" alt="nascimento_taxonomy" src="https://github.com/ivadomed/canproco/assets/67429280/36d9e45e-4a36-40f0-a4f5-e5f3ea3f06a0">

