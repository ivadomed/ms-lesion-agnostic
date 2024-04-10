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

- Then a QC should be produced (prefarably using SCT) and added to a Github issue for further validation by other investigators. 

- If you are not sure of a subject, it should be flagged on Github for a more open discussion. 

## More details:
The following section details the different types of errors which occur during lesion segmentation. It is based on the condensed Nascimento Taxonomy:

<img width="522" alt="nascimento_taxonomy" src="https://github.com/ivadomed/canproco/assets/67429280/36d9e45e-4a36-40f0-a4f5-e5f3ea3f06a0">

