# SOLaCE: Semantic-Level, Object and Language-Guided Coherence Evaluation
This repository contains code and the dataset of the SOLaCE framework for evaluating visual concepts (ICLR 2022 OSC Workshop).

This code allows to run the framework for different datasets and to reproduce the results in our paper. It also contains the exact dataset of 2600 human assessments of visual concepts (in anonymized form) that was used in our user study.

Tested with ``detectron=0.4.1, torch=1.7.1, python=3.7.10, cuda=11.0``

## Files
As a starting point, we recommend looking at the notebooks ``ReadAndAnnotate.ipynb`` (how to load and annotate the concepts) and ``AnalysisUserStudy.ipynb`` (how to use our dataset, how to run the coherence assessors).

Here is a short overview over all the files in the repo and their purpose:
* ``AnalysisUserStudy.ipynb``: Reproduction of Results of our user study, demo of the coherence assessors.
* ``ReadAndAnnotate.ipynb``: An example on how to read and write our dataset of 260 concepts that was used in the user study. It also contains the code that we used to obtain the meanings for these concept, that were shown to the users in the study.
* ``gui_user_study.py``: Run this script to launch the GUI that was shown to the participants of our user study. This requires PyQt5.
* ``DetectronFRCNN.ipynb``: A tutorial-style notebook for the VG and the Detectron FRCNN -> Relies on train_faster_rcnn.py -> Relies on VGPreprocess.ipynb
* ``VGPreprocess.ipynb``: Preprocessing of the Visual Genome dataset. Run this notebook before using any models that rely on FRCNN.
* ``add_images_script.py``: Add the images to the concept files to assemble our concept dataset. See comments below.
* ``concept2word.py``: API for the Semantic Annotation Modules
* ``coherence_assessor.py``: API for the Coherence Assessment Modules
* ``concept_reader.py``: API for our Concept Classes. Read and Write functionality.
* ``data_utils.py``: API that provides Dataloaders for the different datasets etc.
* ``gui_user_study.py``: Run the exact GUI that was used in our study. The results will be locally stored in a file named annotations.json.
* ``train_faster_rcnn.py``: Train the FRCNN model. Run VGPreprocess.ipynb beforehand.
* ``AttributeROIHead.py``: An extension of the detectron2 framework that allows Faster-RCNN to predict attributes. The model that we used relies on this file, so we include it for completeness.

## Datasets: 
Due to space constraints and copyright reasons, we cannot provide the full images for our concept dataset that was used in our study in this Repo. Only the meta data is provided (including activation maps) but we provide a script to assemble the full concept files on your local machine if you have the data sets downloaded from their respective sources. The image data needs to be added to the concepts before they can be visualized and used with our methods:

* Download the dataset for Places365 from http://places2.csail.mit.edu/download.html (to assemble the concepts only validation images are required, use high resolution)
* Download the dataset for AwA2 from https://cvml.ist.ac.at/AwA2/ (only base and the JPEG images required)

To assemble the concepts used in our user study run ``python3 add_images_script.py data/concepts_wo_images <path containing places356 validation images> <path containing AwA2 images>``. This will results in the same concept files that were also used in our user study.

The annotations of the users that were used as a ground truth in this paper can be found in the folder ``user_study``. See ``AnalysisUserStudy`` for an example how to use them.
