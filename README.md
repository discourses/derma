Associated Repositories:

* [dermatology](https://github.com/discourses/dermatology): An image data repository of the original and augmented images.
* [augmentation](https://github.com/discourses/augmentation): This repository's package creates augmentations of the original images.

Tools

* [reader](https://github.com/discourses/reader): **In progress**. The docker image of this repository will be used to run containers that download, and dearchive if necessary, data sets into a volume [for this repository, the augmentation repository, and any other] 

Associated Colab Notebook:

* [preliminary](https://colab.research.google.com/drive/1H1Afh8siQ6bsVdVaaq4qoQASQhNnoyWT): A preliminary assessment of the [raw images in dermatology](https://github.com/greyhypotheses/dermatology/tree/master/data/images) 


<br>
<br>

# Derma

**Note**, a link to a Colab interface is upcoming.  Colab offers access to GPU machines; the times per epoch are superb, hence prototyping is continuing within Colab.

* [Notes In Progress](#notes-in-progress)
  * [Brief Start Notes](#brief-start-notes)
  * [Technical Notes](#technical-notes)
* [Project](#automatic-identification-of-skin-lesion-types)
  * [Problem Statement](#problem-statement)
  * [Rationale](#rationale)
  * [The Data](#the-data)
  * [Preliminary Analysis of Metadata](https://drive.google.com/file/d/1H1Afh8siQ6bsVdVaaq4qoQASQhNnoyWT/view?usp=sharing)
  * [Copyright & Attribution](#copyright-and-attribution)

<br>
<br>

## Notes In Progress

This repository uses the wonderful continuous integration & delivery tool GitHub Actions. Hence, a variety of tests are conducted continuously.  The badges below will continuously highlight the state of each repository branch w.r.t. GitHub Action's actions.

branch|state
:---|:---
develop|![](https://github.com/greyhypotheses/derma/workflows/Derma%20Project/badge.svg?branch=develop)
master|![](https://github.com/greyhypotheses/derma/workflows/Derma%20Project/badge.svg?branch=master)
codebuild develop|![](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiZHJSbisrNDZ3NnI2Vjh3bTVkZmVXRlZvclY5Rm1UZll2cFZFcnpHUDE0bk5nbDRvVUJjbGdlSW1qVDRZN1Q0SFh6VFpUSDFWZURUcS9TTHhJTktNSmhJPSIsIml2UGFyYW1ldGVyU3BlYyI6IlFXaVNXUlFJckl5eW9sOEYiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop)

<br>

### Brief Start Notes

Sometimes the models are run in an AWS machine via docker images; [greyhypotheses @ Docker Hub](https://hub.docker.com/r/greyhypotheses/derma/tags).

An instance, i.e., container, of the image `greyhypotheses/derma:importing` serves dermatoscopic images to the deep 
learning model/s; [importing](./importing) will be replaced with [reader](https://github.com/discourses/reader)

```bash
# Import greyhypotheses/derma:importing from Docker Hub.
sudo docker pull greyhypotheses/derma:importing

# Running docker package greyhypotheses/derma:importing
sudo docker run -v ~/images:/app/images greyhypotheses/derma:importing
```

The [feature extraction deep learning model](./src/modelling/extraction) 

```bash
# Import greyhypotheses/derma:FeatureExtractionDL from Docker Hub.
sudo docker pull greyhypotheses/derma:FeatureExtractionDL

# Runs the FeatureExtractionDL model.  It requires one string argument; the string
# must be a URL oF A  YAML file of hyperparameters, e.g.,
# https://raw.githubusercontent.com/discourses/derma/develop
# /resources/hyperparameters/pattern.yml
sudo docker run -v ~/images:/app/images -v ~/checkpoints:/app/checkpoints 
    greyhypotheses/derma:FeatureExtractionDL src/main.py $1
```

<br>

### Technical Notes

* Local operating system: Windows 7
* Cloud test machine: GitHub Actions Ubuntu

Locally, the python environment was created via [`venv`](https://docs.python.org/3/library/venv.html)

```
>> python -m venv env
```

This virtual environment can be deleted via the command `rm -r env` (Cygwin).  The environment is activated via

```
>> env\Scripts\activate.bat
```

within a Windows operating system; deactivated via the command `env\Scripts\deactivate.bat`.  The command

```
>> env\Scripts\pip list
```

is used to list the set of directly & indirectly installed packages.  Always remember to upgrade pip before populating the environment

```
>> python -m pip install --upgrade pip==21.3.1
```

The [requirements](requirements.txt) document lists the directly installed packages and their versions; and a few 
indirectly installed pckages.  Thus far, the TensorFlow version used by this package/repository is TensorFlow 2.5.0
```
>> env\Scripts\pip install --upgrade tensorflow==2.7.0
```

The TensorFlow installation step installs numpy & requests, and the rest

```shell
pip install --upgrade pandas
pip install --upgrade scikit-learn
pip install --upgrade pytest coverage pytest-cov pylint flake8
pip install --upgrade PyYAML
```

The Python version is can be checked via ``python --version``.  Finally, the requirements document was/is created via

```shell
env\Scripts\pip freeze -r docs/filter.txt > requirements.txt
```

It is edited -> the packages above the line *## The following requirements were added by pip freeze:* are the directly 
installed packages.

<br>
<br>

## Automatic Identification of Skin Lesion Types
Via Dermoscopic Images of Cancerous/Pre-cancerous Skin Lesions

<br>

### Problem Statement
The World Health Organisation lists cancer as the [second leading cause of death globally](https://www.who.int/news-room/fact-sheets/detail/cancer);  the 2018 death estimate is 9.6 million.  And, early diagnosis or effective assessment  is usually critical to effective treatment and survival.  One common tool for early diagnosis, cancer precursor investigations, and/or tumour assessment is medical imaging.  For example, magnetic resonance imaging [for brain tumours](https://www.nature.com/articles/sdata2017117.pdf), [chest radiographs](https://arxiv.org/pdf/1901.07031.pdf) for investigating symptoms suggestive of lung cancer, [mammography](https://breast-cancer-research.biomedcentral.com/track/pdf/10.1186/s13058-015-0525-z) for breast cancer, etc.  A challenge, as the mammography paper illustrates, is accurate interpretation of medical images.

This project is focused on image classification for cancer diagnostics, it is specifically focused on the [International Skin Imaging Collaboration’s](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/isicArchive) dermoscopic images of skin lesions.  The aim is the

  Automatic classification of dermoscopic images according to 9 diagnostic classes: Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), Dermatofibroma, Vascular Lesion, Squamous Cell Carcinoma, Unknown

<br>
<br>

### Rationale
This project has been chosen as a precursor to applying bayesian deep learning, amongst other bayesian techniques, to diagnostic, prognostic, and pathogenetic challenges in medicine.  [Uncertainty](https://www.stat.berkeley.edu/~aldous/157/Papers/Fox_Ulkumen.pdf) is an inherent aspect of medical and health diagnostics, but [deep learning methods that consider uncertainty are rarely used due to the scalability challenges of such methods](https://arxiv.org/pdf/1906.01620.pdf).  A key example being bayesian deep learning methods.

The first objective of this project is to

* Apply deep learning, amongst other methods, to the stated problem within an engineering design/prototype that is not constrained by scalability.

* Investigate and apply interpretability options.

Note: non-bayesian deep convolutional neural networks [has been applied to skin cancer images](https://cs.stanford.edu/people/esteva/nature/).

<br>
<br>

### The Data

As noted above, this project's modelling challenge is focused on the International Skin Imaging Collaboration’s (ISIC's) dermoscopic images of skin lesions.  It is specifically using a subset of the images of the [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/), i.e.,

<br>

|file| description|size|
|:---|:---|:---|
|[ISIC_2019_Training_Input.zip](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip)|25,331 JPEG images of skin lesions|~9GB|
|[ISIC_2019_Training_Metadata.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Metadata.csv)|25,331 metadata entries of age, sex, general anatomic site, and common lesion identifier|1.15MB|
|[ISIC_2019_Training_GroundTruth.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv)|25,331 entries of gold standard lesion diagnoses|1.23MB|

<br>

To ensure availability these three data files are also stored in a [GitHub repository](https://github.com/greyhypotheses/dermatology/tree/master/data).  The images are either the same as those hosted by the [ISIC Archive API](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation) or  down-sampled versions.  Future modelling projects might involve re-visiting the original images of the [ISIC Archive API](https://isic-archive.com/api/v1).  The API is documented at [ISIC Archive API Documentation](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation).  The data set outlined below might be used if the ground truths are released in time.

* [ISIC_2019_Test_Input.zip](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip): 8,238 JPEG images of skin lesions
* [ISIC_2019_Test_Metadata.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Metadata.csv): 8,238 metadata entries of age, sex, and general anatomic site

<br>
<br>

### Preliminary Analysis of Metadata

A preliminary analysis of the metadata is hosted in the notebook [preliminary.ipynb](https://colab.research.google.com/drive/1H1Afh8siQ6bsVdVaaq4qoQASQhNnoyWT).

<br>
<br>

### Copyright and Attribution

`Details: https://challenge2019.isic-archive.com/data.html`

The images and metadata of the "ISIC 2019: Training" data used herein are licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/) (CC-BY-NC).  The copyright holders are:

* BCN_20000 Dataset: &copy; Department of Dermatology, Hospital Clínic de Barcelona, https://arxiv.org/abs/1908.02288 <sup>4</sup>
* HAM10000 Dataset: &copy; ViDIR Group, Department of Dermatology, Medical University of Vienna, https://www.nature.com/articles/sdata2018161 <sup>1</sup>
* MSK Dataset: &copy; Anonymous; https://arxiv.org/abs/1710.05006, https://arxiv.org/abs/1902.03368 <sup>2, 3</sup>

<br>

References

1. P. Tschandl, C. Rosendahl, H. Kittler: [The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://www.nature.com/articles/sdata2018161),  Scietific Data, Volume 5, Article Number: 180161, 2018, doi:10.1038/sdata.2018.161
2. Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: [Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)](https://arxiv.org/pdf/1710.05006.pdf), 2018, arXiv:1710.05006
3. Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael A. Marchetti, Harald Kittler, Allan Halpern: [Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)](https://arxiv.org/abs/1902.03368), 2019, arXiv:1902.03368
4. Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Cristina Carrera, Alicia Barreiro, Allan C. Halpern, Susana Puig, Josep Malvehy: [BCN20000: Dermoscopic Lesions in the Wild](https://arxiv.org/pdf/1908.02288.pdf), 2019, arXiv:1908.02288
