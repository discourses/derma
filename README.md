Associated repositories:

* [dermatology](https://github.com/greyhypotheses/dermatology): An image data repository of the original and augmented images.
* [augmentation](https://github.com/greyhypotheses/augmentation): This repository's package creates augmentations of the original images.
* [derma.statistics]

<br>

# Derma

* [Brief Repository Notes](#brief-repository-notes)
  * [Temporary Notes](#temporary-notes)
* [Project](#automatic-identification-of-skin-lesion-types)
  * [Problem Statement](#problem-statement)
  * [Rationale](#rationale)
  * [The Data](#the-data)
  * [Preliminary Analysis of Metadata](https://drive.google.com/file/d/1H1Afh8siQ6bsVdVaaq4qoQASQhNnoyWT/view?usp=sharing)
  * [Copyright & Attribution](#copyright-and-attribution)

<br>
<br>

## Brief Repository Notes

This repository uses the wonderful continuous integration & delivery tool GitHub Actions. Hence, a variety of tests are conducted continuously.  The badges below will continuously highlight the state of each repository branch w.r.t. GitHub Action's actions.

branch|state
:---|:---
develop|![](https://github.com/greyhypotheses/derma/workflows/Derma%20Python%20Package/badge.svg?branch=develop)
master|![](https://github.com/greyhypotheses/derma/workflows/Derma%20Python%20Package/badge.svg?branch=master)


### Temporary Notes

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

within a Windows operating system, and the command

```
>> pip list
```

is used to list the set of directly & indirectly installed packages.  Always remember to upgrade pip before populating the environment

```
>> python -m pip install --upgrade pip==20.0.2
```

The [requirements](requirements.txt) document lists the directly installed packages and their versions.  Thus far the TensorFlow version used by this package/repository is TensorFlow 2.0.1
```
>> pip install --upgrade tensorflow==2.0.1
```

The TensorFlow installation step installs numpy & requests.  Whereas

* pandas
* pytest
* coverage
* pytest-cov
* pylint
* PyYAML

were installed separately.

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
This project has been chosen as a precursor, test case, to applying bayesian deep learning, amongst other bayesian techniques, to diagnostic, prognostic, and pathogenetic challenges in medicine.  [Uncertainty](https://www.stat.berkeley.edu/~aldous/157/Papers/Fox_Ulkumen.pdf) is an inherent aspect of medical and health diagnostics, but [deep learning methods that consider uncertainty are rarely used due to the scalability challenges of such methods](https://arxiv.org/pdf/1906.01620.pdf).  A key example being bayesian deep learning methods.

Bearing in mind the potential of deep learning in the fields of medicine & health, the objective of this project is to

* Apply bayesian deep learning, amongst other methods, to the stated problem within an engineering design/prototype that is not constrained by scalability; make the best use of bayes for model selection.

* Investigate and apply interpretability options.

Non-bayesian deep convolutional neural networks [has been applied to skin cancer images](https://cs.stanford.edu/people/esteva/nature/).

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
<br>

To ensure availability these three data files are also stored in a [GitHub repository](https://github.com/greyhypotheses/dermatology/tree/master/data).  The images are either the same as those hosted by the [ISIC Archive API](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation) or  down-sampled versions.  Future modelling projects might involve re-visiting the original images of the [ISIC Archive API](https://isic-archive.com/api/v1).  The API is documented at [ISIC Archive API Documentation](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation).  The data set outlined below might be used if the ground truths are released in time.

* [ISIC_2019_Test_Input.zip](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip): 8,238 JPEG images of skin lesions
* [ISIC_2019_Test_Metadata.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Metadata.csv): 8,238 metadata entries of age, sex, and general anatomic site

<br>
<br>

### Preliminary Analysis of Metadata

A preliminary analysis of the metadata is hosted in the notebook [preliminary.ipynb](https://drive.google.com/file/d/1H1Afh8siQ6bsVdVaaq4qoQASQhNnoyWT/view?usp=sharing).

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
