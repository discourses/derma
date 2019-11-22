## Identification of Cancerous/Pre-cancerous Skin Lesion Types
Via Dermoscopic Images of Skin Lesions

<br>

### Problem Statement
The World Health Organisation lists cancer as the [second leading cause of death globally](https://www.who.int/news-room/fact-sheets/detail/cancer);  the 2018 death estimate is 9.6 million.  And, early diagnosis or effective assessment  is usually critical to effective treatment and survival.  One common tool for early diagnosis, cancer precursor investigations, and/or tumour assessment is medical imaging.  For example, magnetic resonance imaging [for brain tumours](https://www.nature.com/articles/sdata2017117.pdf), [chest radiographs](https://arxiv.org/pdf/1901.07031.pdf) for investigating symptoms suggestive of lung cancer, [mammography](https://breast-cancer-research.biomedcentral.com/track/pdf/10.1186/s13058-015-0525-z) for breast cancer, etc.  A challenge, as the mammography paper illustrates, is accurate interpretation of medical images.

This project is focused on image classification for cancer diagnostics, it is specifically focused on the [International Skin Imaging Collaboration’s](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/isicArchive) dermoscopic images of skin lesions.  The aim is the

  Automatic classification of dermoscopic images according to 9 diagnostic classes: Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), Dermatofibroma, Vascular Lesion, Squamous Cell Carcinoma, Unknown

<br>

### Rationale
This project has been chosen as a precursor, test case, to applying bayesian deep learning, amongst other bayesian techniques, to diagnostic, prognostic, and pathogenetic challenges in medicine.  [Uncertainty](https://www.stat.berkeley.edu/~aldous/157/Papers/Fox_Ulkumen.pdf) is an inherent aspect of medical and health diagnostics, but [deep learning methods that consider uncertainty are rarely used due to the scalability challenges of such methods](https://arxiv.org/pdf/1906.01620.pdf).  A key example being bayesian deep learning methods.

Bearing in mind the potential of deep learning in the fields of medicine & health, the objective of this project is to

* Apply bayesian deep learning, amongst other methods, to the stated problem within an engineering design/prototype that is not constrained by scalability; make the best use of bayes for model selection.

* Investigate and apply interpretability options.

Non-bayesian deep convolutional neural networks [has been applied to skin cancer images](https://cs.stanford.edu/people/esteva/nature/).

<br>

### The Data

The data set of the project in question:

|file | description|
|---|---|
|[ISIC_2019_Training_Input.zip](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip)|25,331 JPEG images of skin lesions|
|[ISIC_2019_Training_Metadata.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Metadata.csv)|25,331 metadata entries of age, sex, general anatomic site, and common lesion identifier|
|[ISIC_2019_Training_GroundTruth.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv)|25,331 entries of gold standard lesion diagnoses|

<br>

A set for future use if the ground truths are available:

|file | description|
|---|---|
|[ISIC_2019_Test_Input.zip](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip)|8,238 JPEG images of skin lesions|
|[ISIC_2019_Test_Metadata.csv](https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Metadata.csv)|8,238 metadata entries of age, sex, and general anatomic site|

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

