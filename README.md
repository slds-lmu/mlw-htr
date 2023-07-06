A tailored Handwritten-Text-Recognition System for the Middle Latin Dictionary Project at the Bavarian Academy of Sciences and Humanities
==============================

# How to Install and Run the Main App?

The end-to-end pipeline, which is pip-installable can be found in the `Lectiomat` folder.
Navigate to the `Lectiomat` folder (`cd Lectiomat`) and run `pip install .`.


Run the app after installation in `python`:
```
from lectiomat import Lectiomat
lectio = Lectiomat()

```

# MLW Dictionary

- Annotation of ~ 114k data points for training of an object detection model using OFA (Wang et al., 2022)
- Training of a YOLOv8 model for object detection to extract the lemmas
- Training of a HTR model based on the transformer architecture
- Multiple experiments to obtain the best model (CER 0.015, SWIN + GPT-2)
- Lectiomat library (mlw-lectiomat) for the bavarian academy of sciences and humanities


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile including shortcuts like `make train-ocr` or `make eval`.
    ├── README.md          <- This file.
    ├── .gitignore         <- Version management blacklisting.
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── requirements.txt   <- The requirements.txt file including all dependencies.
    │
    ├── config             <- Include config.yaml files for training.
    │
    ├── data               <- All data for this project.
    │
    ├── paper-package      <- Pip-installable app for the using our models. Another README is provided in this folder.
    │
    ├── mlw-data           <- Files for downloading the data.
    │
    ├── models             <- Trained models and tokenizer.
    │
    ├── notebooks          <- Notebooks, used throughout the project.
    │
    ├── src                <- Source code for use in this project. Another README is provided in this folder.
    │
    └── tests              <- Tests for the MLW dataset.


--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
