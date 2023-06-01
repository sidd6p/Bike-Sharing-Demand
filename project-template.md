# Predict Bike Sharing Demand with AutoGluon Template

## Project: Predict Bike Sharing Demand with AutoGluon
This notebook is a template with each step that you need to complete for the project.

Please fill in your code where there are explicit `?` markers in the notebook. You are welcome to add more cells and code as you see fit.

Once you have completed all the code implementations, please export your notebook as a HTML file so the reviews can view your code. Make sure you have all outputs correctly outputted.

`File-> Export Notebook As... -> Export Notebook as HTML`

There is a writeup to complete as well after all code implememtation is done. Please answer all questions and attach the necessary tables and charts. You can complete the writeup in either markdown or PDF.

Completing the code template and writeup template will cover all of the rubric points for this project.

The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this notebook and also discuss the results in the writeup file.

## Step 1: Create an account with Kaggle

### Create Kaggle Account and download API key
Below is example of steps to get the API username and key. Each student will have their own username and key.

1. Open account settings.
<!-- ![kaggle1.png](attachment:kaggle1.png)
![kaggle2.png](attachment:kaggle2.png) -->
2. Scroll down to API and click Create New API Token.
<!-- ![kaggle3.png](attachment:kaggle3.png)
![kaggle4.png](attachment:kaggle4.png) -->
3. Open up `kaggle.json` and use the username and key.
<!-- ![kaggle5.png](attachment:kaggle5.png) -->

## Step 2: Download the Kaggle dataset using the kaggle python library

### Open up Sagemaker Studio and use starter template

1. Notebook should be using a `ml.t3.medium` instance (2 vCPU + 4 GiB)
2. Notebook should be using kernal: `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`

### Install packages


```python
!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
```

    Requirement already satisfied: pip in /usr/local/lib/python3.8/dist-packages (21.3.1)
    Collecting pip
      Using cached pip-23.1.2-py3-none-any.whl (2.1 MB)
    Installing collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 21.3.1
        Uninstalling pip-21.3.1:
          Successfully uninstalled pip-21.3.1
    Successfully installed pip-23.1.2
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    Requirement already satisfied: setuptools in /usr/local/lib/python3.8/dist-packages (59.3.0)
    Collecting setuptools
      Using cached setuptools-67.8.0-py3-none-any.whl (1.1 MB)
    Requirement already satisfied: wheel in /usr/lib/python3/dist-packages (0.34.2)
    Collecting wheel
      Using cached wheel-0.40.0-py3-none-any.whl (64 kB)
    Installing collected packages: wheel, setuptools
      Attempting uninstall: wheel
        Found existing installation: wheel 0.34.2
        Uninstalling wheel-0.34.2:
          Successfully uninstalled wheel-0.34.2
      Attempting uninstall: setuptools
        Found existing installation: setuptools 59.3.0
        Uninstalling setuptools-59.3.0:
          Successfully uninstalled setuptools-59.3.0
    Successfully installed setuptools-67.8.0 wheel-0.40.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting mxnet<2.0.0
      Using cached mxnet-1.9.1-py3-none-manylinux2014_x86_64.whl (49.1 MB)
    Collecting bokeh==2.0.1
      Using cached bokeh-2.0.1-py3-none-any.whl
    Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (5.4.1)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (2.8.0)
    Requirement already satisfied: Jinja2>=2.7 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (3.0.3)
    Requirement already satisfied: numpy>=1.11.3 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (1.19.1)
    Requirement already satisfied: pillow>=4.0 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (9.0.0)
    Requirement already satisfied: packaging>=16.8 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (21.3)
    Requirement already satisfied: tornado>=5 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (6.0.4)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.8/dist-packages (from bokeh==2.0.1) (4.0.1)
    Requirement already satisfied: requests<3,>=2.20.0 in /usr/local/lib/python3.8/dist-packages (from mxnet<2.0.0) (2.27.1)
    Requirement already satisfied: graphviz<0.9.0,>=0.8.1 in /usr/local/lib/python3.8/dist-packages (from mxnet<2.0.0) (0.8.4)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.8/dist-packages (from Jinja2>=2.7->bokeh==2.0.1) (2.0.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging>=16.8->bokeh==2.0.1) (3.0.7)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.1->bokeh==2.0.1) (1.16.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (1.26.8)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (2.0.10)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests<3,>=2.20.0->mxnet<2.0.0) (3.3)
    Installing collected packages: mxnet, bokeh
      Attempting uninstall: bokeh
        Found existing installation: bokeh 2.4.2
        Uninstalling bokeh-2.4.2:
          Successfully uninstalled bokeh-2.4.2
    Successfully installed bokeh-2.0.1 mxnet-1.9.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting autogluon
      Downloading autogluon-0.7.0-py3-none-any.whl (9.7 kB)
    Collecting autogluon.core[all]==0.7.0 (from autogluon)
      Downloading autogluon.core-0.7.0-py3-none-any.whl (218 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m218.3/218.3 kB[0m [31m101.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.features==0.7.0 (from autogluon)
      Downloading autogluon.features-0.7.0-py3-none-any.whl (60 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.1/60.1 kB[0m [31m154.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.tabular[all]==0.7.0 (from autogluon)
      Downloading autogluon.tabular-0.7.0-py3-none-any.whl (292 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m292.2/292.2 kB[0m [31m327.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.multimodal==0.7.0 (from autogluon)
      Downloading autogluon.multimodal-0.7.0-py3-none-any.whl (331 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m331.1/331.1 kB[0m [31m175.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting autogluon.timeseries[all]==0.7.0 (from autogluon)
      Downloading autogluon.timeseries-0.7.0-py3-none-any.whl (108 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m108.7/108.7 kB[0m [31m194.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting numpy<1.27,>=1.21 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading numpy-1.24.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.3/17.3 MB[0m [31m255.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: scipy<1.12,>=1.5.4 in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (1.7.0)
    Requirement already satisfied: scikit-learn<1.3,>=1.0 in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (1.0.2)
    Collecting networkx<3.0,>=2.3 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading networkx-2.8.8-py3-none-any.whl (2.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m297.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pandas<1.6,>=1.4.1 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading pandas-1.5.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.2/12.2 MB[0m [31m257.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: tqdm<5,>=4.38 in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (4.39.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (2.27.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (3.5.1)
    Requirement already satisfied: boto3<2,>=1.10 in /usr/local/lib/python3.8/dist-packages (from autogluon.core[all]==0.7.0->autogluon) (1.20.42)
    Collecting autogluon.common==0.7.0 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading autogluon.common-0.7.0-py3-none-any.whl (45 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.0/45.0 kB[0m [31m202.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting hyperopt<0.2.8,>=0.2.7 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading hyperopt-0.2.7-py2.py3-none-any.whl (1.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m320.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting ray[tune]<2.3,>=2.2 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading ray-2.2.0-cp38-cp38-manylinux2014_x86_64.whl (57.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.4/57.4 MB[0m [31m126.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting Pillow<9.6,>=9.3 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading Pillow-9.5.0-cp38-cp38-manylinux_2_28_x86_64.whl (3.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.4/3.4 MB[0m [31m287.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting jsonschema<4.18,>=4.14 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading jsonschema-4.17.3-py3-none-any.whl (90 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m90.4/90.4 kB[0m [31m273.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting seqeval<1.3.0,>=1.2.2 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading seqeval-1.2.2.tar.gz (43 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.6/43.6 kB[0m [31m202.7 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting evaluate<0.4.0,>=0.2.2 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading evaluate-0.3.0-py3-none-any.whl (72 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.9/72.9 kB[0m [31m198.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting accelerate<0.17,>=0.9 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading accelerate-0.16.0-py3-none-any.whl (199 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m199.7/199.7 kB[0m [31m310.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting timm<0.7.0,>=0.6.12 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading timm-0.6.13-py3-none-any.whl (549 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m549.1/549.1 kB[0m [31m357.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch<1.14,>=1.9 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading torch-1.13.1-cp38-cp38-manylinux1_x86_64.whl (887.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m887.4/887.4 MB[0m [31m244.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting torchvision<0.15.0 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading torchvision-0.14.1-cp38-cp38-manylinux1_x86_64.whl (24.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.2/24.2 MB[0m [31m223.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting fairscale<0.4.14,>=0.4.5 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading fairscale-0.4.13.tar.gz (266 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m266.3/266.3 kB[0m [31m286.7 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Installing backend dependencies ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hCollecting scikit-image<0.20.0,>=0.19.1 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading scikit_image-0.19.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (14.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.0/14.0 MB[0m [31m191.7 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting pytorch-lightning<1.10.0,>=1.9.0 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading pytorch_lightning-1.9.5-py3-none-any.whl (829 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m829.5/829.5 kB[0m [31m369.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting text-unidecode<1.4,>=1.3 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.2/78.2 kB[0m [31m257.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torchmetrics<0.9.0,>=0.8.0 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading torchmetrics-0.8.2-py3-none-any.whl (409 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m409.8/409.8 kB[0m [31m348.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting transformers<4.27.0,>=4.23.0 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading transformers-4.26.1-py3-none-any.whl (6.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.3/6.3 MB[0m [31m167.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting nptyping<2.5.0,>=1.4.4 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading nptyping-2.4.1-py3-none-any.whl (36 kB)
    Collecting omegaconf<2.3.0,>=2.1.1 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading omegaconf-2.2.3-py3-none-any.whl (79 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.3/79.3 kB[0m [31m255.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting sentencepiece<0.2.0,>=0.1.95 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading sentencepiece-0.1.99-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m284.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pytorch-metric-learning<2.0,>=1.3.0 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading pytorch_metric_learning-1.7.3-py3-none-any.whl (112 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m112.2/112.2 kB[0m [31m275.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nlpaug<1.2.0,>=1.1.10 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading nlpaug-1.1.11-py3-none-any.whl (410 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.5/410.5 kB[0m [31m344.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nltk<4.0.0,>=3.4.5 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading nltk-3.8.1-py3-none-any.whl (1.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m369.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting openmim<0.4.0,>0.1.5 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading openmim-0.3.7-py2.py3-none-any.whl (51 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m51.3/51.3 kB[0m [31m199.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting defusedxml<0.7.2,>=0.7.1 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Requirement already satisfied: jinja2<3.2,>=3.0.3 in /usr/local/lib/python3.8/dist-packages (from autogluon.multimodal==0.7.0->autogluon) (3.0.3)
    Collecting tensorboard<3,>=2.9 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading tensorboard-2.13.0-py3-none-any.whl (5.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.6/5.6 MB[0m [31m270.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pytesseract<0.3.11,>=0.3.9 (from autogluon.multimodal==0.7.0->autogluon)
      Downloading pytesseract-0.3.10-py3-none-any.whl (14 kB)
    Collecting catboost<1.2,>=1.0 (from autogluon.tabular[all]==0.7.0->autogluon)
      Downloading catboost-1.1.1-cp38-none-manylinux1_x86_64.whl (76.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m76.6/76.6 MB[0m [31m221.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting lightgbm<3.4,>=3.3 (from autogluon.tabular[all]==0.7.0->autogluon)
      Downloading lightgbm-3.3.5-py3-none-manylinux1_x86_64.whl (2.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m313.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting xgboost<1.8,>=1.6 (from autogluon.tabular[all]==0.7.0->autogluon)
      Downloading xgboost-1.7.5-py3-none-manylinux2014_x86_64.whl (200.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m200.3/200.3 MB[0m [31m217.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting fastai<2.8,>=2.3.1 (from autogluon.tabular[all]==0.7.0->autogluon)
      Downloading fastai-2.7.12-py3-none-any.whl (233 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m233.1/233.1 kB[0m [31m287.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: joblib<2,>=1.1 in /usr/local/lib/python3.8/dist-packages (from autogluon.timeseries[all]==0.7.0->autogluon) (1.1.0)
    Collecting statsmodels<0.14,>=0.13.0 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading statsmodels-0.13.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.9/9.9 MB[0m [31m228.7 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting gluonts<0.13,>=0.12.0 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading gluonts-0.12.8-py3-none-any.whl (1.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m281.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting statsforecast<1.5,>=1.4.0 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading statsforecast-1.4.0-py3-none-any.whl (91 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m92.0/92.0 kB[0m [31m202.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting ujson<6,>=5 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading ujson-5.7.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (52 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m52.8/52.8 kB[0m [31m205.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting sktime<0.16,>=0.14 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading sktime-0.15.1-py3-none-any.whl (16.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m16.0/16.0 MB[0m [31m185.5 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting tbats<2,>=1.1 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading tbats-1.1.3-py3-none-any.whl (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.0/44.0 kB[0m [31m201.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pmdarima<1.9,>=1.8.2 (from autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading pmdarima-1.8.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (1.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m368.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: psutil<6,>=5.7.3 in /usr/local/lib/python3.8/dist-packages (from autogluon.common==0.7.0->autogluon.core[all]==0.7.0->autogluon) (5.9.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.8/dist-packages (from autogluon.common==0.7.0->autogluon.core[all]==0.7.0->autogluon) (67.8.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.8/dist-packages (from accelerate<0.17,>=0.9->autogluon.multimodal==0.7.0->autogluon) (21.3)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.8/dist-packages (from accelerate<0.17,>=0.9->autogluon.multimodal==0.7.0->autogluon) (5.4.1)
    Requirement already satisfied: botocore<1.24.0,>=1.23.42 in /usr/local/lib/python3.8/dist-packages (from boto3<2,>=1.10->autogluon.core[all]==0.7.0->autogluon) (1.23.42)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.8/dist-packages (from boto3<2,>=1.10->autogluon.core[all]==0.7.0->autogluon) (0.10.0)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /usr/local/lib/python3.8/dist-packages (from boto3<2,>=1.10->autogluon.core[all]==0.7.0->autogluon) (0.5.0)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.8/dist-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.7.0->autogluon) (0.8.4)
    Requirement already satisfied: plotly in /usr/local/lib/python3.8/dist-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.7.0->autogluon) (5.5.0)
    Requirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from catboost<1.2,>=1.0->autogluon.tabular[all]==0.7.0->autogluon) (1.16.0)
    Collecting datasets>=2.0.0 (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading datasets-2.12.0-py3-none-any.whl (474 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m474.6/474.6 kB[0m [31m361.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: dill in /usr/local/lib/python3.8/dist-packages (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon) (0.3.4)
    Collecting tqdm<5,>=4.38 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading tqdm-4.65.0-py3-none-any.whl (77 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.1/77.1 kB[0m [31m254.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting xxhash (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading xxhash-3.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (213 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m213.0/213.0 kB[0m [31m315.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: multiprocess in /usr/local/lib/python3.8/dist-packages (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon) (0.70.12.2)
    Requirement already satisfied: fsspec[http]>=2021.05.0 in /usr/local/lib/python3.8/dist-packages (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon) (2022.1.0)
    Collecting huggingface-hub>=0.7.0 (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading huggingface_hub-0.15.0-py3-none-any.whl (236 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m236.8/236.8 kB[0m [31m334.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting responses<0.19 (from evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: pip in /usr/local/lib/python3.8/dist-packages (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon) (23.1.2)
    Collecting fastdownload<2,>=0.0.5 (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading fastdownload-0.0.7-py3-none-any.whl (12 kB)
    Collecting fastcore<1.6,>=1.5.29 (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading fastcore-1.5.29-py3-none-any.whl (67 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.6/67.6 kB[0m [31m238.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting fastprogress>=0.2.4 (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading fastprogress-1.0.3-py3-none-any.whl (12 kB)
    Collecting spacy<4 (from fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading spacy-3.5.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.8/6.8 MB[0m [31m280.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pydantic~=1.7 (from gluonts<0.13,>=0.12.0->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading pydantic-1.10.8-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.2/3.2 MB[0m [31m335.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting toolz~=0.10 (from gluonts<0.13,>=0.12.0->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading toolz-0.12.0-py3-none-any.whl (55 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.8/55.8 kB[0m [31m220.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.8/dist-packages (from gluonts<0.13,>=0.12.0->autogluon.timeseries[all]==0.7.0->autogluon) (4.0.1)
    Collecting future (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==0.7.0->autogluon)
      Downloading future-0.18.3.tar.gz (840 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m840.9/840.9 kB[0m [31m361.6 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: cloudpickle in /usr/local/lib/python3.8/dist-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==0.7.0->autogluon) (2.0.0)
    Collecting py4j (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==0.7.0->autogluon)
      Downloading py4j-0.10.9.7-py2.py3-none-any.whl (200 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m200.5/200.5 kB[0m [31m283.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.8/dist-packages (from jinja2<3.2,>=3.0.3->autogluon.multimodal==0.7.0->autogluon) (2.0.1)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema<4.18,>=4.14->autogluon.multimodal==0.7.0->autogluon) (21.4.0)
    Collecting importlib-resources>=1.4.0 (from jsonschema<4.18,>=4.14->autogluon.multimodal==0.7.0->autogluon)
      Downloading importlib_resources-5.12.0-py3-none-any.whl (36 kB)
    Collecting pkgutil-resolve-name>=1.3.10 (from jsonschema<4.18,>=4.14->autogluon.multimodal==0.7.0->autogluon)
      Downloading pkgutil_resolve_name-1.3.10-py3-none-any.whl (4.7 kB)
    Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 (from jsonschema<4.18,>=4.14->autogluon.multimodal==0.7.0->autogluon)
      Downloading pyrsistent-0.19.3-py3-none-any.whl (57 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m225.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel in /usr/local/lib/python3.8/dist-packages (from lightgbm<3.4,>=3.3->autogluon.tabular[all]==0.7.0->autogluon) (0.40.0)
    Collecting gdown>=4.0.0 (from nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==0.7.0->autogluon)
      Downloading gdown-4.7.1-py3-none-any.whl (15 kB)
    Collecting click (from nltk<4.0.0,>=3.4.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading click-8.1.3-py3-none-any.whl (96 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.6/96.6 kB[0m [31m265.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting regex>=2021.8.3 (from nltk<4.0.0,>=3.4.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading regex-2023.5.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (771 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m771.9/771.9 kB[0m [31m355.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting antlr4-python3-runtime==4.9.* (from omegaconf<2.3.0,>=2.1.1->autogluon.multimodal==0.7.0->autogluon)
      Downloading antlr4-python3-runtime-4.9.3.tar.gz (117 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m117.0/117.0 kB[0m [31m283.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: colorama in /usr/local/lib/python3.8/dist-packages (from openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon) (0.4.3)
    Collecting model-index (from openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading model_index-0.1.11-py3-none-any.whl (34 kB)
    Collecting rich (from openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading rich-13.4.1-py3-none-any.whl (239 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m239.4/239.4 kB[0m [31m306.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tabulate in /usr/local/lib/python3.8/dist-packages (from openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon) (0.8.9)
    Collecting python-dateutil>=2.8.1 (from pandas<1.6,>=1.4.1->autogluon.core[all]==0.7.0->autogluon)
      Downloading python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m247.7/247.7 kB[0m [31m328.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.8/dist-packages (from pandas<1.6,>=1.4.1->autogluon.core[all]==0.7.0->autogluon) (2021.3)
    Requirement already satisfied: Cython!=0.29.18,>=0.29 in /usr/local/lib/python3.8/dist-packages (from pmdarima<1.9,>=1.8.2->autogluon.timeseries[all]==0.7.0->autogluon) (0.29.26)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.8/dist-packages (from pmdarima<1.9,>=1.8.2->autogluon.timeseries[all]==0.7.0->autogluon) (1.26.8)
    Collecting lightning-utilities>=0.6.0.post0 (from pytorch-lightning<1.10.0,>=1.9.0->autogluon.multimodal==0.7.0->autogluon)
      Downloading lightning_utilities-0.8.0-py3-none-any.whl (20 kB)
    Collecting filelock (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading filelock-3.12.0-py3-none-any.whl (10 kB)
    Collecting msgpack<2.0.0,>=1.0.0 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading msgpack-1.0.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (322 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m322.4/322.4 kB[0m [31m340.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: protobuf!=3.19.5,>=3.15.3 in /usr/local/lib/python3.8/dist-packages (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon) (3.19.3)
    Collecting aiosignal (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Collecting frozenlist (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading frozenlist-1.3.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (161 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m161.3/161.3 kB[0m [31m302.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting virtualenv>=20.0.24 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading virtualenv-20.23.0-py3-none-any.whl (3.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.3/3.3 MB[0m [31m320.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting grpcio>=1.32.0 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading grpcio-1.54.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m297.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboardX>=1.9 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading tensorboardX-2.6-py2.py3-none-any.whl (114 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.5/114.5 kB[0m [31m273.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests->autogluon.core[all]==0.7.0->autogluon) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.8/dist-packages (from requests->autogluon.core[all]==0.7.0->autogluon) (2.0.10)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests->autogluon.core[all]==0.7.0->autogluon) (3.3)
    Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.8/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.7.0->autogluon) (2.14.1)
    Collecting tifffile>=2019.7.26 (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.7.0->autogluon)
      Downloading tifffile-2023.4.12-py3-none-any.whl (219 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m219.4/219.4 kB[0m [31m312.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting PyWavelets>=1.1.1 (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.7.0->autogluon)
      Downloading PyWavelets-1.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.9/6.9 MB[0m [31m266.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from scikit-learn<1.3,>=1.0->autogluon.core[all]==0.7.0->autogluon) (3.0.0)
    Collecting numpy<1.27,>=1.21 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading numpy-1.22.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m16.9/16.9 MB[0m [31m169.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting deprecated>=1.2.13 (from sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading Deprecated-1.2.14-py2.py3-none-any.whl (9.6 kB)
    Requirement already satisfied: numba>=0.55 in /usr/local/lib/python3.8/dist-packages (from sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon) (0.55.0)
    Collecting scipy<1.12,>=1.5.4 (from autogluon.core[all]==0.7.0->autogluon)
      Downloading scipy-1.10.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m34.5/34.5 MB[0m [31m190.6 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting patsy>=0.5.2 (from statsmodels<0.14,>=0.13.0->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading patsy-0.5.3-py2.py3-none-any.whl (233 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m233.8/233.8 kB[0m [31m337.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting absl-py>=0.4 (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading absl_py-1.4.0-py3-none-any.whl (126 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m126.5/126.5 kB[0m [31m290.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth<3,>=1.6.3 (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading google_auth-2.19.0-py2.py3-none-any.whl (181 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.3/181.3 kB[0m [31m316.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
    Collecting markdown>=2.6.8 (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading Markdown-3.4.3-py3-none-any.whl (93 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.9/93.9 kB[0m [31m256.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting protobuf!=3.19.5,>=3.15.3 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading protobuf-4.23.2-cp37-abi3-manylinux2014_x86_64.whl (304 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m304.5/304.5 kB[0m [31m333.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading tensorboard_data_server-0.7.0-py3-none-manylinux2014_x86_64.whl (6.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.6/6.6 MB[0m [31m298.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon) (2.0.2)
    Collecting nvidia-cuda-runtime-cu11==11.7.99 (from torch<1.14,>=1.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl (849 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m346.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu11==8.5.0.96 (from torch<1.14,>=1.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl (557.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m242.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cublas-cu11==11.10.3.66 (from torch<1.14,>=1.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl (317.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m193.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cuda-nvrtc-cu11==11.7.99 (from torch<1.14,>=1.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl (21.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m224.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting pyDeprecate==0.3.* (from torchmetrics<0.9.0,>=0.8.0->autogluon.multimodal==0.7.0->autogluon)
      Downloading pyDeprecate-0.3.2-py3-none-any.whl (10 kB)
    Collecting tokenizers!=0.11.3,<0.14,>=0.11.1 (from transformers<4.27.0,>=4.23.0->autogluon.multimodal==0.7.0->autogluon)
      Downloading tokenizers-0.13.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m238.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.8/dist-packages (from matplotlib->autogluon.core[all]==0.7.0->autogluon) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.8/dist-packages (from matplotlib->autogluon.core[all]==0.7.0->autogluon) (4.29.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->autogluon.core[all]==0.7.0->autogluon) (1.3.2)
    Requirement already satisfied: pyparsing>=2.2.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->autogluon.core[all]==0.7.0->autogluon) (3.0.7)
    Collecting pyarrow>=8.0.0 (from datasets>=2.0.0->evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading pyarrow-12.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (39.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m39.0/39.0 MB[0m [31m201.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting aiohttp (from datasets>=2.0.0->evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading aiohttp-3.8.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m359.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting wrapt<2,>=1.10 (from deprecated>=1.2.13->sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading wrapt-1.15.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (81 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.5/81.5 kB[0m [31m245.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting beautifulsoup4 (from gdown>=4.0.0->nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==0.7.0->autogluon)
      Downloading beautifulsoup4-4.12.2-py3-none-any.whl (142 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m143.0/143.0 kB[0m [31m302.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading cachetools-5.3.1-py3-none-any.whl (9.3 kB)
    Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.3/181.3 kB[0m [31m310.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.8/dist-packages (from google-auth<3,>=1.6.3->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon) (4.7.2)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<1.1,>=0.5->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from importlib-resources>=1.4.0->jsonschema<4.18,>=4.14->autogluon.multimodal==0.7.0->autogluon) (3.7.0)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.8/dist-packages (from markdown>=2.6.8->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon) (4.10.1)
    Requirement already satisfied: llvmlite<0.39,>=0.38.0rc1 in /usr/local/lib/python3.8/dist-packages (from numba>=0.55->sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon) (0.38.0)
    INFO: pip is looking at multiple versions of numba to determine which version is compatible with other requirements. This could take a while.
    Collecting numba>=0.55 (from sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading numba-0.57.0-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.6/3.6 MB[0m [31m294.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting llvmlite<0.41,>=0.40.0dev0 (from numba>=0.55->sktime<0.16,>=0.14->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading llvmlite-0.40.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.1/42.1 MB[0m [31m257.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting typing-extensions~=4.0 (from gluonts<0.13,>=0.12.0->autogluon.timeseries[all]==0.7.0->autogluon)
      Downloading typing_extensions-4.6.2-py3-none-any.whl (31 kB)
    Collecting spacy-legacy<3.1.0,>=3.0.11 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading spacy_legacy-3.0.12-py2.py3-none-any.whl (29 kB)
    Collecting spacy-loggers<2.0.0,>=1.0.0 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading spacy_loggers-1.0.4-py3-none-any.whl (11 kB)
    Collecting murmurhash<1.1.0,>=0.28.0 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading murmurhash-1.0.9-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (21 kB)
    Collecting cymem<2.1.0,>=2.0.2 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading cymem-2.0.7-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (36 kB)
    Collecting preshed<3.1.0,>=3.0.2 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading preshed-3.0.8-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.8/130.8 kB[0m [31m301.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting thinc<8.2.0,>=8.1.8 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading thinc-8.1.10-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (928 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m928.2/928.2 kB[0m [31m258.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting wasabi<1.2.0,>=0.9.1 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading wasabi-1.1.1-py3-none-any.whl (27 kB)
    Collecting srsly<3.0.0,>=2.4.3 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading srsly-2.4.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (493 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m493.5/493.5 kB[0m [31m360.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting catalogue<2.1.0,>=2.0.6 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading catalogue-2.0.8-py3-none-any.whl (17 kB)
    Collecting typer<0.8.0,>=0.3.0 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading typer-0.7.0-py3-none-any.whl (38 kB)
    Collecting pathy>=0.10.0 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading pathy-0.10.1-py3-none-any.whl (48 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m48.9/48.9 kB[0m [31m210.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting smart-open<7.0.0,>=5.2.1 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading smart_open-6.3.0-py3-none-any.whl (56 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.8/56.8 kB[0m [31m218.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting langcodes<4.0.0,>=3.2.0 (from spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading langcodes-3.3.0-py3-none-any.whl (181 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.6/181.6 kB[0m [31m310.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting protobuf!=3.19.5,>=3.15.3 (from ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading protobuf-3.20.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m354.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting distlib<1,>=0.3.6 (from virtualenv>=20.0.24->ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading distlib-0.3.6-py2.py3-none-any.whl (468 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m468.5/468.5 kB[0m [31m357.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting platformdirs<4,>=3.2 (from virtualenv>=20.0.24->ray[tune]<2.3,>=2.2->autogluon.core[all]==0.7.0->autogluon)
      Downloading platformdirs-3.5.1-py3-none-any.whl (15 kB)
    Collecting ordered-set (from model-index->openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading ordered_set-4.1.0-py3-none-any.whl (7.6 kB)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.8/dist-packages (from plotly->catboost<1.2,>=1.0->autogluon.tabular[all]==0.7.0->autogluon) (8.0.1)
    Collecting markdown-it-py<3.0.0,>=2.2.0 (from rich->openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading markdown_it_py-2.2.0-py3-none-any.whl (84 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.5/84.5 kB[0m [31m259.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.8/dist-packages (from rich->openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon) (2.14.0)
    Collecting mdurl~=0.1 (from markdown-it-py<3.0.0,>=2.2.0->rich->openmim<0.4.0,>0.1.5->autogluon.multimodal==0.7.0->autogluon)
      Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.8/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon) (0.4.8)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<3,>=2.9->autogluon.multimodal==0.7.0->autogluon)
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m306.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting blis<0.8.0,>=0.7.8 (from thinc<8.2.0,>=8.1.8->spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading blis-0.7.9-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.2/10.2 MB[0m [31m262.3 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting confection<1.0.0,>=0.0.1 (from thinc<8.2.0,>=8.1.8->spacy<4->fastai<2.8,>=2.3.1->autogluon.tabular[all]==0.7.0->autogluon)
      Downloading confection-0.0.4-py3-none-any.whl (32 kB)
    Collecting multidict<7.0,>=4.5 (from aiohttp->datasets>=2.0.0->evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading multidict-6.0.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (121 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.3/121.3 kB[0m [31m286.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting async-timeout<5.0,>=4.0.0a3 (from aiohttp->datasets>=2.0.0->evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
    Collecting yarl<2.0,>=1.0 (from aiohttp->datasets>=2.0.0->evaluate<0.4.0,>=0.2.2->autogluon.multimodal==0.7.0->autogluon)
      Downloading yarl-1.9.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (266 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m266.9/266.9 kB[0m [31m324.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting soupsieve>1.2 (from beautifulsoup4->gdown>=4.0.0->nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==0.7.0->autogluon)
      Downloading soupsieve-2.4.1-py3-none-any.whl (36 kB)
    Collecting PySocks!=1.5.7,>=1.5.6 (from requests->autogluon.core[all]==0.7.0->autogluon)
      Downloading PySocks-1.7.1-py3-none-any.whl (16 kB)
    Building wheels for collected packages: fairscale, antlr4-python3-runtime, seqeval, future
      Building wheel for fairscale (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for fairscale: filename=fairscale-0.4.13-py3-none-any.whl size=332112 sha256=6311282742d67bcfaac3b3d03e2d69ffc5ccc624f45376d161badab22519d278
      Stored in directory: /tmp/pip-ephem-wheel-cache-dp0eg_ti/wheels/b8/02/9b/dc7d4ff5145afdd28f456dae6605a46619af0370eca30d8d7e
      Building wheel for antlr4-python3-runtime (setup.py) ... [?25ldone
    [?25h  Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.9.3-py3-none-any.whl size=144554 sha256=ce091d38e637f2f300422f9e072b5593c4127c8ebaf4eb06e30b7406cfc138d5
      Stored in directory: /tmp/pip-ephem-wheel-cache-dp0eg_ti/wheels/b1/a3/c2/6df046c09459b73cc9bb6c4401b0be6c47048baf9a1617c485
      Building wheel for seqeval (setup.py) ... [?25ldone
    [?25h  Created wheel for seqeval: filename=seqeval-1.2.2-py3-none-any.whl size=16165 sha256=086b0f666a5c92fb30629105340ec5773610576e4a32f0444c4da3b5cd1726c3
      Stored in directory: /tmp/pip-ephem-wheel-cache-dp0eg_ti/wheels/ad/5c/ba/05fa33fa5855777b7d686e843ec07452f22a66a138e290e732
      Building wheel for future (setup.py) ... [?25ldone
    [?25h  Created wheel for future: filename=future-0.18.3-py3-none-any.whl size=492022 sha256=e4480c010e6de99f0fbd6be41b809acad746a51f81aaeb00701114a165c9acb9
      Stored in directory: /tmp/pip-ephem-wheel-cache-dp0eg_ti/wheels/a0/0b/ee/e6994fadb42c1354dcccb139b0bf2795271bddfe6253ccdf11
    Successfully built fairscale antlr4-python3-runtime seqeval future
    Installing collected packages: tokenizers, text-unidecode, sentencepiece, py4j, msgpack, distlib, cymem, antlr4-python3-runtime, xxhash, wrapt, wasabi, ujson, typing-extensions, tqdm, toolz, tensorboard-data-server, spacy-loggers, spacy-legacy, soupsieve, smart-open, regex, python-dateutil, PySocks, pyrsistent, pyDeprecate, pyasn1-modules, protobuf, platformdirs, pkgutil-resolve-name, Pillow, ordered-set, omegaconf, oauthlib, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cublas-cu11, numpy, networkx, murmurhash, multidict, mdurl, llvmlite, langcodes, importlib-resources, grpcio, future, frozenlist, filelock, fastprogress, defusedxml, click, catalogue, cachetools, async-timeout, absl-py, yarl, virtualenv, typer, tifffile, tensorboardX, srsly, scipy, responses, requests-oauthlib, PyWavelets, pytesseract, pydantic, pyarrow, preshed, patsy, pandas, nvidia-cudnn-cu11, numba, nptyping, nltk, markdown-it-py, markdown, lightning-utilities, jsonschema, huggingface-hub, google-auth, fastcore, deprecated, blis, beautifulsoup4, aiosignal, xgboost, transformers, torch, statsmodels, scikit-image, rich, ray, pathy, model-index, hyperopt, google-auth-oauthlib, gluonts, gdown, fastdownload, confection, catboost, aiohttp, torchvision, torchmetrics, thinc, tensorboard, statsforecast, sktime, seqeval, pytorch-metric-learning, pmdarima, openmim, nlpaug, lightgbm, fairscale, accelerate, timm, tbats, spacy, pytorch-lightning, datasets, autogluon.common, fastai, evaluate, autogluon.features, autogluon.core, autogluon.tabular, autogluon.multimodal, autogluon.timeseries, autogluon
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.0.1
        Uninstalling typing_extensions-4.0.1:
          Successfully uninstalled typing_extensions-4.0.1
      Attempting uninstall: tqdm
        Found existing installation: tqdm 4.39.0
        Uninstalling tqdm-4.39.0:
          Successfully uninstalled tqdm-4.39.0
      Attempting uninstall: python-dateutil
        Found existing installation: python-dateutil 2.8.0
        Uninstalling python-dateutil-2.8.0:
          Successfully uninstalled python-dateutil-2.8.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 3.19.3
        Uninstalling protobuf-3.19.3:
          Successfully uninstalled protobuf-3.19.3
      Attempting uninstall: Pillow
        Found existing installation: Pillow 9.0.0
        Uninstalling Pillow-9.0.0:
          Successfully uninstalled Pillow-9.0.0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.19.1
        Uninstalling numpy-1.19.1:
          Successfully uninstalled numpy-1.19.1
      Attempting uninstall: llvmlite
        Found existing installation: llvmlite 0.38.0
        Uninstalling llvmlite-0.38.0:
          Successfully uninstalled llvmlite-0.38.0
      Attempting uninstall: scipy
        Found existing installation: scipy 1.7.0
        Uninstalling scipy-1.7.0:
          Successfully uninstalled scipy-1.7.0
      Attempting uninstall: pyarrow
        Found existing installation: pyarrow 6.0.1
        Uninstalling pyarrow-6.0.1:
          Successfully uninstalled pyarrow-6.0.1
      Attempting uninstall: pandas
        Found existing installation: pandas 1.3.0
        Uninstalling pandas-1.3.0:
          Successfully uninstalled pandas-1.3.0
      Attempting uninstall: numba
        Found existing installation: numba 0.55.0
        Uninstalling numba-0.55.0:
          Successfully uninstalled numba-0.55.0
    Successfully installed Pillow-9.5.0 PySocks-1.7.1 PyWavelets-1.4.1 absl-py-1.4.0 accelerate-0.16.0 aiohttp-3.8.4 aiosignal-1.3.1 antlr4-python3-runtime-4.9.3 async-timeout-4.0.2 autogluon-0.7.0 autogluon.common-0.7.0 autogluon.core-0.7.0 autogluon.features-0.7.0 autogluon.multimodal-0.7.0 autogluon.tabular-0.7.0 autogluon.timeseries-0.7.0 beautifulsoup4-4.12.2 blis-0.7.9 cachetools-5.3.1 catalogue-2.0.8 catboost-1.1.1 click-8.1.3 confection-0.0.4 cymem-2.0.7 datasets-2.12.0 defusedxml-0.7.1 deprecated-1.2.14 distlib-0.3.6 evaluate-0.3.0 fairscale-0.4.13 fastai-2.7.12 fastcore-1.5.29 fastdownload-0.0.7 fastprogress-1.0.3 filelock-3.12.0 frozenlist-1.3.3 future-0.18.3 gdown-4.7.1 gluonts-0.12.8 google-auth-2.19.0 google-auth-oauthlib-1.0.0 grpcio-1.54.2 huggingface-hub-0.15.0 hyperopt-0.2.7 importlib-resources-5.12.0 jsonschema-4.17.3 langcodes-3.3.0 lightgbm-3.3.5 lightning-utilities-0.8.0 llvmlite-0.40.0 markdown-3.4.3 markdown-it-py-2.2.0 mdurl-0.1.2 model-index-0.1.11 msgpack-1.0.5 multidict-6.0.4 murmurhash-1.0.9 networkx-2.8.8 nlpaug-1.1.11 nltk-3.8.1 nptyping-2.4.1 numba-0.57.0 numpy-1.22.4 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 oauthlib-3.2.2 omegaconf-2.2.3 openmim-0.3.7 ordered-set-4.1.0 pandas-1.5.3 pathy-0.10.1 patsy-0.5.3 pkgutil-resolve-name-1.3.10 platformdirs-3.5.1 pmdarima-1.8.5 preshed-3.0.8 protobuf-3.20.3 py4j-0.10.9.7 pyDeprecate-0.3.2 pyarrow-12.0.0 pyasn1-modules-0.3.0 pydantic-1.10.8 pyrsistent-0.19.3 pytesseract-0.3.10 python-dateutil-2.8.2 pytorch-lightning-1.9.5 pytorch-metric-learning-1.7.3 ray-2.2.0 regex-2023.5.5 requests-oauthlib-1.3.1 responses-0.18.0 rich-13.4.1 scikit-image-0.19.3 scipy-1.10.1 sentencepiece-0.1.99 seqeval-1.2.2 sktime-0.15.1 smart-open-6.3.0 soupsieve-2.4.1 spacy-3.5.3 spacy-legacy-3.0.12 spacy-loggers-1.0.4 srsly-2.4.6 statsforecast-1.4.0 statsmodels-0.13.5 tbats-1.1.3 tensorboard-2.13.0 tensorboard-data-server-0.7.0 tensorboardX-2.6 text-unidecode-1.3 thinc-8.1.10 tifffile-2023.4.12 timm-0.6.13 tokenizers-0.13.3 toolz-0.12.0 torch-1.13.1 torchmetrics-0.8.2 torchvision-0.14.1 tqdm-4.65.0 transformers-4.26.1 typer-0.7.0 typing-extensions-4.6.2 ujson-5.7.0 virtualenv-20.23.0 wasabi-1.1.1 wrapt-1.15.0 xgboost-1.7.5 xxhash-3.2.0 yarl-1.9.2
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

### Setup Kaggle API Key


```python
!pip install -q Kaggle

!mkdir -p /root/.kaggle
!touch /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c bike-sharing-demand
!unzip -o bike-sharing-demand.zip    
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mbike-sharing-demand.zip: Skipping, found more recently modified local copy (use --force to force download)
    Archive:  bike-sharing-demand.zip
      inflating: sampleSubmission.csv    
      inflating: test.csv                
      inflating: train.csv               


### Download and explore dataset


```python
import json
import pandas as pd

import matplotlib.pyplot as plt
import autogluon.core as ag

from autogluon.tabular import TabularPredictor
```

    /usr/local/lib/python3.8/dist-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
kaggle_username = "siddp6"
kaggle_key = "2e5f7520d0f1ef3e8e133d6b1241caf9"

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```

### Go to the [bike sharing demand competition](https://www.kaggle.com/c/bike-sharing-demand) and agree to the terms
<!-- ![kaggle6.png](attachment:kaggle6.png) -->


```python
train = pd.read_csv("train.csv",parse_dates=["datetime"])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.00000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.506614</td>
      <td>0.028569</td>
      <td>0.680875</td>
      <td>1.418427</td>
      <td>20.23086</td>
      <td>23.655084</td>
      <td>61.886460</td>
      <td>12.799395</td>
      <td>36.021955</td>
      <td>155.552177</td>
      <td>191.574132</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.116174</td>
      <td>0.166599</td>
      <td>0.466159</td>
      <td>0.633839</td>
      <td>7.79159</td>
      <td>8.474601</td>
      <td>19.245033</td>
      <td>8.164537</td>
      <td>49.960477</td>
      <td>151.039033</td>
      <td>181.144454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.82000</td>
      <td>0.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.94000</td>
      <td>16.665000</td>
      <td>47.000000</td>
      <td>7.001500</td>
      <td>4.000000</td>
      <td>36.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.50000</td>
      <td>24.240000</td>
      <td>62.000000</td>
      <td>12.998000</td>
      <td>17.000000</td>
      <td>118.000000</td>
      <td>145.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>26.24000</td>
      <td>31.060000</td>
      <td>77.000000</td>
      <td>16.997900</td>
      <td>49.000000</td>
      <td>222.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>41.00000</td>
      <td>45.455000</td>
      <td>100.000000</td>
      <td>56.996900</td>
      <td>367.000000</td>
      <td>886.000000</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB



```python
test = pd.read_csv("test.csv",parse_dates=["datetime"])
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission = pd.read_csv("sampleSubmission.csv",parse_dates=["datetime"])
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Train a model using AutoGluon’s Tabular Prediction

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.


```python
predictor = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230601_143631/"



```python
predictor.fit(train_data=train, time_limit=120, presets="best_quality")
```

    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 120s
    AutoGluon will save models to "AutogluonModels/ag-20230601_143631/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 11
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    3058.78 MB
    	Train Data (Original)  Memory Usage: 0.78 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['season', 'holiday', 'workingday', 'weather', 'humidity']
    	Types of features in processed data (raw dtype, special dtypes):
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 3 | ['season', 'weather', 'humidity']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.1s = Fit runtime
    	9 features in original data used to generate 13 features in processed data.
    	Train Data (Processed) Memory Usage: 0.98 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.12s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 79.89s of the 119.87s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.04s	 = Training   runtime
    	0.05s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 76.2s of the 116.17s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 75.99s of the 115.96s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-131.4609	 = Validation score   (-root_mean_squared_error)
    	58.24s	 = Training   runtime
    	9.85s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 8.93s of the 48.9s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-131.0771	 = Validation score   (-root_mean_squared_error)
    	21.46s	 = Training   runtime
    	1.42s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.88s of the 23.52s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.26s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 23.2s of the 23.19s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-61.1166	 = Validation score   (-root_mean_squared_error)
    	36.24s	 = Training   runtime
    	3.18s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 119.88s of the -17.5s of remaining time.
    	-61.1166	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 137.71s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230601_143631/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fe07cbc6dc0>



### Review AutoGluon's training run with ranking of models that did the best.


```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757                3.184328          36.244766            2       True          6
    1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841                0.001231           0.005084            3       True          7
    2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795                0.038734           0.029795            1       True          2
    3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632                0.000736           0.255837            2       True          5
    4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213                0.045319           0.040213            1       True          1
    5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628                1.415203          21.460628            1       True          4
    6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355                9.849940          58.240355            1       True          3
    Number of models trained: 7
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 3 | ['season', 'weather', 'humidity']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_143631/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'LightGBMXT_BAG_L1': -131.46090891834504,
      'LightGBM_BAG_L1': -131.0770800258179,
      'WeightedEnsemble_L2': -84.12506123181602,
      'LightGBMXT_BAG_L2': -61.11662760935856,
      'WeightedEnsemble_L3': -61.11662760935856},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/LightGBMXT_BAG_L1/',
      'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/LightGBM_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_143631/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230601_143631/models/LightGBMXT_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_143631/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.04021286964416504,
      'KNeighborsDist_BAG_L1': 0.029795408248901367,
      'LightGBMXT_BAG_L1': 58.240354776382446,
      'LightGBM_BAG_L1': 21.460627794265747,
      'WeightedEnsemble_L2': 0.25583696365356445,
      'LightGBMXT_BAG_L2': 36.244765758514404,
      'WeightedEnsemble_L3': 0.00508427619934082},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.045318603515625,
      'KNeighborsDist_BAG_L1': 0.03873395919799805,
      'LightGBMXT_BAG_L1': 9.849940299987793,
      'LightGBM_BAG_L1': 1.4152026176452637,
      'WeightedEnsemble_L2': 0.0007355213165283203,
      'LightGBMXT_BAG_L2': 3.184328317642212,
      'WeightedEnsemble_L3': 0.0012314319610595703},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757   
     1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841   
     2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795   
     3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632   
     4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213   
     5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628   
     6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                3.184328          36.244766            2       True   
     1                0.001231           0.005084            3       True   
     2                0.038734           0.029795            1       True   
     3                0.000736           0.255837            2       True   
     4                0.045319           0.040213            1       True   
     5                1.415203          21.460628            1       True   
     6                9.849940          58.240355            1       True   
     
        fit_order  
     0          6  
     1          7  
     2          2  
     3          5  
     4          1  
     5          4  
     6          3  }




```python
leaderboard_df = pd.DataFrame(predictor.leaderboard())
leaderboard_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757                3.184328          36.244766            2       True          6
    1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841                0.001231           0.005084            3       True          7
    2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795                0.038734           0.029795            1       True          2
    3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632                0.000736           0.255837            2       True          5
    4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213                0.045319           0.040213            1       True          1
    5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628                1.415203          21.460628            1       True          4
    6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355                9.849940          58.240355            1       True          3



    
![png](output_27_1.png)
    


### Create predictions from test dataset


```python
predictions = predictor.predict(test)
predictions.head()
```




    0    36.413548
    1    42.656593
    2    49.778008
    3    52.446659
    4    55.068695
    Name: count, dtype: float32



#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0.


```python
predictions[predictions < 0] = 0  
```

### Set predictions to submission dataframe, save, and submit


```python
submission["count"] = predictions
submission.to_csv("submission.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "Initial Submission"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 370kB/s]
    Successfully submitted to Bike Sharing Demand

#### View submission via the command line or in the web browser under the competition's page - `My Submissions`


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission.csv               2023-06-01 14:39:34  Initial Submission                                                                                  pending                              
    submission_new_hyp_3.csv     2023-06-01 07:29:37  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  complete  0.50079      0.50079       
    submission_new_hyp_2.csv     2023-06-01 07:29:35  new features with hyperparameters epoch, boost round, learning rate, extra trees                    complete  0.65221      0.65221       
    submission_new_hyp_1.csv     2023-06-01 07:29:33  new features with hyperparameters epoch, boost round                                                complete  0.60933      0.60933       
    tail: write error: Broken pipe


#### Initial score of 2.06204     

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.


```python
train.hist(figsize=(15,20))  
plt.tight_layout()
plt.show()
```


    
![png](output_39_0.png)
    



```python
train["hour"] = train["datetime"].dt.hour
train["day"] = train["datetime"].dt.dayofweek
train.drop(["datetime"], axis=1, inplace=True)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
test["hour"] = test["datetime"].dt.hour
test["day"] = test["datetime"].dt.dayofweek
test.drop(["datetime"], axis=1, inplace=True)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.


```python
train["season"] = train["season"].astype("category")
train["weather"] = train["weather"].astype("category")

test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.hist(figsize=(15, 20))
plt.tight_layout()
plt.show()
```


    
![png](output_45_0.png)
    


## Step 5: Rerun the model with the same settings as before, just with more features


```python
predictor_new_features = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230601_143940/"



```python
predictor_new_features.fit(train_data=train, time_limit=120, presets="best_quality")
```

    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 120s
    AutoGluon will save models to "AutogluonModels/ag-20230601_143940/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2588.44 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.1s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 79.91s of the 119.9s of remaining time.
    	-117.0607	 = Validation score   (-root_mean_squared_error)
    	0.02s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 79.67s of the 119.66s of remaining time.
    	-114.004	 = Validation score   (-root_mean_squared_error)
    	0.02s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 79.43s of the 119.41s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.1971	 = Validation score   (-root_mean_squared_error)
    	89.13s	 = Training   runtime
    	24.77s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.9s of the 24.13s of remaining time.
    	-60.1251	 = Validation score   (-root_mean_squared_error)
    	0.22s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 23.84s of the 23.83s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.9777	 = Validation score   (-root_mean_squared_error)
    	16.38s	 = Training   runtime
    	0.22s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 4.25s of the 4.24s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.209	 = Validation score   (-root_mean_squared_error)
    	14.98s	 = Training   runtime
    	0.11s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 119.9s of the -14.47s of remaining time.
    	-59.964	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 134.81s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230601_143940/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fdf408fd520>




```python
predictor_new_features.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456                0.000761           0.163419            3       True          7
    1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040                0.000833           0.217180            2       True          4
    2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638               24.770082          89.127638            1       True          3
    3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664                0.114307          14.976510            2       True          6
    4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528                0.217721          16.378374            2       True          5
    5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222                0.102599           0.022222            1       True          2
    6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294                0.093551           0.022294            1       True          1
    Number of models trained: 7
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_143940/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -117.06074757128302,
      'KNeighborsDist_BAG_L1': -114.00404505882429,
      'LightGBMXT_BAG_L1': -60.19709831103628,
      'WeightedEnsemble_L2': -60.12510035164886,
      'LightGBMXT_BAG_L2': -60.977719808009496,
      'LightGBM_BAG_L2': -60.209038712399085,
      'WeightedEnsemble_L3': -59.96398356047644},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/LightGBMXT_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_143940/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230601_143940/models/LightGBMXT_BAG_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_143940/models/LightGBM_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_143940/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.022294282913208008,
      'KNeighborsDist_BAG_L1': 0.02222156524658203,
      'LightGBMXT_BAG_L1': 89.12763810157776,
      'WeightedEnsemble_L2': 0.2171802520751953,
      'LightGBMXT_BAG_L2': 16.378373622894287,
      'LightGBM_BAG_L2': 14.97650957107544,
      'WeightedEnsemble_L3': 0.16341924667358398},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.0935513973236084,
      'KNeighborsDist_BAG_L1': 0.1025993824005127,
      'LightGBMXT_BAG_L1': 24.770081520080566,
      'WeightedEnsemble_L2': 0.0008327960968017578,
      'LightGBMXT_BAG_L2': 0.21772050857543945,
      'LightGBM_BAG_L2': 0.11430692672729492,
      'WeightedEnsemble_L3': 0.0007610321044921875},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456   
     1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040   
     2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638   
     3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664   
     4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528   
     5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222   
     6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000761           0.163419            3       True   
     1                0.000833           0.217180            2       True   
     2               24.770082          89.127638            1       True   
     3                0.114307          14.976510            2       True   
     4                0.217721          16.378374            2       True   
     5                0.102599           0.022222            1       True   
     6                0.093551           0.022294            1       True   
     
        fit_order  
     0          7  
     1          4  
     2          3  
     3          6  
     4          5  
     5          2  
     6          1  }




```python
leaderboard_new_df = pd.DataFrame(predictor_new_features.leaderboard())
leaderboard_new_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456                0.000761           0.163419            3       True          7
    1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040                0.000833           0.217180            2       True          4
    2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638               24.770082          89.127638            1       True          3
    3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664                0.114307          14.976510            2       True          6
    4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528                0.217721          16.378374            2       True          5
    5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222                0.102599           0.022222            1       True          2
    6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294                0.093551           0.022294            1       True          1



    
![png](output_50_1.png)
    



```python
predictions_new_features = predictor_new_features.predict(test)
predictions_new_features.head()
```




    0    25.509943
    1    11.078977
    2     9.098299
    3     8.904437
    4     8.784183
    Name: count, dtype: float32




```python
predictions_new_features.describe()
predictions_new_features[predictions_new_features < 0] = 0
```


```python
submission_new_features = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "Two new features (hours & Weekday)"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 324kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission_new_features.csv  2023-06-01 14:43:01  Two new features (hours & Weekday)                                                                  complete  0.58036      0.58036       
    submission.csv               2023-06-01 14:39:34  Initial Submission                                                                                  complete  2.06263      2.06263       
    submission_new_hyp_3.csv     2023-06-01 07:29:37  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  complete  0.50079      0.50079       
    submission_new_hyp_2.csv     2023-06-01 07:29:35  new features with hyperparameters epoch, boost round, learning rate, extra trees                    complete  0.65221      0.65221       
    tail: write error: Broken pipe


#### New Score of 0.58036      

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.


```python
hyperparameters_1 = {
    "NN_TORCH": {
        "num_epochs": 100
    },  
    "GBM": {
        "num_boost_round": 1000
    },  
}
```


```python
hyperparameters_2 = {
    "NN_TORCH": {
        "num_epochs": 100,
        "learning_rate": 1e-5,
    },
    "GBM": {
        "num_boost_round": 1000,
        "extra_trees": True,
    },
}

```


```python
hyperparameters_3 = {  
    "GBM": {"extra_trees": True, "num_boost_round": 1000, "num_leaves": 5},
    "NN_TORCH": {"num_epochs": 100, "learning_rate": 1e-5, "dropout_prob": 0.05},
}
```


```python
predictor_new_hp_1 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_1.fit(
    train_data=train,
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_1,
    refit_full="best",
)

predictor_new_hp_1.fit_summary()

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230601_144303/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 120s
    AutoGluon will save models to "AutogluonModels/ag-20230601_144303/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2552.53 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.12s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 79.9s of the 119.88s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-61.5645	 = Validation score   (-root_mean_squared_error)
    	20.44s	 = Training   runtime
    	1.53s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 55.73s of the 95.71s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-86.1104	 = Validation score   (-root_mean_squared_error)
    	60.66s	 = Training   runtime
    	0.18s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.88s of the 31.48s of remaining time.
    	-61.5134	 = Validation score   (-root_mean_squared_error)
    	0.17s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 31.25s of the 31.24s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-63.3117	 = Validation score   (-root_mean_squared_error)
    	14.68s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 12.8s of the 12.8s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-65.4893	 = Validation score   (-root_mean_squared_error)
    	27.25s	 = Training   runtime
    	0.39s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 119.88s of the -17.93s of remaining time.
    	-62.9094	 = Validation score   (-root_mean_squared_error)
    	0.17s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 138.28s ... Best model: "WeightedEnsemble_L2"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	1.59s	 = Training   runtime
    Fitting 1 L1 models ...
    Fitting model: NeuralNetTorch_BAG_L1_FULL ...
    	6.85s	 = Training   runtime
    Fitting model: WeightedEnsemble_L2_FULL | Skipping fit via cloning parent ...
    	0.17s	 = Training   runtime
    Refit complete, total runtime = 10.5s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230601_144303/")


    *** Summary of fit() ***
    Estimated performance of each model:
                            model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         WeightedEnsemble_L2 -61.513434       1.707213   81.267027                0.000782           0.166178            2       True          3
    1             LightGBM_BAG_L1 -61.564474       1.525321   20.440489                1.525321          20.440489            1       True          1
    2         WeightedEnsemble_L3 -62.909433       2.188551  123.195870                0.000821           0.168333            3       True          6
    3             LightGBM_BAG_L2 -63.311743       1.793295   95.779210                0.086864          14.678361            2       True          4
    4       NeuralNetTorch_BAG_L2 -65.489339       2.100866  108.349176                0.394434          27.248328            2       True          5
    5       NeuralNetTorch_BAG_L1 -86.110372       0.181111   60.660360                0.181111          60.660360            1       True          2
    6    WeightedEnsemble_L2_FULL        NaN            NaN    8.604759                     NaN           0.166178            2       True          9
    7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN    6.851289                     NaN           6.851289            1       True          8
    8        LightGBM_BAG_L1_FULL        NaN            NaN    1.587292                     NaN           1.587292            1       True          7
    Number of models trained: 9
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144303/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2_FULL': 'WeightedEnsembleModel'},
     'model_performance': {'LightGBM_BAG_L1': -61.5644737203475,
      'NeuralNetTorch_BAG_L1': -86.11037239063127,
      'WeightedEnsemble_L2': -61.51343352900248,
      'LightGBM_BAG_L2': -63.311743480440065,
      'NeuralNetTorch_BAG_L2': -65.4893394423414,
      'WeightedEnsemble_L3': -62.90943306224446,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L1_FULL/',
      'WeightedEnsemble_L2_FULL': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 20.440488576889038,
      'NeuralNetTorch_BAG_L1': 60.66035985946655,
      'WeightedEnsemble_L2': 0.16617822647094727,
      'LightGBM_BAG_L2': 14.678361177444458,
      'NeuralNetTorch_BAG_L2': 27.24832773208618,
      'WeightedEnsemble_L3': 0.1683330535888672,
      'LightGBM_BAG_L1_FULL': 1.587292194366455,
      'NeuralNetTorch_BAG_L1_FULL': 6.8512890338897705,
      'WeightedEnsemble_L2_FULL': 0.16617822647094727},
     'model_pred_times': {'LightGBM_BAG_L1': 1.5253205299377441,
      'NeuralNetTorch_BAG_L1': 0.18111109733581543,
      'WeightedEnsemble_L2': 0.0007817745208740234,
      'LightGBM_BAG_L2': 0.08686351776123047,
      'NeuralNetTorch_BAG_L2': 0.39443445205688477,
      'WeightedEnsemble_L3': 0.0008211135864257812,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2_FULL': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                         model  score_val  pred_time_val    fit_time  \
     0         WeightedEnsemble_L2 -61.513434       1.707213   81.267027   
     1             LightGBM_BAG_L1 -61.564474       1.525321   20.440489   
     2         WeightedEnsemble_L3 -62.909433       2.188551  123.195870   
     3             LightGBM_BAG_L2 -63.311743       1.793295   95.779210   
     4       NeuralNetTorch_BAG_L2 -65.489339       2.100866  108.349176   
     5       NeuralNetTorch_BAG_L1 -86.110372       0.181111   60.660360   
     6    WeightedEnsemble_L2_FULL        NaN            NaN    8.604759   
     7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN    6.851289   
     8        LightGBM_BAG_L1_FULL        NaN            NaN    1.587292   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000782           0.166178            2       True   
     1                1.525321          20.440489            1       True   
     2                0.000821           0.168333            3       True   
     3                0.086864          14.678361            2       True   
     4                0.394434          27.248328            2       True   
     5                0.181111          60.660360            1       True   
     6                     NaN           0.166178            2       True   
     7                     NaN           6.851289            1       True   
     8                     NaN           1.587292            1       True   
     
        fit_order  
     0          3  
     1          1  
     2          6  
     3          4  
     4          5  
     5          2  
     6          9  
     7          8  
     8          7  }




```python
predictor_new_hp_2 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_2.fit(
    train_data=train,
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_2,
    refit_full="best",
)

predictor_new_hp_2.fit_summary()

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230601_144532/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 120s
    AutoGluon will save models to "AutogluonModels/ag-20230601_144532/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2555.8 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.1s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 79.92s of the 119.9s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-63.3456	 = Validation score   (-root_mean_squared_error)
    	21.84s	 = Training   runtime
    	2.1s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 54.35s of the 94.34s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-148.0209	 = Validation score   (-root_mean_squared_error)
    	60.14s	 = Training   runtime
    	0.14s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.9s of the 30.75s of remaining time.
    	-63.3456	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 30.52s of the 30.51s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-64.04	 = Validation score   (-root_mean_squared_error)
    	18.14s	 = Training   runtime
    	0.8s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 7.9s of the 7.9s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-320.0707	 = Validation score   (-root_mean_squared_error)
    	23.65s	 = Training   runtime
    	0.14s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 119.9s of the -19.08s of remaining time.
    	-64.04	 = Validation score   (-root_mean_squared_error)
    	0.17s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 139.43s ... Best model: "WeightedEnsemble_L2"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	1.54s	 = Training   runtime
    Refit complete, total runtime = 3.95s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230601_144532/")


    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0        LightGBM_BAG_L1  -63.345619       2.101660   21.840686                2.101660          21.840686            1       True          1
    1    WeightedEnsemble_L2  -63.345619       2.102354   22.002602                0.000694           0.161916            2       True          3
    2        LightGBM_BAG_L2  -64.039953       3.039567  100.124812                0.795607          18.140151            2       True          4
    3    WeightedEnsemble_L3  -64.039953       3.040397  100.289885                0.000830           0.165072            3       True          6
    4  NeuralNetTorch_BAG_L1 -148.020895       0.142299   60.143975                0.142299          60.143975            1       True          2
    5  NeuralNetTorch_BAG_L2 -320.070707       2.388217  105.632385                0.144257          23.647723            2       True          5
    6   LightGBM_BAG_L1_FULL         NaN            NaN    1.541076                     NaN           1.541076            1       True          7
    Number of models trained: 7
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144532/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB'},
     'model_performance': {'LightGBM_BAG_L1': -63.345618670334574,
      'NeuralNetTorch_BAG_L1': -148.02089469234645,
      'WeightedEnsemble_L2': -63.345618670334574,
      'LightGBM_BAG_L2': -64.03995275810236,
      'NeuralNetTorch_BAG_L2': -320.0707074589126,
      'WeightedEnsemble_L3': -64.03995275810236,
      'LightGBM_BAG_L1_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144532/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144532/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144532/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144532/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L1_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 21.840686082839966,
      'NeuralNetTorch_BAG_L1': 60.143975496292114,
      'WeightedEnsemble_L2': 0.16191577911376953,
      'LightGBM_BAG_L2': 18.140150785446167,
      'NeuralNetTorch_BAG_L2': 23.64772319793701,
      'WeightedEnsemble_L3': 0.16507220268249512,
      'LightGBM_BAG_L1_FULL': 1.5410757064819336},
     'model_pred_times': {'LightGBM_BAG_L1': 2.1016604900360107,
      'NeuralNetTorch_BAG_L1': 0.14229941368103027,
      'WeightedEnsemble_L2': 0.0006935596466064453,
      'LightGBM_BAG_L2': 0.7956070899963379,
      'NeuralNetTorch_BAG_L2': 0.1442568302154541,
      'WeightedEnsemble_L3': 0.0008304119110107422,
      'LightGBM_BAG_L1_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0        LightGBM_BAG_L1  -63.345619       2.101660   21.840686   
     1    WeightedEnsemble_L2  -63.345619       2.102354   22.002602   
     2        LightGBM_BAG_L2  -64.039953       3.039567  100.124812   
     3    WeightedEnsemble_L3  -64.039953       3.040397  100.289885   
     4  NeuralNetTorch_BAG_L1 -148.020895       0.142299   60.143975   
     5  NeuralNetTorch_BAG_L2 -320.070707       2.388217  105.632385   
     6   LightGBM_BAG_L1_FULL         NaN            NaN    1.541076   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                2.101660          21.840686            1       True   
     1                0.000694           0.161916            2       True   
     2                0.795607          18.140151            2       True   
     3                0.000830           0.165072            3       True   
     4                0.142299          60.143975            1       True   
     5                0.144257          23.647723            2       True   
     6                     NaN           1.541076            1       True   
     
        fit_order  
     0          1  
     1          3  
     2          4  
     3          6  
     4          2  
     5          5  
     6          7  }




```python
predictor_new_hp_3 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_3.fit(
    train_data=train,
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_3,
    refit_full="best",
)

predictor_new_hp_3.fit_summary()
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230601_144755/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 120s
    AutoGluon will save models to "AutogluonModels/ag-20230601_144755/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2547.57 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.1s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 79.91s of the 119.9s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-72.9884	 = Validation score   (-root_mean_squared_error)
    	15.43s	 = Training   runtime
    	0.87s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 61.31s of the 101.3s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-147.0264	 = Validation score   (-root_mean_squared_error)
    	65.32s	 = Training   runtime
    	0.37s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.9s of the 32.53s of remaining time.
    	-72.9884	 = Validation score   (-root_mean_squared_error)
    	0.19s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 32.27s of the 32.26s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-68.5594	 = Validation score   (-root_mean_squared_error)
    	15.73s	 = Training   runtime
    	0.76s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 12.78s of the 12.77s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-275.5854	 = Validation score   (-root_mean_squared_error)
    	26.71s	 = Training   runtime
    	0.13s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 119.9s of the -17.44s of remaining time.
    	-68.5594	 = Validation score   (-root_mean_squared_error)
    	0.17s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 137.79s ... Best model: "WeightedEnsemble_L3"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	0.6s	 = Training   runtime
    Fitting 1 L1 models ...
    Fitting model: NeuralNetTorch_BAG_L1_FULL ...
    	7.8s	 = Training   runtime
    Fitting 1 L2 models ...
    Fitting model: LightGBM_BAG_L2_FULL ...
    	0.6s	 = Training   runtime
    Refit complete, total runtime = 10.88s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230601_144755/")


    *** Summary of fit() ***
    Estimated performance of each model:
                            model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0             LightGBM_BAG_L2  -68.559370       2.007048   96.484189                0.763687          15.727599            2       True          4
    1         WeightedEnsemble_L3  -68.559370       2.007807   96.651094                0.000760           0.166905            3       True          6
    2             LightGBM_BAG_L1  -72.988425       0.868441   15.433599                0.868441          15.433599            1       True          1
    3         WeightedEnsemble_L2  -72.988425       0.869536   15.625655                0.001095           0.192056            2       True          3
    4       NeuralNetTorch_BAG_L1 -147.026409       0.374919   65.322992                0.374919          65.322992            1       True          2
    5       NeuralNetTorch_BAG_L2 -275.585425       1.372813  107.470856                0.129453          26.714266            2       True          5
    6  NeuralNetTorch_BAG_L1_FULL         NaN            NaN    7.803822                     NaN           7.803822            1       True          8
    7        LightGBM_BAG_L2_FULL         NaN            NaN    9.006895                     NaN           0.599314            2       True          9
    8        LightGBM_BAG_L1_FULL         NaN            NaN    0.603759                     NaN           0.603759            1       True          7
    Number of models trained: 9
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144755/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'LightGBM_BAG_L2_FULL': 'StackerEnsembleModel_LGB'},
     'model_performance': {'LightGBM_BAG_L1': -72.98842476114623,
      'NeuralNetTorch_BAG_L1': -147.02640928935065,
      'WeightedEnsemble_L2': -72.98842476114623,
      'LightGBM_BAG_L2': -68.55937007326823,
      'NeuralNetTorch_BAG_L2': -275.58542499426017,
      'WeightedEnsemble_L3': -68.55937007326823,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144755/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144755/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L1_FULL/',
      'LightGBM_BAG_L2_FULL': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 15.433598518371582,
      'NeuralNetTorch_BAG_L1': 65.32299184799194,
      'WeightedEnsemble_L2': 0.19205641746520996,
      'LightGBM_BAG_L2': 15.727598667144775,
      'NeuralNetTorch_BAG_L2': 26.714265823364258,
      'WeightedEnsemble_L3': 0.16690540313720703,
      'LightGBM_BAG_L1_FULL': 0.6037588119506836,
      'NeuralNetTorch_BAG_L1_FULL': 7.803822040557861,
      'LightGBM_BAG_L2_FULL': 0.5993137359619141},
     'model_pred_times': {'LightGBM_BAG_L1': 0.868441104888916,
      'NeuralNetTorch_BAG_L1': 0.37491941452026367,
      'WeightedEnsemble_L2': 0.0010950565338134766,
      'LightGBM_BAG_L2': 0.7636873722076416,
      'NeuralNetTorch_BAG_L2': 0.12945270538330078,
      'WeightedEnsemble_L3': 0.0007596015930175781,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                         model   score_val  pred_time_val    fit_time  \
     0             LightGBM_BAG_L2  -68.559370       2.007048   96.484189   
     1         WeightedEnsemble_L3  -68.559370       2.007807   96.651094   
     2             LightGBM_BAG_L1  -72.988425       0.868441   15.433599   
     3         WeightedEnsemble_L2  -72.988425       0.869536   15.625655   
     4       NeuralNetTorch_BAG_L1 -147.026409       0.374919   65.322992   
     5       NeuralNetTorch_BAG_L2 -275.585425       1.372813  107.470856   
     6  NeuralNetTorch_BAG_L1_FULL         NaN            NaN    7.803822   
     7        LightGBM_BAG_L2_FULL         NaN            NaN    9.006895   
     8        LightGBM_BAG_L1_FULL         NaN            NaN    0.603759   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.763687          15.727599            2       True   
     1                0.000760           0.166905            3       True   
     2                0.868441          15.433599            1       True   
     3                0.001095           0.192056            2       True   
     4                0.374919          65.322992            1       True   
     5                0.129453          26.714266            2       True   
     6                     NaN           7.803822            1       True   
     7                     NaN           0.599314            2       True   
     8                     NaN           0.603759            1       True   
     
        fit_order  
     0          4  
     1          6  
     2          1  
     3          3  
     4          2  
     5          5  
     6          8  
     7          9  
     8          7  }




```python
# Leaderboard dataframe
leaderboard_new_hp_df_1 = pd.DataFrame(predictor_new_hp_1.leaderboard(silent=True))
leaderboard_new_hp_df_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L2</td>
      <td>-61.513434</td>
      <td>1.707213</td>
      <td>81.267027</td>
      <td>0.000782</td>
      <td>0.166178</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_BAG_L1</td>
      <td>-61.564474</td>
      <td>1.525321</td>
      <td>20.440489</td>
      <td>1.525321</td>
      <td>20.440489</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WeightedEnsemble_L3</td>
      <td>-62.909433</td>
      <td>2.188551</td>
      <td>123.195870</td>
      <td>0.000821</td>
      <td>0.168333</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LightGBM_BAG_L2</td>
      <td>-63.311743</td>
      <td>1.793295</td>
      <td>95.779210</td>
      <td>0.086864</td>
      <td>14.678361</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-65.489339</td>
      <td>2.100866</td>
      <td>108.349176</td>
      <td>0.394434</td>
      <td>27.248328</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-86.110372</td>
      <td>0.181111</td>
      <td>60.660360</td>
      <td>0.181111</td>
      <td>60.660360</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.604759</td>
      <td>NaN</td>
      <td>0.166178</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.851289</td>
      <td>NaN</td>
      <td>6.851289</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.587292</td>
      <td>NaN</td>
      <td>1.587292</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Leaderboard dataframe
leaderboard_new_hp_df_2 = pd.DataFrame(predictor_new_hp_2.leaderboard(silent=True))
leaderboard_new_hp_df_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LightGBM_BAG_L1</td>
      <td>-63.345619</td>
      <td>2.101660</td>
      <td>21.840686</td>
      <td>2.101660</td>
      <td>21.840686</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WeightedEnsemble_L2</td>
      <td>-63.345619</td>
      <td>2.102354</td>
      <td>22.002602</td>
      <td>0.000694</td>
      <td>0.161916</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L2</td>
      <td>-64.039953</td>
      <td>3.039567</td>
      <td>100.124812</td>
      <td>0.795607</td>
      <td>18.140151</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WeightedEnsemble_L3</td>
      <td>-64.039953</td>
      <td>3.040397</td>
      <td>100.289885</td>
      <td>0.000830</td>
      <td>0.165072</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-148.020895</td>
      <td>0.142299</td>
      <td>60.143975</td>
      <td>0.142299</td>
      <td>60.143975</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-320.070707</td>
      <td>2.388217</td>
      <td>105.632385</td>
      <td>0.144257</td>
      <td>23.647723</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.541076</td>
      <td>NaN</td>
      <td>1.541076</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Leaderboard dataframe
leaderboard_new_hp_df_3 = pd.DataFrame(predictor_new_hp_3.leaderboard(silent=True))
leaderboard_new_hp_df_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LightGBM_BAG_L2</td>
      <td>-68.559370</td>
      <td>2.007048</td>
      <td>96.484189</td>
      <td>0.763687</td>
      <td>15.727599</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WeightedEnsemble_L3</td>
      <td>-68.559370</td>
      <td>2.007807</td>
      <td>96.651094</td>
      <td>0.000760</td>
      <td>0.166905</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L1</td>
      <td>-72.988425</td>
      <td>0.868441</td>
      <td>15.433599</td>
      <td>0.868441</td>
      <td>15.433599</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WeightedEnsemble_L2</td>
      <td>-72.988425</td>
      <td>0.869536</td>
      <td>15.625655</td>
      <td>0.001095</td>
      <td>0.192056</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-147.026409</td>
      <td>0.374919</td>
      <td>65.322992</td>
      <td>0.374919</td>
      <td>65.322992</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-275.585425</td>
      <td>1.372813</td>
      <td>107.470856</td>
      <td>0.129453</td>
      <td>26.714266</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.803822</td>
      <td>NaN</td>
      <td>7.803822</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LightGBM_BAG_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.006895</td>
      <td>NaN</td>
      <td>0.599314</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.603759</td>
      <td>NaN</td>
      <td>0.603759</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
leaderboard_new_hp_df_1.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_2.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_3.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
plt.show()
```


    
![png](output_68_0.png)
    



    
![png](output_68_1.png)
    



    
![png](output_68_2.png)
    



```python
predictions_new_hyp_1 = predictor_new_hp_1.predict(test)
predictions_new_hyp_1.head()
```




    0    21.584213
    1     4.534583
    2     0.507605
    3     1.623464
    4     1.280358
    Name: count, dtype: float32




```python
predictions_new_hyp_2 = predictor_new_hp_2.predict(test)
predictions_new_hyp_2.head()
```




    0    29.971573
    1     8.629538
    2     4.112672
    3    -1.318882
    4    -1.364102
    Name: count, dtype: float32




```python
predictions_new_hyp_3 = predictor_new_hp_3.predict(test)
predictions_new_hyp_3.head()
```




    0    24.633934
    1     8.686478
    2     8.407059
    3     8.726441
    4     8.746774
    Name: count, dtype: float32




```python
predictions_new_hyp_1[predictions_new_hyp_1<0] = 0 
predictions_new_hyp_2[predictions_new_hyp_2<0] = 0 
predictions_new_hyp_3[predictions_new_hyp_3<0] = 0 
```


```python
submission_new_hyp_1 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_1["count"] = predictions_new_hyp_1
submission_new_hyp_1.to_csv("submission_new_hyp_1.csv", index=False)

submission_new_hyp_2 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_2["count"] = predictions_new_hyp_2
submission_new_hyp_2.to_csv("submission_new_hyp_2.csv", index=False)

submission_new_hyp_3 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_3["count"] = predictions_new_hyp_3
submission_new_hyp_3.to_csv("submission_new_hyp_3.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_1.csv -m "new features with hyperparameters epoch, boost round"
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_2.csv -m "new features with hyperparameters epoch, boost round, learning rate, extra trees"
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_3.csv -m "new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves"
```

    100%|█████████████████████████████████████████| 187k/187k [00:00<00:00, 345kB/s]
    100%|█████████████████████████████████████████| 187k/187k [00:00<00:00, 394kB/s]
    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 342kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission_new_hyp_3.csv     2023-06-01 14:50:49  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  pending                              
    submission_new_hyp_2.csv     2023-06-01 14:50:46  new features with hyperparameters epoch, boost round, learning rate, extra trees                    pending                              
    submission_new_hyp_1.csv     2023-06-01 14:50:44  new features with hyperparameters epoch, boost round                                                complete  0.60783      0.60783       
    submission_new_features.csv  2023-06-01 14:43:01  Two new features (hours & Weekday)                                                                  complete  0.58036      0.58036       
    tail: write error: Broken pipe


#### New Score

- Hy1:  0.60933
- Hy2:  0.65221
- Hy3:  0.50079      

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report


```python
fig = (
    pd.DataFrame(
        {
            "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [61.018598, 59.963984, 61.523895, 63.345619, 68.541952],
        }
    )
    .plot(x="model", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_train_score.png")

```


    
![png](output_78_0.png)
    



```python
fig = (
    pd.DataFrame(
        {
            "test_eval": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [2.08327, 0.58036, 0.73551, 0.65221, 0.50079],
        }
    )
    .plot(x="test_eval", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_test_score.png")
```


    
![png](output_79_0.png)
    


### Hyperparameter table


```python
pd.DataFrame({
    "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
    "hpo1": ["default", "default", "epoch, boost round", "epoch, boost round", "epoch, boost round"],
    "hpo2": ["default", "default", "default", "learning rate, extra trees", "learning rate, extra trees"],
    "hpo3": ["default", "default", "default", "default", "drop-out, leaves"],
    "score": [2.08327, 0.58036, 0.73551, 0.65221, 0.50079]
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>hpo1</th>
      <th>hpo2</th>
      <th>hpo3</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>initial</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>2.08327</td>
    </tr>
    <tr>
      <th>1</th>
      <td>add_features</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>0.58036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hp1</td>
      <td>epoch, boost round</td>
      <td>default</td>
      <td>default</td>
      <td>0.73551</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hp2</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>default</td>
      <td>0.65221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hp3</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>drop-out, leaves</td>
      <td>0.50079</td>
    </tr>
  </tbody>
</table>
</div>


