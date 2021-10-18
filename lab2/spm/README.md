# Spatial Pyramid Matching

## Dataset

1. Download the dataset from the link [Caltech-101] (http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download)

2. Extract the data from the ``tar.gz`` file and rename the folder as ``data``

## Dependency

```shell
conda install --file requirements.txt -c conda-forge
```

## Run

```shell
python run.py --data <path_to_data_folder> --output <path_to_output_folder>
```
Default parameters:
- ``data``: ``"./data"``
- ``output``: ``"./"``


## References

1. [Bag of Features](https://ai.stackexchange.com/questions/21914/what-are-bag-of-features-in-computer-vision)

2. [Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories](https://inc.ucsd.edu/mplab/users/marni/Igert/Lazebnik_06.pdf)