## ACSeS: Ant Colony Semantic Search of Vector Encoded Documents ##

### Installation: ###
The prerequisites for this project are as follows:
1. Anaconda or Miniconda
2. A CUDA Compatible GPU (optional for tensor acceleration)

To install SharedSight from source, do the following:
1. Run the following command:
```bash
conda env create -f env.yml
```
2. Download and extract the dataset from the [Huffington News Caption Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
3. Activate the ```shared_sight``` conda environment
4. Set up the search indices, embedding banks, and search metadata by running the following script (Make sure to input the file to store the embedding vectors in and the dataset path):
```bash
python store_visualize.py
```

### Usage: ###
Run the ``main.py`` script with the embedding pickle file in the "-i" command, with any other ACO parameters specified afterwards.
