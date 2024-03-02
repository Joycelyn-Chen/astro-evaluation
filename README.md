# Astro Semi-supervised evaluation package

This package is used to evaluate semi-supervised video single-object segmentation models for the <a href="" target="_blank">Astro</a> dataset. 


### Installation
```bash
# Download the code
git clone https://github.com/Joycelyn-Chen/astro-evaluation && cd astro-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```
If you don't want to specify the ASTRO path every time, you can modify the default value in the variable `default_astro_path` in `evaluation_method.py`(the following examples assume that you have set it). 
Otherwise, you can specify the path in every call using using the flag `--astro_path /path/to/ASTRO` when calling `evaluation_method.py`.

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results: 
- `global_results-SUBSET.csv` contains the overall results for a certain `SUBSET`. 
- `per-sequence_results-SUBSET.csv` contain the per sequence results for a certain `SUBSET`.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Evaluate Astro Semi-supervised
In order to evaluate your semi-supervised method in DAVIS 2017, execute the following command substituting `results/semi-supervised/astro` by the folder path that contains your results:
```bash
python evaluation_method.py --task semi-supervised --results_path ../results/astro --astro_path /path/to/ASTRO
```



## Citation

Please cite this paper in your publications if Astro or this code helps your research.

```latex

```

Contact joycelyn.chen@ubc.ca for astro dataset download.  


