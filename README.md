# Chlamy_Project_v2

This repository contains the code for analyzing data collected by Dr. Adrien Burlacot's lab.

## Repository Structure

- **Scripts**: Contains the main Python (`.py`) files with functions and tools essential for data analysis. These scripts are called within the notebooks.
  
- **Notebooks**: Contains Jupyter notebooks used for exploratory data analysis. While most notebooks focus on various aspects of the data and are exploratory in nature (and rather unstructured), the primary results—including gene extraction and clustering—are generated in the `gene_clustering` notebook.

## Data Processing and Dependencies

The `gene_clustering` notebook relies on preprocessed data files, which are not included in this repository due to file size limitations. However, you can generate this data using the `get_format_data_without_na` function in the `Data_v2.py` script (located in the Scripts folder). This function accepts arguments (`phase1`, `phase2`, or `transition`) to specify the data to load.
