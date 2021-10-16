# Multi-class classification of Consumer Complaints on Financial Products: An Analysis with  Multinomial Naive Bayes and XGBoost

## Machine Learning Engineer Nanodegree - Udacity
<br>
The data used in this project contain complaints that have been made by consumers regarding financial services and products (e.g. student loans, credit reports, etc) in the US from 2011 to the current date. Each of the complaints is marked to belong under one Product category. This makes the data ideal for supervised learning purposes, with the text (complaint) as the input, and the category that the complaint belongs to as the target variable.<br><br>

### Data Retrieval Process

This is a public dataset and can be found in the following location: https://catalog.data.gov/dataset/consumer-complaint-database.

Please note that the dataset is keep getting updated daily, and the latest version can be found here: https://files.consumerfinance.gov/ccdb/complaints.csv.zip.
(Note: If you click on this link it will automatically start to download the dataset)

Because of the constant updates on this dataset and it's increase in the size, we are going to use a _subset_ of the dataset. Specifically, we are going to use the complaints that have been made by consumers between January 2019 to December 2020. This will allow us to focus on specific years of interest, while keeping the observations in the dataset (and thus the size) to some logical ranges so that we can work with them on AWS.

**Step 0.**

Make sure you have installed the following Python packages:
```txt
wget==3.2
pandas==1.3.0
zipfile==3.7.6
```
**Step 1.**

To retrieve the <em>whole</em> dataset in your local directory, please run the following Python script:

```python
import os
import wget
import zipfile

data_out_path_dir = "/Users/georgiosspyrou/Desktop" # !Change this to the path where data will be saved in your local machine

data_web_location = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
full_data_csv_name = 'complaints.csv'

print("Dataset will be saved in '{0}'".format(data_out_path_dir))

if not os.path.exists(data_out_path_dir):
    os.makedirs(data_out_path_dir)

if not os.path.isfile(os.path.join(data_out_path_dir, full_data_csv_name)):
    zip_filename = wget.download(url=data_web_location, out=data_out_path_dir)

    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        print("\nExtracting files from {0} to {1}\n".format(zip_filename, data_out_path_dir))
        zipf.extractall(data_out_path_dir)
    
    print("Removing file {0} ...".format(zip_filename))
    os.remove(zip_filename) 

```

Please note that the output file from this will be around 1.5GB.

**Step 2.** 

If you further want to filter out the data to the specific complaints that were conducted between January 2019 and December 2020 (which will be the main dataset of this project), please run the script in Step 1 to retrieve the data, and then run the following Python script:

```python
import os
import pandas as pd

csv_file_path = os.path.join(data_out_path_dir, full_data_csv_name)
complaints_df = pd.read_csv(csv_file_path)

complaints_df['Date received'] = pd.to_datetime(complaints_df['Date received'])

# Get the year that the complaint took place as a separate column
complaints_df['Year'] = complaints_df['Date received'].apply(lambda date: date.year)

# Define a function to help reduce the dataset by date
def subset_dataframe_on_date_column(
    df: pd.DataFrame,
    date_col: str,
    min_date: str,
    max_date: str
) -> pd.DataFrame:
    reduced_df = df[(df[date_col] >= min_date) & (df[date_col] <= max_date)]
    return reduced_df

complaints_df = subset_dataframe_on_date_column(
    df=complaints_df,
    date_col='Date received', 
    min_date='2019-01-01',
    max_date='2020-12-31'
    )

reduced_csv_filename = 'complaints_reduced.csv'
reduced_csv_filepath = os.path.join(data_out_path_dir, reduced_csv_filename)

# Save the reduced dataset locally
complaints_df.to_csv(reduced_csv_filepath, index=False)
```

Please note that the output file from this will be around 550MB.
