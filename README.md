# Patient Prediction System fyp

This web application is designed to help users predict their risk of developing stroke and diabetes. By entering personal information such as lifestyle factors, users can receive a personalized risk assessment and information on prevention strategies.

## File Structure
- `comparing_diabetes_models.ipynb`: Jupyter Notebook containing code for comparing and evaluating six diabetes models. (default parameter and after parameter tuning)
- `comparing_stroke_models.ipynb`: Jupyter Notebook containing code for comparing and evaluating six stroke models. (default parameter and after parameter tuning)
- `data_preprocessing.ipynb`: Jupyter Notebook containing code for preprocessing the diabetes and stroke datasets.
- `export_diabetes_model.py`: Python script for exporting a optimal diabetes model.
- `export_stroke_model.py`: Python script for exporting a optimal stroke model.
- `final_testset1_diabetes.csv`: CSV file containing the final test dataset for the diabetes model. 
- `final_testset1_stroke.csv`: CSV file containing the final test dataset for the stroke model.
- `health.jpg`: Image file used in the web application.
- `lr_clf_dia.pkl`: Pickle file containing a trained logistic regression classifier for the diabetes model. (from export_diabetes_model.py)
- `main.py`: Python script containing the main function for running the diabetes and stroke models. (Streamlit application)
- `minmax_dia.gz`: Gzip file containing the MinMaxScaler object for the diabetes model.
- `minmax_stroke.gz`: Gzip file containing the MinMaxScaler object for the stroke model.
- `requirements.txt`: Text file containing the required Python packages and their versions.
- `rnd_clf_stroke.pkl`: Pickle file containing a trained random forest classifier for the stroke model. (from export_stroke_model.py)
- `all the csv file`: csv file containing the dataset

## Getting Started
To run the application, you will need to have Python 3 installed on your computer. If you do not have Python 3 installed, you can download it from the official Python website: https://www.python.org/downloads/

Once you have Python 3 installed, you can install the necessary packages by following the steps below:

  1. Open your terminal or command prompt.

  2. Navigate to the directory where you have downloaded the project files.

  3. Run the following command to create a virtual streamlit environment:
  ```python
virtualenv streamlitenv
```
  4. Activate the virtual environment by running the following command:
```python
streamlitenv\Scripts\activate
```
  5. Install the necessary packages by running the following command:
```python
pip install -r requirements.txt
```
Once the packages are installed, you can launch the application by following the steps below:

  1. Make sure you are in the project directory and the virtual environment is activated.

  2. Run the following command:
```python
streamlit run main.py
```
  3. The application will open in your web browser.

### Or you can directly use deployed demo application by: [https://cindy0627-fyp-main-0uidmz.streamlit.app/](https://cindy0627-fyp-main-0uidmz.streamlit.app/)

## Usage
To use the application, simply enter your personal information into the form after clicking consent agreement. You can select either stroke or diabetes as the disease you want to predict.

Once you have entered your information, click the "Predict" button to receive your risk assessment. The application will display your predicted risk level along with information on prevention strategies and resources for accessing healthcare services.

## Model Details
The stroke and diabetes prediction models were built using data collected by the National Health and Nutrition Examination Survey (NHANES) 2013-2014. NHANES is a program of studies designed to assess the health and nutritional status of adults and children in the United States.

The NHANES 2013-2014 dataset includes a wide range of demographic, lifestyle, and clinical data, including information on medical history, physical exam results, and laboratory tests. However, the features used to train the models are all daily-life based data. These features were chosen to ensure that individuals can easily provide the necessary information to receive a risk assessment, without needing to undergo laboratory testing or other invasive procedures.

The models were trained using a combination of machine learning algorithms, including random forest, SVM, and logistic regression. The models were validated using different testing procedures to ensure accuracy and reliability.

