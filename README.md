# Patient Prediction System fyp

This web application is designed to help users predict their risk of developing stroke and diabetes. By entering personal information such as lifestyle factors, users can receive a personalized risk assessment and information on prevention strategies.

## Getting Started
To run the application, you will need to have Python 3 installed on your computer. You can install the necessary packages by running the following command in your terminal:

pip install -r requirements.txt
Once the packages are installed, you can launch the application by running the following command:

streamlit run app.py
The application will open in your web browser.

Or you can directly use deployed demo application by: https://cindy0627-fyp-main-0uidmz.streamlit.app/

## Usage
To use the application, simply enter your personal information into the form after clicking consent agreement. You can select either stroke or diabetes as the disease you want to predict.

Once you have entered your information, click the "Predict" button to receive your risk assessment. The application will display your predicted risk level along with information on prevention strategies and resources for accessing healthcare services.

## Model Details
The stroke and diabetes prediction models were built using data collected by the National Health and Nutrition Examination Survey (NHANES) 2013-2014. NHANES is a program of studies designed to assess the health and nutritional status of adults and children in the United States.

The NHANES 2013-2014 dataset includes a wide range of demographic, lifestyle, and clinical data, including information on medical history, physical exam results, and laboratory tests. However, the features used to train the models are all daily-life based data. These features were chosen to ensure that individuals can easily provide the necessary information to receive a risk assessment, without needing to undergo laboratory testing or other invasive procedures.

The models were trained using a combination of machine learning algorithms, including random forest, SVM, and logistic regression. The models were validated using rigorous testing procedures to ensure accuracy and reliability.

