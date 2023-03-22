import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# -------SETTINGS---------------
page_title = "Patient Health Prediction"
page_icon = ":hospital:"
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)


# ------CLASS--------
class Stroke:
    # --variable----
    name = "Stroke"
    df = pd.read_csv("final_testset1_stroke.csv")
    target = "MCQ160F"
    model = joblib.load("rnd_clf_stroke.pkl")

    var_string_list = ["MCQ160E", "MCQ365A", "PFQ090", "OHQ860", "CDQ010", "BPQ020", "OHQ614", "MCQ203", "HSD010",
                       "PFQ051", "CSQ030", "SLD010H", "INQ030", "MCQ370B", "PFQ061R", "DUQ200", "OCQ180", "DLQ040",
                       "DBD900", "FSDAD"]

    health_questions = {
        "MCQ160E": "Has a doctor or other health professional ever told you that you had a heart attack?",
        "MCQ365A": "During the past 12 months have you ever been told by a doctor to control your weight?",
        "PFQ090": "Do you now have any health problem that requires you to use special equipment?",
        "OHQ860": "Have you ever been told by a dentist that you lost bone around your teeth?",
        "CDQ010": "Have you had shortness of breath either when hurrying on the level?",
        "BPQ020": "Have you ever been told by a doctor that you had high blood pressure?",
        "OHQ614": "Did a dentist told you about the importance of examining your mouth for oral cancer?",
        "MCQ203": "Has anyone ever told you that you had jaundice? Please do not include infant jaundice.",
        "HSD010": "Would you say your health in general is : (1: Excellent, 2:Very good, 3:Good, 4:Fair, 5:Poor)"
    }

    other_questions = {
        "PFQ051": "Are you limited in the kind or amount of work because of a physical or mental problem?",
        "CSQ030": "Do some smells bother you although they do not bother other people?",
        "SLD010H": "How much sleep do you usually get at night on weekdays or workdays? (Answer in hours)",
        "INQ030": "Did you or any family members living here receive income in 2022 from Social Security?",
        "MCQ370B": "Are you now doing any of the following: increasing your physical activity or exercise?",
        "PFQ061R": "How much difficulty do you have participating in social activities [visiting friends, attending clubs or meetings or going to parties? (#1 No difficulty, #2 Some difficulty, #3 Much difficulty, #4 Unable to do, #5 Do not do this activity)]",
        "DUQ200": "Have you ever used marijuana or hashish?",
        "OCQ180": "How many hours did you work last week at all jobs or businesses?",
        "DLQ040": "Do you have serious difficulty concentrating, remembering, or making decisions?",
        "DBD900": "How many of those meals did you get from a fast-food or pizza place a week?",
        "FSDAD": "What is your food security level? (1 for full, 2 for marginal, 3 for low, 4 for very low)"
    }
    special_list = ["HSD010", "SLD010H", "PFQ061R", "OCQ180", "DBD900", "FSDAD", "MCQ160F"]

    minmax = joblib.load("minmax_stroke.gz")


class Diabetes:
    # --variable----
    name = "Diabetes"
    df = pd.read_csv("final_testset1_diabetes.csv")
    model = joblib.load("lr_clf_dia.pkl")
    target = "DIQ010"

    var_string_list = ["BPQ020", "DIQ180", "MCQ300C", "MCQ080", "MCQ365B", "HEQ030", "HSQ520", "OHQ835",
                       "MCQ365C", "INQ020", "CSQ110", "FSD032A", "DUQ200", "MGATHAND", "DBD100", "INQ030", "OCQ260",
                       "FSD032C"]
    special_list = ["FSD032A", "DBD100", "OCQ260", "FSD032C"]

    health_questions = {
        "BPQ020": "Have you ever been told by a doctor that you had high blood pressure?",
        "DIQ180": "Have you had a blood test for high blood sugar within the past three years?",
        "MCQ300C": "Close relative had diabetes?",
        "MCQ080": "Do Doctor ever said you were overweight?",
        "MCQ365B": "Do Doctor told you to exercise?",
        "HEQ030": "Do you ever have Hepatitis C?",
        "HSQ520": "Do you ever have flu, pneumonia, or ear infection?",
        "OHQ835": "Do you think you might have gum disease?"
    }

    other_questions = {
        "MCQ365C": "Have you ever been told by someone to reduce the amount of sodium or salt in your diet?",
        "INQ020": "Did you receive income in 2022 from wages and salaries?",
        "CSQ110": "During the past 12 months have you had a taste in your mouth that does not go away?",
        "FSD032A": "Have you worried whether your food would run out before you got money to buy more? (1 for Often true, 2 for Sometimes true, 3 for Never true)",
        "DUQ200": "Have you ever used marijuana or hashish?",
        "MGATHAND": "Which hand do you often use? (1 for right, 2 for left)",
        "DBD100": "How often do you add ordinary salt to your food at the table? (1 for Rarely, 2 for Occasionally, 3 for Very often)",
        "INQ030": "Did you or any family members living here receive income in 2022 from Social Security?",
        "OCQ260": "Which of these best describes your job or work situation? (1 for employee of a private company, business, or individual for wages, salary, or commission, 2 for federal government employee, 3 for government employee, 4 for local government employee, 5 for Self-employed in own business, professional practice or farm, 6 for Working without pay in family business or farm)",
        "FSD032C": "Have you ever could not afford to eat balanced meals? (1 for Often true, 2 for Sometimes true, 3 for Never true)"
    }

    minmax = joblib.load("minmax_dia.gz")


class Patient:
    list = []  # init the input list
    prediction_result = ""

    def __init__(self, choice):
        self.choice = choice  # 1 for stroke 2 for diabetes

    # -----TO COMBINE the input with variable
    def df_combine(self):
        if self.choice == 1:
            columns = Stroke.var_string_list
        else:
            columns = Diabetes.var_string_list

        df = pd.DataFrame(self.list, columns=columns)
        return df

    # ------TO PERFORM prediction
    def prediction(self, df):

        if self.choice == 1:
            transformData = Stroke.minmax.transform(df)
            prediction = Stroke.model.predict(transformData)
            if prediction[0] == 0:
                self.prediction_result = "negative"
                st.subheader("Congratulations! You are possibly not having a stroke.")
            else:
                self.prediction_result = "positive"
                st.subheader("You may having a risk of getting stroke! ")
            st.markdown("""---""")

        else:
            transformData = Diabetes.minmax.transform(df)
            prediction = Diabetes.model.predict(transformData)
            if prediction[0] == 0:
                self.prediction_result = "negative"
                st.subheader("Congratulations! You are possibly not having a diabetes.")
                st.header(":relieved:")

            else:
                self.prediction_result = "positive"
                st.markdown("<h1 style='text-align: center; color: grey;'>"
                            "You may having a risk of getting diabetes! "
                            "</h1>", unsafe_allow_html=True)
            st.markdown("""---""")


# ------------------------------

# ------FUNCTION-----------
def BackgroundInfo():
    st.header(f"Background Information")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            "Chronic diseases are a major public health concern in Hong Kong. According to the Department of Health, chronic diseases such as diabetes, hypertension, and cardiovascular diseases are the leading causes of death in the city. ")
        st.caption(
            "In addition, the prevalence of these diseases has been increasing over the years due to an aging population, unhealthy lifestyle habits, and the growing burden of obesity. The government has implemented various measures to address this issue, such as promoting healthy living through education and awareness campaigns, providing affordable healthcare services, and encouraging regular health screenings.")
        st.caption("However, there is still much work to be done to combat the rise of chronic diseases in Hong Kong.")
    with col2:
        from PIL import Image
        image = Image.open('health.jpg')
        st.image(image, caption='')


def StrokePredict():
    # Display the stroke prediction model header
    st.header(f"Stroke prediction model")
    # Create a new patient and stroke instance
    user = Patient(1)
    now = Stroke()

    # Display the consent agreement form
    ConsentAgree()
    st.write("")
    st.write("By clicking 'I agree.', you agree to the terms and conditions.")
    check = st.checkbox("I agree.")
    if check:

        with st.form("entry_form"):
            st.subheader("Please fill in the following information:")
            listing = []
            # Display the health-related input fields in an expander
            with st.expander("Health"):
                # Collect health-related input fields
                for key in now.health_questions:
                    if key == "HSD010":
                        listing.append(st.slider(now.health_questions[key], 1, 5, 1, key=key))
                    else:
                        listing.append(1 if st.radio(now.health_questions[key], ('Yes', 'No'), key=key) == 'Yes' else 2)
            # Display the daily-related input fields in an expander
            with st.expander("Daily"):
                # Collect daily-related input fields
                for key in now.other_questions:
                    if key in ["PFQ051", "CSQ030", "INQ030", "MCQ370B", "DUQ200", "DLQ040"]:
                        listing.append(1 if st.radio(now.other_questions[key], ('Yes', 'No'), key=key) == 'Yes' else 2)
                    elif key == "SLD010H":
                        listing.append(st.slider(now.other_questions[key], 2, 12, 2, key=key))
                    elif key == "PFQ061R":
                        listing.append(st.slider(now.other_questions[key], 1, 5, 1, key=key))
                    elif key == "OCQ180":
                        listing.append(st.number_input(now.other_questions[key], min_value=1, max_value=120, key=key))
                    elif key == "DBD900":
                        listing.append(st.slider(now.other_questions[key], 1, 21, 1, key=key))
                    elif key == "FSDAD":
                        listing.append(st.slider(now.other_questions[key], 1, 4, 1, key=key))
            # save user inputs
            submitted = st.form_submit_button("Predict Stroke Diseases")
            if submitted:
                user.list = [listing]
                st.success("Data saved!")

                # combine user inputs into a dataframe
                df = user.df_combine()
                # predict diabetes diseases
                user.prediction(df)
                VisualData(now)
                Recommend(now.name, user)


def DiabetesPredict():
    # Display the stroke prediction model header
    st.header(f"Diabetes prediction model")
    # Create a new patient and diabetes instance
    user = Patient(2)
    now = Diabetes()

    # Display the consent agreement form
    ConsentAgree()
    st.write("By clicking 'I agree.', you agree to the terms and conditions.")
    check = st.checkbox("I agree.")
    if check:

        with st.form("entry_form", clear_on_submit=False):
            st.subheader("Please fill in the following information:")

            listing = []
            # Display the health-related input fields in an expander
            with st.expander("Health"):
                for key in now.health_questions:
                    listing.append(1 if st.radio(now.health_questions[key], ('Yes', 'No'), key=key) == 'Yes' else 2)

            # Display the daily-related input fields in an expander
            with st.expander("Daily"):
                for key in now.other_questions:
                    if key in ["OCQ260"]:
                        listing.append(st.slider(now.other_questions[key], 1, 6, 1, key=key))
                    elif key in ["FSD032A", "DBD100", "FSD032C"]:
                        listing.append(st.slider(now.other_questions[key], 1, 3, 1, key=key))
                    else:
                        listing.append(1 if st.radio(now.other_questions[key], ('Yes', 'No'), key=key) == 'Yes' else 2)

            # save user inputs
            submitted = st.form_submit_button("Predict Diabetes Diseases")
            if submitted:
                user.list = [listing]
                st.success("Data saved!")
                # combine user inputs into a dataframe
                df = user.df_combine()
                # predict diabetes diseases
                user.prediction(df)
                VisualData(now)
                Recommend(now.name, user)


def InfoQuery():
    # ---------INFO PAGE --------
    st.header(f"More information about health check")
    st.markdown(
        """
        If you're interested in knowing more about your body status and want to perform a body check in Hong Kong, there are many hospitals and clinics that offer comprehensive health screening services. These services can provide you with a detailed assessment of your overall health and help identify any potential health issues early on.

        Some of the hospitals in Hong Kong that offer health screening services include:
        
        -	St. Paul's Hospital: Provides comprehensive health check packages that include blood tests, imaging tests, and consultations with medical professionals.
        -	Queen Mary Hospital: Offers a range of health screening packages, including diabetes screening and stroke risk assessment. The packages include a physical examination, blood tests, and imaging studies.
        -   Prince of Wales Hospital: Offers a variety of health check plans, including diabetes screening and stroke risk assessment. The packages include blood tests, imaging studies, and consultations with specialists.
        
        The types of body checks that are included in these packages vary, but may include:
        -	Blood tests to check for cholesterol, glucose, liver and kidney function, and other markers of overall health.
        -	Imaging tests such as X-rays, ultrasounds, and CT scans to check for any abnormalities in organs or tissues.
        -	Cardiovascular tests such as electrocardiograms (ECGs) and stress tests to assess heart health and detect any potential issues.
        -	Cancer screenings such as mammograms, Pap smears, and colonoscopies to detect cancer early on.
        
        It is important to note that the cost of health check packages can vary depending on the hospital and the level of service provided. It is also recommended to consult with a medical professional to determine which health check package is most appropriate for your individual needs.
        If you're interested in learning more about health screening services in Hong Kong, it is recommended to contact your preferred hospital or clinic directly to inquire about the specific services they offer.

        """
    )
    # Hospital information dictionary
    hospital_info = {
        "St. Paul's Hospital": {
            "Address": "2 Eastern Hospital Road, Causeway Bay, Hong Kong",
            "Phone": "+852 2890 6888"
        },
        "Queen Mary Hospital": {
            "Address": "102 Pok Fu Lam Road, Hong Kong",
            "Phone": "+852 2255 3838"
        },
        "Prince of Wales Hospital": {
            "Address": "30-32 Ngan Shing Street, Sha Tin, New Territories, Hong Kong",
            "Phone": "+852 2632 2211"
        }
    }

    # Main application code
    st.header("Hospital Information")

    # Select hospital from dropdown list
    selected_hospital = st.selectbox("Select a hospital:", list(hospital_info.keys()))

    # Display hospital information
    st.subheader(selected_hospital)
    st.write("Address:", hospital_info[selected_hospital]["Address"])
    st.write("Phone:", hospital_info[selected_hospital]["Phone"])

    #


def ConsentAgree():
    st.header("Privacy and Consent Statement for Online Prediction")
    st.markdown(
        '<div style="text-align: justify;">We are currently conducting a prediction study focused on stroke and diabetes, incorporating inquiries about daily habits. Our primary objective is to enhance public awareness of risk factors for chronic diseases and promote preventive measures. We assure you that all information collected in this study is held in strict confidentiality and will not be disclosed to any third parties. We do not request any personally identifiable data and the input data will not be retained. Your responses will solely serve research purposes and will be reported in an aggregated format.</div>',
        unsafe_allow_html=True)


def get_risk_factors(disease):
    # create a dictionary of daily life habits and their associated risk factors for stroke
    stroke_risk_factors = {
        "smoking": ["increased risk of blood clots"],
        "alcohol consumption": ["increased risk of high blood pressure"],
        "physical inactivity": ["increased risk of obesity and high blood pressure"],
        "poor diet": ["increased risk of high blood pressure and high cholesterol"],
        "stress": ["increased risk of high blood pressure and heart disease"]
    }

    # create a dictionary of daily life habits and their associated risk factors for diabetes
    diabetes_risk_factors = {
        "poor diet": ["increased risk of high blood sugar"],
        "physical inactivity": ["increased risk of obesity and insulin resistance"],
        "family history": ["increased risk of genetic predisposition"],
    }

    # return the list of risk factors based on the patient's daily life habits for the specific disease
    if disease == "Stroke":
        return stroke_risk_factors
    elif disease == "Diabetes":
        return diabetes_risk_factors
    else:
        return []


def get_unaware_risk_factors(disease):
    # create a dictionary of daily life habits and their associated unaware risk factors for stroke
    stroke_unaware_risk_factors = {
        "sleep apnea": ["increased risk of high blood pressure and heart disease"],
        "atrial fibrillation": ["increased risk of blood clots and stroke"],
        "migraines": ["increased risk of stroke"],
        "oral health": ["increased risk of heart disease and stroke"]
    }

    # create a dictionary of daily life habits and their associated unaware risk factors for diabetes
    diabetes_unaware_risk_factors = {
        "lack of sleep": ["increased risk of insulin resistance and high blood sugar"],
        "air pollution": ["increased risk of insulin resistance and high blood sugar"],
        "night shift work": ["increased risk of insulin resistance and high blood sugar"],
        "poor oral health": ["increased risk of gum disease and high blood sugar"]
    }

    # return the list of unaware risk factors based on the patient's daily life habits for the specific disease
    if disease == "Stroke":
        return stroke_unaware_risk_factors
    elif disease == "Diabetes":
        return diabetes_unaware_risk_factors
    else:
        return []


def get_advice(disease):
    # create a dictionary of advice for stroke
    stroke_advice = {
        "quit smoking": "Smoking is a major risk factor for stroke. Quitting smoking can help reduce your risk.",
        "moderate alcohol consumption": "Drinking too much alcohol can increase your risk of high blood pressure and stroke. Moderate alcohol consumption is recommended.",
        "regular exercise": "Physical activity is important for maintaining a healthy weight and reducing your risk of high blood pressure and stroke.",
        "healthy diet": "Eating a healthy diet low in sodium, cholesterol, and saturated fat can help reduce your risk of high blood pressure and stroke.",
        "stress management": "Stress can increase your risk of high blood pressure and heart disease. Managing your stress levels can help reduce your risk."
    }

    # create a dictionary of advice for diabetes
    diabetes_advice = {
        "healthy diet": "Eating a healthy diet low in sugar and carbohydrates can help manage your blood sugar levels.",
        "regular exercise": "Physical activity can help your body use insulin more effectively and manage your blood sugar levels.",
        "maintain a healthy weight": "Being overweight or obese can increase your risk of insulin resistance and high blood sugar. Maintaining a healthy weight can help manage your blood sugar levels.",
        "check blood sugar levels regularly": "Checking your blood sugar levels regularly can help you monitor your condition and make adjustments to your diet and medication as needed.",
        "take medication as prescribed": "Taking your medication as prescribed by your doctor can help manage your blood sugar levels."
    }

    # return the list of advice for the patient to reduce their risk factors
    if disease == "Stroke":
        return stroke_advice.values()
    elif disease == "Diabetes":
        return diabetes_advice.values()
    else:
        return []


def Recommend(disease, user):
    # ------Recommendation-----------
    st.subheader("Recommendation")
    prediction_result = user.prediction_result

    # obtain risk factors based on daily life habits for the specific disease
    risk_factors = get_risk_factors(disease)
    # obtain unaware risk factors based on daily life habits for the specific disease
    unaware_risk_factors = get_unaware_risk_factors(disease)

    if prediction_result == "positive":
        # provide feedback to the patient based on their risk factors
        st.write("Based on your daily habits, you are at risk of getting", disease)
        st.write("The following risk factors may contribute to your condition:")
        for factor in risk_factors:
            st.write("- ", factor, ": ", (str(risk_factors[factor]).replace("['", '').replace("']", '')))
        st.write("There are also some unaware risk factors that you should be aware of:")
        for factor in unaware_risk_factors:
            st.write("- ", factor)
        st.write("We recommend that you take the following actions to reduce your risk:")
        for advice in get_advice(disease):
            st.write("- ", advice)
    else:
        # provide feedback to the patient to maintain their healthy lifestyle habits
        st.write("Based on your daily habits, you are not at risk of getting", disease)
        st.write("However, there are some unaware risk factors that you should be aware of:")
        for factor in unaware_risk_factors:
            st.write("- ", factor, ":", (str(unaware_risk_factors[factor]).replace("['", '').replace("']", '')))
        st.write(
            "We recommend that you continue to maintain your healthy lifestyle habits to prevent the risk of getting",
            disease, ".")


def VisualData(now):
    # ------VISUAL-----------
    st.subheader("Data Visualization")

    df1 = now.df[now.df[now.target] == 1]
    visual_list = [x for x in now.var_string_list if (x not in now.special_list)]

    for variable in visual_list:
        fig = go.Figure()
        fig.update_layout(
            height=400,
            width=680,
            paper_bgcolor="rgba(178,216,216,50)",

        )

        df2 = df1[now.df[{variable}] == 1]
        df3 = df1[now.df[{variable}] == 2]

        fig.add_trace(go.Bar(
            x=df1[now.target].unique(),
            y=df2[{variable}].value_counts(),
            name=f"{variable}=Yes",
            marker_color="rgba(0,76,76,50)"

        ))
        fig.add_trace(go.Bar(
            x=df1[now.target].unique(),
            y=df3[{variable}].value_counts(),
            name=f"{variable}=No",
            marker_color="rgba(102,178,178,50)"
        ))
        for v in now.health_questions:
            if v == variable:
                question = now.health_questions[v]
        for v in now.other_questions:
            if v == variable:
                question = now.other_questions[v]

        fig.update_layout(title_text=f"{question} ", barmode="group")
        fig.update_yaxes(title="Count")
        fig.update_xaxes(title=f"Having {now.name}")

        st.plotly_chart(fig, use_container_width=True)


# ----NAVIGATION MENU-------


selected = option_menu(
    menu_title=None,
    options=["Home", "Stroke Prediction", "Diabetes Prediction", "Information Query"],
    icons=["house", "hospital-fill", "hospital-fill", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    BackgroundInfo()

elif selected == "Stroke Prediction":
    StrokePredict()

elif selected == "Diabetes Prediction":
    DiabetesPredict()

elif selected == "Information Query":
    InfoQuery()
