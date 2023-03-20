import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import sklearn
from streamlit_option_menu import option_menu
import joblib

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
    model = joblib.load("rnd_clf.pkl")
    var_val_list = [["PFQ051", "MCQ160E", "CSQ030", "SLD010H", "INQ030", "MCQ365A", "PFQ090", "OHQ860", "CDQ010",
                     "MCQ370B", "PFQ061R", "BPQ020", "DUQ200", "OCQ180", "DLQ040", "OHQ614", "DBD900", "FSDAD",
                     "MCQ203", "HSD010"]]
    var_string_list = ["PFQ051", "MCQ160E", "CSQ030", "SLD010H", "INQ030", "MCQ365A", "PFQ090", "OHQ860", "CDQ010",
                       "MCQ370B", "PFQ061R", "BPQ020", "DUQ200", "OCQ180", "DLQ040", "OHQ614", "DBD900", "FSDAD",
                       "MCQ203", "HSD010"]

    special_list = ["HSD010", "SLD010H", "PFQ061R", "OCQ180", "DBD900", "FSDAD", "MCQ160F"]

    health = ["MCQ160E", "CSQ030", "MCQ365A", "PFQ090", "OHQ860", "CDQ010", "BPQ020", "OHQ614", "MCQ203", "HSD010"]
    daily = ["PFQ051", "SLD010H", "INQ030", "MCQ370B", "PFQ061R", "DUQ200", "OCQ180",
             "DLQ040", "DBD900", "FSDAD"]
    minmax = joblib.load("minmax_stroke.gz")


class Diabetes:
    # --variable----
    name = "Diabetes"
    df = pd.read_csv("final_testset1_diabetes.csv")
    model = joblib.load("lr_clf_dia.pkl")
    target = "DIQ010"
    var_val_list = [["BPQ020", "DIQ180", "MCQ365C", "MCQ300C", "MCQ080", "INQ020", "CSQ110", "FSD032A",
                     "DUQ200", "MGATHAND", "MCQ365B", "HEQ030", "DBD100", "HSQ520", "INQ030", "OCQ260", "OHQ835",
                     "FSD032C"]]
    var_string_list = ["BPQ020", "DIQ180", "MCQ365C", "MCQ300C", "MCQ080", "INQ020", "CSQ110", "FSD032A",
                       "DUQ200", "MGATHAND", "MCQ365B", "HEQ030", "DBD100", "HSQ520", "INQ030", "OCQ260", "OHQ835",
                       "FSD032C"]
    special_list = ["FSD032A", "DBD100", "OCQ260", "FSD032C"]

    health = ["BPQ020", "DIQ180", "MCQ300C", "MCQ080", "MCQ365B", "HEQ030", "HSQ520", "OHQ835"]
    daily = ["MCQ365C", "INQ020", "CSQ110", "FSD032A", "DUQ200", "MGATHAND", "DBD100", "INQ030", "OCQ260", "FSD032C"]
    minmax = joblib.load("minmax_dia.gz")


class Patient:
    list = []  # init the input list

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
                st.write("Congratulations! You are possibly not having a stroke.")
            else:
                st.write("You may having a risk of getting stroke! ")

        else:
            transformData = Diabetes.minmax.transform(df)
            prediction = Diabetes.model.predict(transformData)
            if prediction[0] == 0:
                st.write("Congratulations! You are possibly not having a diabetes.")

            else:
                st.write("You may having a risk of getting diabetes! ")


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
        st.image(image, caption='tbd')


def StrokePredict():
    # Display the stroke prediction model header
    st.header(f"Stroke prediction model")
    # Create a new patient and stroke instance
    user = Patient(1)
    now = Stroke()

    # Display the consent agreement form
    ConsentAgree()
    st.write("By clicking 'I agree.', you agree to the terms and conditions.")
    check = st.checkbox("I agree.")
    if check:

        with st.form("entry_form"):
            st.subheader("Please fill in the following information:")

            # Display the health-related input fields in an expander
            with st.expander("Health"):
                # Collect health-related input fields
                MCQ160E = 1 if st.radio("Has a doctor or other health professional ever told you that you had a heart attack?", ('Yes', 'No'), key="MCQ160E") == 'Yes' else 2
                MCQ365A = 1 if st.radio("During the past 12 months have you ever been told by a doctor to control your weight?", ('Yes', 'No'), key="MCQ365A") == 'Yes' else 2
                PFQ090 = 1 if st.radio("Do you now have any health problem that requires you to use special equipment?", ('Yes', 'No'), key="PFQ090") == 'Yes' else 2
                OHQ860 = 1 if st.radio("Have you ever been told by a dentist that you lost bone around your teeth?", ('Yes', 'No'), key="OHQ860") == 'Yes' else 2
                CDQ010 = 1 if st.radio("Have you had shortness of breath either when hurrying on the level or walking up a slight hill?", ('Yes', 'No'), key="CDQ010") == 'Yes' else 2
                BPQ020 = 1 if st.radio("Have you ever been told by a doctor that you had high blood pressure?: ", ('Yes', 'No'), key="BPQ020") == 'Yes' else 2
                OHQ614 = 1 if st.radio("Did a dentist have a conversation with you about the importance of examining your mouth for oral cancer?: ", ('Yes', 'No'), key="OHQ614") == 'Yes' else 2
                MCQ203 = 1 if st.radio("Has anyone ever told you that you had jaundice? Please do not include infant jaundice, which is common during the first weeks after birth.", ('Yes', 'No'), key="MCQ203") == 'Yes' else 2
                HSD010 = st.slider('Would you say your health in general is : (1: Excellent, 2:Very good, 3:Good, 4:Fair, 5:Poor)', 1, 5, 1, key="HSD010")

            # Display the daily-related input fields in an expander
            with st.expander("Daily"):
                # Collect daily-related input fields
                PFQ051 = 1 if st.radio("Are you limited in the kind or amount of work you can do because of a physical, mental or emotional problem? (1 for Yes, 2 for No):", ('Yes', 'No'), key="PFQ051") == 'Yes' else 2
                CSQ030 = 1 if st.radio("Do some smells bother you although they do not bother other people? (1 for Yes, 2 for No):", ('Yes', 'No'), key="CSQ030") == 'Yes' else 2
                SLD010H = st.slider("How much sleep do you usually get at night on weekdays or workdays? (Answer in hours):", 2, 12, 2, key="SLD010H")
                INQ030 = 1 if st.radio("Did you or any family members living here receive income in 2022 from Social Security? (1 for Yes, 2 for No):", ('Yes', 'No'), key="INQ030") == 'Yes' else 2
                MCQ370B = 1 if st.radio("Are you now doing any of the following: increasing your physical activity or exercise?", ('Yes', 'No'), key="MCQ370B") == 'Yes' else 2
                PFQ061R = st.slider("How much difficulty do you have participating in social activities [visiting friends, attending clubs or meetings or going to parties? (#1 No difficulty, #2 Some difficulty, #3 Much difficulty, #4 Unable to do, #5 Do not do this activity): ", 1, 5, 1, key="PFQ061R")
                DUQ200 = 1 if st.radio("Have you ever used marijuana or hashish?", ('Yes', 'No'), key="DUQ200") == 'Yes' else 2
                OCQ180 = st.number_input("How many hours did you work last week at all jobs or businesses?: ", min_value=1, max_value=120, key="OCQ180")
                DLQ040 = 1 if st.radio("Do you have serious difficulty concentrating, remembering, or making decisions?: ", ('Yes', 'No'), key="DLQ040") == 'Yes' else 2
                DBD900 = st.slider("How many of those meals did you get from a fast-food or pizza place a week?", 0, 21, 1, key="DBD900")
                FSDAD = st.slider("What is your food security level? (1 for full, 2 for marginal, 3 for low, 4 for very low)", 1, 4, 1, key="FSDAD")

            "---"
            # save user inputs
            submitted = st.form_submit_button("Predict Stroke Diseases")
            if submitted:
                health = {health: st.session_state[health] for health in Stroke.health}
                daily = {daily: st.session_state[daily] for daily in Stroke.daily}
                st.write(f"healths: {health}")
                st.write(f"dailys: {daily}")
                st.success("Data saved!")

                user.list = [[PFQ051, MCQ160E, CSQ030, SLD010H, INQ030, MCQ365A, PFQ090, OHQ860, CDQ010,
                              MCQ370B, PFQ061R, BPQ020, DUQ200, OCQ180, DLQ040, OHQ614, DBD900, FSDAD, MCQ203,
                              HSD010]]
                # combine user inputs into a dataframe
                df = user.df_combine()
                # predict diabetes diseases
                user.prediction(df)
                VisualData(now)
                Recommend()


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

            # Display the health-related input fields in an expander
            with st.expander("Health"):
                BPQ020 = 1 if st.radio("Have you ever been told by a doctor that you had high blood pressure? : ", ('Yes', 'No'), key="BPQ020") == 'Yes' else 2
                DIQ180 = 1 if st.radio("Have you had a blood test for high blood sugar within the past three years? : ", ('Yes', 'No'), key="DIQ180") == 'Yes' else 2
                MCQ300C = 1 if st.radio("Close relative had diabetes? :", ('Yes', 'No'), key="MCQ300C") == 'Yes' else 2
                MCQ080 = 1 if st.radio("Do Doctor ever said you were overweight? :", ('Yes', 'No'), key="MCQ080") == 'Yes' else 2
                MCQ365B = 1 if st.radio("Do Doctor told you to exercise? :", ('Yes', 'No'), key="MCQ365B") == 'Yes' else 2
                HEQ030 = 1 if st.radio("Do you ever have Hepatitis C? :", ('Yes', 'No'), key="HEQ030") == 'Yes' else 2
                HSQ520 = 1 if st.radio("Do you ever have flu, pneumonia, or ear infection? :", ('Yes', 'No'), key="HSQ520") == 'Yes' else 2
                OHQ835 = 1 if st.radio("Do you think you might have gum disease? :", ('Yes', 'No'), key="OHQ835") == 'Yes' else 2

            # Display the daily-related input fields in an expander
            with st.expander("Daily"):
                MCQ365C = 1 if st.radio("Have you ever been told by someone to reduce the amount of sodium or salt in your diet? :", ('Yes', 'No'), key="MCQ365C") == 'Yes' else 2
                INQ020 = 1 if st.radio("Did you receive income in 2022 from wages and salaries? :", ('Yes', 'No'), key="INQ020") == 'Yes' else 2
                CSQ110 = 1 if st.radio("During the past 12 months have you had a taste or other sensation in your mouth that does not go away? :", ('Yes', 'No'), key="CSQ110") == 'Yes' else 2
                FSD032A = st.slider("Have you worried whether your food would run out before you got money to buy more? (1 for Often true, 2 for Sometimes true, 3 for Never true):", 1, 3, 1, key="FSD032A")
                DUQ200 = 1 if st.radio("Have you ever used marijuana or hashish? : ", ('Yes', 'No'), key="DUQ200") == 'Yes' else 2
                MGATHAND = 1 if st.radio("Which hand do you often use? (1 for right, 2 for left): ", ('Yes', 'No'), key="MGATHAND") == 'Yes' else 2
                DBD100 = st.slider("How often do you add ordinary salt to your food at the table? (1 for Rarely, 2 for Occasionally, 3 for Very often): ",1, 3, 1, key="DBD100")
                INQ030 = 1 if st.radio("Did you or any family members living here receive income in 2022 from Social Security? :", ('Yes', 'No'), key="INQ030") == 'Yes' else 2
                OCQ260 = st.slider("Which of these best describes your job or work situation? (1 for employee of a private company, business, or individual for wages, salary, or commission, 2 for federal government employee, 3 for government employee, 4 for local government employee, 5 for Self-employed in own business, professional practice or farm, 6 for Working without pay in family business or farm): ", 1, 6, 1, key="OCQ260")
                FSD032C = st.slider("Have you ever could not afford to eat balanced meals? (1 for Often true, 2 for Sometimes true, 3 for Never true):", 1, 3, 1, key="FSD032C")

            "---"

            # save user inputs
            submitted = st.form_submit_button("Predict Diabetes Diseases")
            if submitted:
                health = {health: st.session_state[health] for health in Diabetes.health}
                daily = {daily: st.session_state[daily] for daily in Diabetes.daily}
                st.write(f"healths: {health}")
                st.write(f"dailys: {daily}")
                st.success("Data saved!")

                user.list = [[BPQ020, DIQ180, MCQ365C, MCQ300C, MCQ080, INQ020, CSQ110, FSD032A,
                              DUQ200, MGATHAND, MCQ365B, HEQ030, DBD100, HSQ520, INQ030, OCQ260, OHQ835,
                              FSD032C]]
                # combine user inputs into a dataframe
                df = user.df_combine()
                # predict diabetes diseases
                user.prediction(df)
                VisualData(now)
                Recommend()


def InfoQuery():
    # ---------INFO PAGE --------
    st.header(f"More information about health check")
    st.markdown(
        """
        If you're interested in knowing more about your body status and want to perform a body check in Hong Kong, there are many hospitals and clinics that offer comprehensive health screening services. These services can provide you with a detailed assessment of your overall health and help identify any potential health issues early on.

        Some of the hospitals in Hong Kong that offer health screening services include:
        
        -	Hong Kong Adventist Hospital: Offers a range of health check packages that include general health assessments, cancer screenings, and cardiovascular screenings.
        -	St. Paul's Hospital: Provides comprehensive health check packages that include blood tests, imaging tests, and consultations with medical professionals.
        -	Hong Kong Sanatorium & Hospital: Offers a variety of health check packages that include general health assessments, cancer screenings, and cardiovascular screenings.
        -	Matilda International Hospital: Provides a range of health check packages that include blood tests, imaging tests, and consultations with medical professionals.
      
        The types of body checks that are included in these packages vary, but may include:
        -	Blood tests to check for cholesterol, glucose, liver and kidney function, and other markers of overall health.
        -	Imaging tests such as X-rays, ultrasounds, and CT scans to check for any abnormalities in organs or tissues.
        -	Cardiovascular tests such as electrocardiograms (ECGs) and stress tests to assess heart health and detect any potential issues.
        -	Cancer screenings such as mammograms, Pap smears, and colonoscopies to detect cancer early on.
        
        It is important to note that the cost of health check packages can vary depending on the hospital and the level of service provided. It is also recommended to consult with a medical professional to determine which health check package is most appropriate for your individual needs.
        If you're interested in learning more about health screening services in Hong Kong, it is recommended to contact your preferred hospital or clinic directly to inquire about the specific services they offer.

        """
    )

    #


def ConsentAgree():
    st.header("Privacy and Consent Statement for Online Prediction")
    st.markdown('<div style="text-align: justify;">We are currently conducting a prediction study focused on stroke and diabetes, incorporating inquiries about daily habits. Our primary objective is to enhance public awareness of risk factors for chronic diseases and promote preventive measures. We assure you that all information collected in this study is held in strict confidentiality and will not be disclosed to any third parties. We do not request any personally identifiable data and the input data will not be retained. Your responses will solely serve research purposes and will be reported in an aggregated format.</div>', unsafe_allow_html=True)


def Recommend():
    # ------Recommendation-----------
    st.header("Recommendation")



# TODO: recommend some advice according to the ans of user


def VisualData(now):
    # ------VISUAL-----------
    st.header("Data Visualization")

    df1 = now.df[now.df[now.target] == 1]
    visual_list = [x for x in now.var_string_list if (x not in now.special_list)]

    for variable in visual_list:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(207,226,243,50)",
        )

        df2 = df1[now.df[{variable}] == 1]
        df3 = df1[now.df[{variable}] == 2]

        fig.add_trace(go.Bar(
            x=df1[now.target].unique(),
            y=df2[{variable}].value_counts(),
            name=f"{variable}=1",
            marker_color="red"

        ))
        fig.add_trace(go.Bar(
            x=df1[now.target].unique(),
            y=df3[{variable}].value_counts(),
            name=f"{variable}=2",
            marker_color="lightgreen"
        ))

        fig.update_layout(title_text=f"{variable} vs {now.name}", barmode="group")
        fig.update_yaxes(title="Count")
        fig.update_xaxes(title=f"Having {now.name}")

        st.write(fig)


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
