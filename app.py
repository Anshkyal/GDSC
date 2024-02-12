import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/LENOVO/training_data.csv")

le_oly = LabelEncoder()
le_scho = LabelEncoder()
le_sch = LabelEncoder()
le_fav = LabelEncoder()
le_pro = LabelEncoder()
le_med = LabelEncoder()
le_car = LabelEncoder()
le_act = LabelEncoder()
le_fan = LabelEncoder()
le_won = LabelEncoder()
le_ho = LabelEncoder()
label_encoder = LabelEncoder()

df['Olympiad_Participation_n'] = le_oly.fit_transform(df['Olympiad_Participation'])
df['Scholarship_n'] = le_scho.fit_transform(df['Scholarship'])
df['School_n'] = le_sch.fit_transform(df['School'])
df['Fav_sub_n'] = le_fav.fit_transform(df['Fav_sub'])
df['Projects_n'] = le_pro.fit_transform(df['Projects'])
df['Medals_n'] = le_med.fit_transform(df['Medals'])
df['Career_sprt_n'] = le_car.fit_transform(df['Career_sprt'])
df['Act_sprt_n'] = le_act.fit_transform(df['Act_sprt'])
df['Fant_arts_n'] = le_fan.fit_transform(df['Fant_arts'])
df['Won_arts_n'] = le_won.fit_transform(df['Won_arts'])
df['Predicted_Hobby_n'] = le_ho.fit_transform(df['Predicted Hobby'])

X_train = df.drop(columns=["Predicted Hobby", "Predicted_Hobby_n"])
y_train = df["Predicted_Hobby_n"]
X_train_encoded = df[['Olympiad_Participation_n','Scholarship_n','School_n','Fav_sub_n','Projects_n','Medals_n','Career_sprt_n','Act_sprt_n','Fant_arts_n','Won_arts_n']]

st.title("Hobby Prediction")

olympiad_participation = st.selectbox("Olympiad Participation", ["Yes", "No"])
scholarship = st.selectbox("Scholarship", ["Yes", "No"])
school = st.selectbox("School", ["Yes", "No"]) 
fav_sub = st.selectbox("Favorite Subject", ["Mathematics", "Any language", "Science", "History/Geography"])
projects = st.selectbox("Projects", ["Yes", "No"])
grasp_pow = st.slider("Grasping Power", min_value=1, max_value=5, value=3)
time_sprt = st.slider("Time for Sports", min_value=1, max_value=10, value=5)
medals = st.selectbox("Medals", ["No", "Yes"])
career_sprt = st.selectbox("Career in Sports", ["No", "Yes"])
act_sprt = st.selectbox("Active in Sports", ["No", "Yes"])
fant_arts = st.selectbox("Fantasy in Arts", ["No", "Yes"])
won_arts = st.selectbox("Won in Arts", ["No", "Yes"])
time_art = st.slider("Time for Arts", min_value=1, max_value=5, value=3)

olympiad_participation_n = le_oly.transform([olympiad_participation])[0]
scholarship_n = le_scho.transform([scholarship])[0]
school_n = le_sch.transform([school])[0]
fav_sub_n = le_fav.transform([fav_sub])[0]
projects_n = le_pro.transform([projects])[0]
medals_n = le_med.transform([medals])[0]
career_sprt_n = le_car.transform([career_sprt])[0]
act_sprt_n = le_act.transform([act_sprt])[0]
fant_arts_n = le_fan.transform([fant_arts])[0]
won_arts_n = le_won.transform([won_arts])[0]

model = SVC(gamma='auto', kernel='rbf', random_state=0)
model.fit(X_train_encoded, y_train)

prediction = model.predict([[olympiad_participation_n, scholarship_n, school_n, fav_sub_n, projects_n, medals_n, career_sprt_n, act_sprt_n, fant_arts_n, won_arts_n]])

predicted_hobby = le_ho.inverse_transform(prediction)[0]

st.write(f"Predicted Hobby: {predicted_hobby}")