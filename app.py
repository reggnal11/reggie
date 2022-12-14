import streamlit as st






#### Add header to describe app

st.markdown("# Predict Whether or Not You Are a LinkedIn User")



#### Create Income input

input_income = st.selectbox("Income (Annual)",

             options = ["Less than $10,000",

                        "$10,000 to $19,999",

                        "$20,000 to $29,999",

                        "$30,000 to $39,999",

                        "$40,000 to $49,999",

                        "$50,000 to $74,999",

                        "$75,000 to $99,999",

                        "$100,000 to $149,999",

                        "$150,000 or more"

                        ])




if input_income == "Less than $10,000":

    input_income = 1

elif input_income == "$10,000 to $19,999":

    input_income = 2

elif input_income == "$20,000 to $29,999":

    input_income = 3

elif input_income == "$30,000 to $39,999":

    input_income = 4

elif input_income == "$40,000 to $49,999":

    input_income = 5

elif input_income == "$50,000 to $74,999":

    input_income = 6

elif input_income == "$75,000 to $99,999":

    input_income = 7

elif input_income == "$100,000 to $149,999":

    input_income = 8

else:

    input_income = 9






#### Create Education input

input_education = st.selectbox("Education level",

             options = ["Less Than High School",

                        "Some High School",

                        "High School Graduate",

                        "Some College, No Degree",

                        "Two Year Associates Degree",

                        "Bachelors Degree",

                        "Some Post Graduate Schooling",

                        "Postgraduate or Professional Degree"                        

                        ])




if input_education == "Less Than High School":

    input_education = 1

elif input_education == "Some High School":

    input_education = 2

elif input_education == "High School Graduate":

    input_education = 3

elif input_education == "Some College, No Degree":

    input_education = 4

elif input_education == "Two Year Associates Degree":

    input_education = 5

elif input_education == "Bachelors Degree":

    input_education = 6

elif input_education == "Some Post Graduate Schooling":

    input_education = 7

else:

    input_education = 8







#### Create Parent input

input_parent = st.selectbox("Are You a Parent",

             options = ["I am a Parent",

                        "I am Not a Parent"])



if input_parent == "I am a Parent":

    input_parent = 1

else:

    input_parent = 0







 

#### Create Married input

input_married = st.selectbox("Are You Married",

             options = ["I am Married",

                        "I am Not Married"])



if input_married == "I am Married":

    input_married = 1

else:

    input_married = 0


#### Create Female input

input_female = st.selectbox("Do You Identify as Female",

             options = ["Yes",

                        "No"])

 

if input_female == "Yes":

    input_female = 1

else:

    input_female = 0




#### Create Age input

input_age = st.slider(label="Enter Your Age",

          min_value=1,

          max_value=98,

          value=50)










########################################################################################
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report




s = pd.read_csv('social_media_usage.csv')





def clean_sm(x):
    x = np.where(x == 1,1,0)
    return x 




ss = pd.DataFrame({
    "income": np.where(s["income"] <= 9,s["income"],np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"]==1,1,0),
    "married": np.where(s["marital"]==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age": np.where(s["age"] <= 98, s["age"], np.nan),
    "sm_li": clean_sm(s["web1h"])  
})





ss = ss.dropna()







y = ss["sm_li"] # Target Vector
X = ss[["income", "education", "parent", "married", "female", "age"]]




x_training, x_test, y_training, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y, 
                                                    test_size = 0.2,
                                                    random_state = 1030)




Logreg = LogisticRegression(class_weight = "balanced")
Logreg.fit(x_training, y_training) 




# prediction
y_pred = Logreg.predict(x_test)


Logreg.score(x_training, y_training)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")




newdata = pd.DataFrame({
    "income": [8,8],
    "education": [7,7],
    "parent": [0,0],
    "married": [1,1],
    "female":[1,1],
    "age": [42, 82]})





######
newdata["prediction_LinkedIn_User"] = Logreg.predict(newdata)
newdata

person1 = [8,7,0,1,1,42]
person2 = [8,7,0,1,1,82]



####
person = pd.DataFrame({
    "income": [input_income],
    "education": [input_education],
    "parent": [input_parent],
    "married": [input_married],
    "female":[input_female],
    "age": [input_age]})





predicted_class = Logreg.predict(person)

if predicted_class[0]==0:

    user_or_not = "NOT a LinkedIn User"

else:

    user_or_not = "a LinkedIn User"





probs = Logreg.predict_proba(person)

percent_prob = round((probs[0][1])*100, 2)






if st.button("Click for results"):

    st.write("The probability that you are a LinkedIn User is",round((probs[0][1])*100, 2),"%")

    st.write("I believe that you are", user_or_not)










#predicted_class = Logreg.predict([person1])


#probability_Pos = Logreg.predict_proba([person1])

#print(f"Predicted class: {predicted_class[0]}") 
#print(f"Probability that this person is a LinkedIn User: {probability_Pos[0][1]}")


































