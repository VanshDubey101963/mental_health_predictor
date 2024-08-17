import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv('dataset\Mental Health Dataset 2.csv')

def replace_values(column):
    unique_values = data[column].unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    data[column] = data[column].map(value_map)
    return value_map

gender_map = replace_values('Gender')
country_map = replace_values('Country')
occupation_map = replace_values('Occupation')
self_employed_map = replace_values('self_employed')
family_history_map = replace_values('family_history')
treatment_map = replace_values('treatment')
days_indoors_map = replace_values('Days_Indoors')
growing_stress_map = replace_values('Growing_Stress')
changes_habits_map = replace_values('Changes_Habits')
mental_health_history_map = replace_values('Mental_Health_History')
mood_swings_map = replace_values('Mood_Swings')
coping_struggles_map = replace_values('Coping_Struggles')
care_options_map = replace_values('care_options')
mental_health_interview_map = replace_values('mental_health_interview')
social_weakness_map = replace_values('Social_Weakness')
work_interest_map = replace_values('Work_Interest')


data.drop(['Timestamp'], axis=1, inplace=True)

X = data.values[:, 1:16]
Y = data.values[:, 15]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

first_layer_size = 32
num_classes = 3
model = Sequential()
model.add(Dense(first_layer_size, activation='sigmoid', input_shape=(15,)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

Y_train[Y_train >= num_classes] = num_classes - 1
Y_test[Y_test >= num_classes] = num_classes - 1

Y_train = keras.utils.to_categorical(Y_train, num_classes, dtype="float")
Y_test = keras.utils.to_categorical(Y_test, num_classes, dtype="float")

history = model.fit(X_train, Y_train, batch_size=10, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print("test loss", score[0])
print("test accuracy", score[1])


gender_map = {'Female': 0, 'Male': 1}
country_map = {'United States': 0, 'Poland': 1, 'Australia': 2, 'Canada': 3, 'United Kingdom': 4, 'South Africa': 5, 'Sweden': 6, 'New Zealand': 7, 'Netherlands': 8, 'India': 9, 'Belgium': 10, 'Ireland': 11, 'France': 12, 'Portugal': 13, 'Brazil': 14, 'Costa Rica': 15, 'Russia': 16, 'Germany': 17, 'Switzerland': 18, 'Finland': 19, 'Israel': 20, 'Italy': 21, 'Bosnia and Herzegovina': 22, 'Singapore': 23, 'Nigeria': 24, 'Croatia': 25, 'Thailand': 26, 'Denmark': 27, 'Mexico': 28, 'Greece': 29, 'Moldova': 30, 'Colombia': 31, 'Georgia': 32, 'Czech Republic': 33, 'Philippines': 34}
occupation_map = {'Corporate': 0, 'Student': 1, 'Business': 2, 'Housewife': 3, 'Others': 4}
self_employed_map = {'No': 0, 'Yes': 1}
family_history_map = {'No': 0, 'Yes': 1}
treatment_map = {'Yes': 0, 'No': 1}
days_indoors_map = {'1-14 days': 0, 'Go out Every day': 1, 'More than 2 months': 2, '15-30 days': 3, '31-60 days': 4, 'Everyday': 1, 'Often': 5}
growing_stress_map = {'Yes': 0, 'No': 1, 'Maybe': 2}
changes_habits_map = {'No': 0, 'Yes': 1, 'Maybe': 2}
mental_health_history_map = {'Yes': 0, 'No': 1, 'Maybe': 2}
mood_swings_map = {'Medium': 0, 'Low': 1, 'High': 2}
coping_struggles_map = {'No': 0, 'Yes': 1}
mental_health_interview_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
social_weakness_map = {'Yes': 0, 'No': 1, 'Maybe': 2}
work_interest_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
care_options_map = {0: 'Yes', 1: 'No', 2: 'Maybe'}



def process_user_input(user_input):
    def get_mapped_value(map_dict, key):
        if key not in map_dict:
            raise ValueError(f"Unexpected input: {key}. Please provide a valid input.")
        return map_dict[key]

    processed_input = [
        get_mapped_value(gender_map, user_input['Gender']),
        get_mapped_value(country_map, user_input['Country']),
        get_mapped_value(occupation_map, user_input['Occupation']),
        get_mapped_value(self_employed_map, user_input['self_employed']),
        get_mapped_value(family_history_map, user_input['family_history']),
        get_mapped_value(treatment_map, user_input['treatment']),
        get_mapped_value(days_indoors_map, user_input['Days_Indoors']),
        get_mapped_value(growing_stress_map, user_input['Growing_Stress']),
        get_mapped_value(changes_habits_map, user_input['Changes_Habits']),
        get_mapped_value(mental_health_history_map, user_input['Mental_Health_History']),
        get_mapped_value(mood_swings_map, user_input['Mood_Swings']),
        get_mapped_value(coping_struggles_map, user_input['Coping_Struggles']),
        get_mapped_value(mental_health_interview_map, user_input['mental_health_interview']),
        get_mapped_value(social_weakness_map, user_input['Social_Weakness']),
        get_mapped_value(work_interest_map, user_input['Work_Interest'])
    ]
    return processed_input


def predict_care_option(model, user_input):
    try:
        processed_input = process_user_input(user_input)
        processed_input = np.array(processed_input).reshape(1, -1)  # Reshape for the model
        prediction = model.predict(processed_input)
        care_option = np.argmax(prediction, axis=1)[0]
        return care_option
    except ValueError as e:
        print(e)
        return None


def prediction(model, user_input):
    care_option = predict_care_option(model, user_input)
    if care_option is not None:
        care_option_str = care_options_map.get(care_option, 'Unknown')
    else:
        care_option_str = 'Invalid input provided'
    return care_option_str


def predictToUser(userInput):
    print(userInput)
    care_option = prediction(model,userInput)
    return care_option