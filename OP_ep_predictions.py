'''
Predicting what episode a one piece character will appear for the first time
(based off of the chapter they first appeared)
'''

### imports & preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

chars = pd.read_csv('Characters.csv', usecols=[0,1,2,3])
    # not importing notes column

# make all numerical columns integers (instead of floats)
chars.episode = pd.to_numeric(chars.episode, downcast='integer', errors='coerce')
chars.chapter = pd.to_numeric(chars.chapter, downcast='integer', errors='coerce')
chars.year = pd.to_numeric(chars.year, downcast='integer', errors='coerce')

# make categorical variable (name) numeric
label_encoder = LabelEncoder()
chars['encoded_name'] = label_encoder.fit_transform(chars['name'])

# remove rows with relevant N/A values
chars = chars.dropna(subset=['chapter', 'episode'])

# remove rows with year <1997 (when the manga was first released)
chars = chars.drop(chars[chars.year < 1997].index)

### look at the data & make sure all is good
chars.info()
print(chars.head())



### Model Creation

# feature selection
X = chars[['chapter', 'year']] # use data from these columns
Y = chars['episode'] # to predict whats in this column

# split into train/test sets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=42)

# train model & make predictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# limit predictions to highest episode number
y_pred =  np.clip(y_pred, None, chars['episode'].max())



### Model Evalution

# finding model error
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# accuracy within 50 episodes
accuracy50 = np.abs(y_test - y_pred) <= 50
buffer_room = np.mean(accuracy50) * 100

# print results
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print(f"Accuracy within 50 episodes: {buffer_room:.2f}%")

# plot results
plt.scatter(y_test, y_pred, color="red", alpha=0.6)
plt.xlabel("Actual Episode")
plt.ylabel("Predicted Episode")
plt.title("Predicted vs Actual Episode of one Piece Character Appearances")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linestyle=":")
plt.show()