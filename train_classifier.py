import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Predict labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)

# Print the accuracy
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
f.close()
