import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Find the maximum length of hand landmarks in the dataset
max_length = max(len(item) for item in data_dict['data'])

# Flatten each inner list to create a 2D array
data_flattened = [np.array(item).flatten() for item in data_dict['data']]

# Pad each inner list to have a consistent length
data_padded = [np.concatenate([item, [0.0] * ((max_length * 2) - len(item))]) for item in data_flattened]

data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
