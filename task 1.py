#!/usr/bin/env python
# coding: utf-8

# # Logistic regression

# In[10]:

#import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import os

# Load and preprocess the dataset
data_path = 'captions.csv'
image_folder = 'Images'
data = pd.read_csv(data_path, names=['description'], header=None)
data = data.iloc[:1000]  # Reduce dataset to 1000 items
image_files = sorted(os.listdir(image_folder))[:1000]

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((64, 64), Image.ANTIALIAS)
    return np.array(img).flatten()

image_features = np.array([process_image(os.path.join(image_folder, img)) for img in image_files])
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(data['description']).toarray()
features = np.hstack((image_features, text_features))
labels = np.random.randint(0, 2, size=len(data))

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

# Print results
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# # Support vector machine

# In[12]:


#import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import os

# Load and preprocess the dataset
data_path = 'captions.csv'
image_folder = 'Images'
data = pd.read_csv(data_path, names=['description'], header=None)
data = data.iloc[:1000]  # Reduce dataset to 1000 items
image_files = sorted(os.listdir(image_folder))[:1000]

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((64, 64), Image.ANTIALIAS)
    return np.array(img).flatten()

image_features = np.array([process_image(os.path.join(image_folder, img)) for img in image_files])
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(data['description']).toarray()
features = np.hstack((image_features, text_features))
labels = np.random.randint(0, 2, size=len(data))

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

# Print results
print("SVM Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# # Decision tree

# In[15]:


#import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import os

# Load and preprocess the dataset
data_path = 'captions.csv'
image_folder = 'Images'
data = pd.read_csv(data_path, names=['description'], header=None)
data = data.iloc[:1000]  # Reduce dataset to 1000 items
image_files = sorted(os.listdir(image_folder))[:1000]

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((64, 64), Image.ANTIALIAS)
    return np.array(img).flatten()

image_features = np.array([process_image(os.path.join(image_folder, img)) for img in image_files])
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(data['description']).toarray()
features = np.hstack((image_features, text_features))
labels = np.random.randint(0, 2, size=len(data))

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

# Print results
print("Decision Tree Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# In[ ]:




