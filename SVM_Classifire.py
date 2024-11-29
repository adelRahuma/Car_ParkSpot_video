import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


input_dir = '.\data'
categories = ['empty','not_empty']

data = []
lables = []

for  cat_id, category in enumerate(categories):
    # print(str(cat_id) +'   '+category)
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category, file)
        img = imread(img_path)
        img = resize(img,(15,15))
        #print(img)   # ---> it is list of list
        data.append(img.flatten())  # to put the image in one
        lables.append(cat_id)
    print(f'loaded category:{category} successfully')

data = np.asarray(data)
lables = np.asarray(lables)

x_train, x_test, y_train, y_test = train_test_split(data, lables, test_size=0.2, shuffle=True, stratify=lables)


clf = SVC()

parameters = [{'gamma':[0.01,0.001,0.0001], 'C':[1, 10, 100, 1000]}]
#if you want to use model tuning GridSearchCV is your friend
grid_search = GridSearchCV(clf,parameters)
grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_
# best_estimator -> here in our Example it has the SVC(C=10, gamma=0.01)
# We fed the best_estimator with x_test which the images
y_prediction = best_estimator.predict(x_test)
print(best_estimator)
# 4 - Test Performance
# comarison between the predicted class "y_prediction"  Vs true class value "y_test"
score = accuracy_score(y_prediction,y_test)
print(" EvaLUATION... {}% samples were corrected classified".format(str(score * 100)))
pickle.dump(best_estimator, open('./SVM_Model.p', 'wb'))









