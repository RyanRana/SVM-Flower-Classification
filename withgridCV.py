import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle

Categories=['daisy', 'dandelion','rose','sunflower','tulip']
flat_data_arr=[] 
target_arr=[] 
datadir='/flowers'

for i in Categories:
    path=os.path.join(datadir,i)
    
    for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(100,100,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
            
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data)
    df['Target']=target
    x=df.iloc[:,:-1] 
    y=df.iloc[:,-1]
    
from sklearn import svm
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=77,stratify=y)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(accuracy_score(y_pred,y_test))



url=input('Enter URL of Image :')
img=imread(url)
plt.imshow(img)
plt.show()
img_resize=resize(img,(100,100,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
print("The predicted image is : "+Categories[model.predict(l)[0]])
