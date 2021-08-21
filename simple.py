import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

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
model=svm.SVC(probability=True)

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
