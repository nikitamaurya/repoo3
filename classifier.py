import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,accuracy_score

def TrainingModel():
    dataset = pd.read_csv('HR-Employee-Attrition.csv')
    dataset['Attrition_ind'] = 0 
    print(dataset.columns)
    dataset.loc[dataset['Attrition'] =='Yes', 'Attrition_ind'] = 1

    dataset = pd.get_dummies(dataset)

    data_main=dataset.drop(['EmployeeCount','EmployeeNumber','Over18_Y','StandardHours','Attrition_No', 'Attrition_Yes'],axis=1)

    data_main['Attrition']=data_main['Attrition_ind']

    data_main=data_main.drop(['Attrition_ind'],axis=1)

    X=data_main.drop('Attrition',axis=1)
    y=data_main.Attrition

    features_label = data_main.columns[:-1]

    # Splitting into Train and Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train2 = pd.DataFrame(sc.fit_transform(X_train))
    X_test2 = pd.DataFrame(sc.transform(X_test))
    X_train2.columns = X_train.columns.values
    X_test2.columns = X_test.columns.values
    X_train2.index = X_train.index.values
    X_test2.index = X_test.index.values
    X_train = X_train2
    X_test = X_test2

    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()
    classifier.add(Dense(units = 26, kernel_initializer = 'uniform', activation = 'relu', input_dim = 51))
    classifier.add(Dense(units = 26, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 40)
    y_pred = classifier.predict(X_test)

    y_pred = (y_pred > 0.5)
    xl=(list(X_test.index))
    tn=[]
    for i in range(0,len(xl)):
        if y_pred[i][0]==True:
            tn.append(str(xl[i]))
            print(str(xl[i])+ " - " +str(y_pred[i][0]))

    from sklearn.metrics import confusion_matrix,accuracy_score
    cm = confusion_matrix(y_test, y_pred) 
    df_cm = pd.DataFrame(cm, index = (1, 0), columns = (1, 0))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig("frontend/static/cm.png")
    print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    return tn

def testing(fName):
    from joblib import load
    reconstructed_model = keras.models.load_model("model")
    sc=load('std_scaler.bin')

    dataset = pd.read_csv(fName)
    #dataset['temp']=0
    # dataset['Attrition_ind'] = 0 
    # dataset.loc[dataset['Attrition'] =='Yes', 'Attrition_ind'] = 1  
    dataset = pd.get_dummies(dataset)
    data_main=dataset.drop(['EmployeeCount',
        'EmployeeNumber','Over18_Y','StandardHours','Attrition_No', 'Attrition_Yes'],axis=1)
    # data_main['Attrition']=data_main['Attrition_ind']
    # data_main=data_main.drop(['Attrition_ind'],axis=1)
    # X=data_main.drop('Attrition',axis=1)
    # y=data_main.Attrition
    X = data_main
    # print(y[0])
    X_test2 = pd.DataFrame(sc.transform(X))
    X_test2.columns = X.columns.values
    X_test2.index = X.index.values
    X_test = X_test2
    # print(X_test.head())
    y_pred = reconstructed_model.predict(X_test)

    y_pred = (y_pred > 0.5)
    #print(y_pred)
    xl=(list(X_test.index))
    tn=[]
    for i in range(0,len(xl)):
        if y_pred[i][0]==True:
            tn.append(xl[i])
            print(str(xl[i])+ " - " +str(y_pred[i][0]))
    return tn
