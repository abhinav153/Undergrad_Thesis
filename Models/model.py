from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler
import torch
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class ANN:
   
    def __init__(self,X,Y,name,epochs=100,batch_size=64,lr=1e-2) -> None:

        self.name = name

        #Standardization Scaling
        scalers = {}
        scaler = StandardScaler()
        X=scaler.fit_transform(X)

        #print(X.shape)

        #one hot encode the labels
        self.encoder = OneHotEncoder()
        self.y_unencoded = Y
        self.X_train,self.X_test,self.y_train_unencoded,self.y_test_unencoded = train_test_split(X,self.y_unencoded,stratify=self.y_unencoded,shuffle=True)
        self.y_train = self.encoder.fit_transform(self.y_train_unencoded).toarray()
        self.y_test  = self.encoder.fit_transform(self.y_test_unencoded).toarray()
        self.categories = self.encoder.categories_[0]
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.X_train.shape[1],60)),
            ('relu1', nn.ReLU()),
            ('dropout1',nn.Dropout(0.2)),
            ('fc2', nn.Linear(60, 60)),
            ('relu2', nn.LeakyReLU()),
            ('dropout2',nn.Dropout(0.2)),
            ('fc3', nn.Linear(60, 60)),
            ('relu3', nn.ReLU()),
            ('dropout3',nn.Dropout(0.2)),
            ('fc4', nn.Linear(60, 60)),
            ('relu4', nn.ReLU()),
            ('dropout4',nn.Dropout(0.2)),
            ('fc5', nn.Linear(60, 30)),
            ('relu5', nn.ReLU()),
            ('dropout5',nn.Dropout(0.2)),
            ('fc6', nn.Linear(30, 30)),
            ('relu6', nn.ReLU()),
            ('fc7', nn.Linear(30, len(self.categories))),
            ('softmax', nn.Softmax(dim=1))
        ]))

        #converting arrays to tensors
        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_test  = torch.from_numpy(self.X_test).float()
        self.y_train = torch.from_numpy(self.y_train).float()
        self.y_test  = torch.from_numpy(self.y_test).float()

        #reshape arrays to 3dimensions
        #self.X_train = torch.reshape(self.X_train,(self.X_train.shape[0],4,-1))
        #self.X_train = torch.reshape(self.X_test,(self.X_test.shape[0],4,-1))


        #Model Paramters
        self.learning_rate = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #move data to gpu if cuda available
        if torch.cuda.is_available:
            self.X_train = self.X_train.cuda()
            self.y_train = self.y_train.cuda()
            self.X_test = self.X_test.cuda()
            self.y_test = self.y_test.cuda()
            self.model = self.model.cuda()
            

    def train(self,):
        self.model.train()

        val_loss_tracker = {}
        
  
        no_of_batches = self.X_train.shape[0]//self.batch_size
        for epoch in range(self.epochs):
            print('Current epoch:',epoch+1)
            #creating batches of data
            for mini_batch_no in range(no_of_batches+1):
                X_batch = self.X_train[mini_batch_no*self.batch_size : (mini_batch_no+1)*self.batch_size,:]
                y_batch = self.y_train[mini_batch_no*self.batch_size : (mini_batch_no+1)*self.batch_size]

                #zero the gradients
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output,y_batch)

                loss.backward()#calculate gradients
                self.optimizer.step()#update weights
            
            epoch_output = self.model(self.X_train)
            epoch_loss   = self.loss_fn(epoch_output,self.y_train)
            
            #track loss per epoch
            val_loss_tracker[f'Epoch_{epoch+1}'] = epoch_loss.item()

        self.model_trained = True
        
        return val_loss_tracker
                    
    def test(self):
        '''
        Following method will be used to evaluate metrics using test data
        '''
        if self.model:

            self.model.eval()#set model in eval mode, for ignoring dropout
            output = self.model(self.X_test)
            inverse_encoded = torch.argmax(output,dim=1)
            inverse_actual  = torch.argmax(self.y_test,dim=1)
            test_accuracy = accuracy_score(inverse_actual.cpu(),inverse_encoded.cpu())
            

            return test_accuracy.item()
        
class SVM:

    def __init__(self,X,Y,name) -> None:

        self.name = name

        #MinMax Scaling
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,Y,stratify=Y,shuffle=True)
        
        self.categories = np.unique(Y)

        #model parameters
        self.model = SVC()


    def train(self):
        
        self.model.fit(self.X_train,self.y_train.ravel())

    def test(self):

        output = self.model.predict(self.X_test)
        z = zip(output,self.y_test)
        
        test_accuracy = accuracy_score(self.y_test,output)
        print(f'{self.name}  Test Accuracy:{test_accuracy.item()}')
        
        return test_accuracy.item()
    

class RF:
    def __init__(self,X,Y,name) -> None:

        self.name = name

        #MinMax Scaling
        #scaler = MinMaxScaler()
        #X = scaler.fit_transform(X)

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,Y,stratify=Y,shuffle=True)
        self.categories = np.unique(Y)

        #model parameters
        self.model = RandomForestClassifier(max_depth=50,n_estimators=466,max_features='sqrt')

    
    def train(self):
        '''
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
        # Number of features to consider at every split
        max_features = [None, 'sqrt','log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 50, num = 2)]
        grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                  }
        rf_random = RandomizedSearchCV(estimator = self.model, param_distributions = grid, random_state=42,cv = 2, verbose=2, n_jobs = -1,scoring='accuracy')
        rf_random.fit(self.X_train,self.y_train.ravel())
       
        self.model = rf_random.best_estimator_
        '''
        self.model.fit(self.X_train,self.y_train.ravel())

    def test(self):
        output = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test,output)
        print(f'{self.name}  Test Accuracy:{test_accuracy.item()}')
        
        return test_accuracy.item()


        










        





            


