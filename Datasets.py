#LOADING DATASETS FOR PAPER JAN KALINA

import numpy as np
import pandas as p


class datasets:

        def load_auto(self):

            data_mpg = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/auto-mpg.csv',sep = ',')
            data_mpg = data_mpg.drop(index = [32,126,330,336,354,374], axis = 1)
            data_mpg = data_mpg.drop(columns = ['cylinders','model year','origin','car name','Unnamed: 0'], axis = 0)
            data_mpg_x = data_mpg.iloc[:,1:5]
            data_mpg_y = data_mpg.iloc[:,0]

            return data_mpg_x,data_mpg_y


        def load_california_housing(self):

            data_train = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/California_housing/california_housing_train.csv',sep = ',')
            data_test = p.read_csv("https://raw.githubusercontent.com/jankalinaUI/Datasets/master/California_housing/california_housing_test.csv", sep = ",")
            data_dev = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/California_housing/california_housing_dev.csv', sep = ',')
            train_x = data_train.iloc[:,1:9]
            test_x = data_test.iloc[:,[1,2,3,4]]
            dev_x = data_dev.iloc[:,1:9]
            train_y = data_train.iloc[:,9]
            test_y = data_test.iloc[:,8]
            dev_y = data_dev.iloc[:,9]

            return train_x.values,test_x.values,dev_x.values,train_y.values,test_y.values,dev_y.values

        def load_boston_housing(self):

            data = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/BostonHousing.csv')
            train_xx = data.iloc[:,:11].values
            train_yy = data.iloc[:,11].values

            train_xx = train_xx.reshape((len(train_xx),train_xx.shape[1]))
            train_yy = train_yy.reshape((len(train_yy),1))

            #train_x = train_x.reshape((len(train_x),train_x.shape[1]))
            #train_y = train_y.reshape((len(train_y),1))

            return train_xx,train_yy

        def nonlinear_data(self,eps,author):

            if author == 'TJ':
                X = np.arange(0,10,0.005)
                Y = np.sin(4 * X) + np.random.normal(0,1/2,len(X))
                #X = np.append(X,np.random.choice(np.arange(0,10,0.0,20, replace = False))
                X = np.append(X,np.arange(0,10,0.1))
                Y = np.append(Y,np.repeat(7,400))
                Y = Y[np.argsort(X,axis = 0)]
                X = np.sort(X,axis = 0)

                train_xx = X
                train_yy = Y

                train_xx = train_xx.reshape((len(train_xx),1))
                train_yy = train_yy.reshape((len(train_yy),1))

                return train_xx,train_yy

            elif author == 'PV1':
                data = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/data4.txt',sep = ' ')
                X = data.iloc[:,0].values
                Y = data.iloc[:,1].values
                
                train_xx = X.reshape((len(X),1))
                train_yy = Y.reshape((len(Y),1))
                    
                return train_xx,train_yy
                
            elif author == 'PV2':
                data = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/data5.txt', sep = ' ')
                X = data.iloc[:,0].values
                Y = data.iloc[:,1].values
                
                train_xx = X.reshape((len(X),1))
                train_yy = Y.reshape((len(Y),1))

                return train_xx,train_yy

            elif author == 'TJ2':

                dict_data = {}
                
                for eps in eps:
                    
                    data = p.read_csv('https://raw.githubusercontent.com/jankalinaUI/Datasets/master/Data/data_eps_' + str(eps) + '.csv', 
                                      sep = ',',dtype = np.float64)
                    
                    
                    dict_data[str(eps)] = data.values
                
                    
                return dict_data
        
        def auto_final(self,eps):
            
            dict_data = {}
            
            for eps in eps:
                
                data = p.read_csv('https://github.com/jankalinaUI/Datasets/blob/master/Data_auto/auto_eps_' + str(eps) + '.csv',sep = ',',dtype = np.float64)
                
                dict_data[str(eps)] = data.values
            return dict_data
