from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import numpy
import pickle

def readData():
    f = open("GSW.data", "r")
    inputParameters = []
    outputClass = []
    for line in f:
        strArray = line.rstrip().split("\t")
        inputParameters.append(map(float, strArray[0:4]))
        outputClass.append(float(strArray[4]))
    return (inputParameters, outputClass)
#end readAbaloneData

def dataStratification(X, y, K):
    ##K-fold stratification
    skf = StratifiedKFold(n_splits=K)
    X_Train_Container = []
    X_Test_Container = []
    y_Train_Container = []
    y_Test_Container = []
    
    for train_index, test_index in skf.split(X, y):
        
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_Train = []
        y_Train = []
        X_Test = []
        y_Test= []
        
        for i in train_index:
            X_Train.append(X[i])
            y_Train.append(y[i])

        for i in test_index:
            X_Test.append(X[i])
            y_Test.append(y[i])

        X_Train_Container.append(X_Train)
        y_Train_Container.append(y_Train)
        X_Test_Container.append(X_Test)
        y_Test_Container.append(y_Test)
    #end for

    return  (X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container)
#end dataStratification


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')   

def runMLP(hd_layers_p = (30, ), activation_p = 'tanh' , solver_p = 'adam', learn_rate_p = 0.001, early_stopping_p = True, momentum_p = 0.9 ,max_iter_p = 1000 ):
    
    k_fold_lenght = 4
    scoreList = []
    MLPs = []
    y_Test_Total = []
    y_Pred_Total = []

    (X, y) = readData()
    #(X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container) = dataStratification(X, y, k_fold_lenght)

    #for i in range(k_fold_lenght):
    mlp = MLPRegressor(hidden_layer_sizes=hd_layers_p , activation=activation_p, solver=solver_p, learning_rate_init=learn_rate_p, max_iter=max_iter_p, momentum = momentum_p )
    
    #Get subsets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.33, random_state=42)

    print 'Size of X_Train: ' + str(len(X_Train))
    print 'Size of y_Train: ' + str(len(y_Train))
    print 'Size of X_Test: ' + str(len(X_Test))
    print 'Size of y_Test: ' + str(len(y_Test))

    y_Test_Total += y_Test
    #Scale
    scaler = StandardScaler()
    scaler.fit(X_Train)
    X_Train = scaler.transform(X_Train)
    X_Test = scaler.transform(X_Test)
    #Train
    mlp.fit(X_Train, y_Train)
    #print "Weight matrix", mlp.coefs_

    #Test
    scoreList.append(mlp.score(X_Test, y_Test))
    #y_Pred_Total += mlp.predict(X_Test)
    print "\tTest score ", "\t", scoreList[0]
    #end for
    #End Train and run MLP
    mean = 0
    for score in scoreList:
        mean += score

    mean = mean/len(scoreList)

        #print vetor1
    #print type(vetor1)

    #cnf_matrix = confusion_matrix(y_Test_Total,y_Pred_Total)
     
    #print cnf_matrix

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=['M','F','I'],
    #                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=['M','F','I'], normalize=True,
    #                     title='Normalized confusion matrix')


    #plt.show()

    return mean

def main():
    mean = runMLP()
    print "Taxa de acerto: ", mean
    #for i in range(40):
    #    neurons_1 = (i)*25
    #    if(neurons_1 == 0):
    #        neurons_1 = 5
    #    print neurons_1 , "\t", runMLP(hd_layers_p = (neurons_1, ))
    #print 1000 , "\t", runMLP(hd_layers_p = (1000, ))
    #camada = ()
    #for i in range(10):
    #    camada = camada + (25,)
    #    print camada, "\t", runMLP(hd_layers_p = camada)



    #for act in  ["identity", "logistic", "tanh", "relu"]:
        #for sol in ['lbfgs', 'sgd', 'adam']:
            #mean = 0
            #for i in range(5):
                #mean += runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol)
            #mean = mean/5
            #print sol,"\t", act, "\t", mean
    
    
    #for sol in ['lbfgs', 'sgd', 'adam']:
        #print sol, "\t", runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol)
    

    #for div in range(5):
    #    mean = 0
    #    d = 10**(div)
    #    learn_rate = 0.1/d
    #    for i in range(5):
    #        mean += runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol, learn_rate_p=learn_rate)
    #    print learn_rate, "\t", mean/5

    
    #hdls = 25
    #for hdls2 in range(24):
    #    hdlsP = (hdls+1, (hdls2+1)*5)
    #    print hdls, '\t', (hdls2+1)*5, '\t', runMLP(hd_layers_p = hdlsP)
    
    
                
if __name__ == "__main__":
    main();
