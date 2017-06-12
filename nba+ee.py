from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle


def fitness():
    mean = 0
    for i in range(10):
        mean += NBA.runMLP()
    mean = mean/10
    return mean

class Chromossome:
    def __init__(self, genes = np.array(2*[0]), mutation_step = 1, num_mutations = 0, num_successful_mutations = 0):
        self.genes = genes
        self.mutation_step = mutation_step
        self.num_mutations = num_mutations
        self.num_successful_mutations = num_successful_mutations
        self.f_nba = NBA.runMLP()
        
    def get_mutation_vector(self, n_genes = 2):
        return np.random.normal(0, self.mutation_step, n_genes)
    
    def get_success_probability(self):
        if(self.num_mutations == 0):
            return 0;
        return float(self.num_successful_mutations) / float(self.num_mutations)

    def adjust_mutation_step(self, mutation_constant, success_rate):
        ps = self.get_success_probability()
        self.num_mutations = 0
        self.num_successful_mutations = 0
        if ps > success_rate:
            self.mutation_step /= mutation_constant
        elif ps < success_rate:
            self.mutation_step *= mutation_constant
            if(self.mutation_step  <  1e-15):
                self.mutation_step  = 1e-15
    
    def mutation_delta_exp (self, delta_mutation, n_genes = 2):
        new_mutation_step = self.mutation_step*np.exp(np.random.normal(0, delta_mutation))
        

        if(new_mutation_step <  1e-15):
            new_mutation_step = 1e-15
        elif(new_mutation_step > 1e10):
            new_mutation_step = 1e10
            
        new_genes = self.genes + np.random.normal(0, self.mutation_step, n_genes)
        self.mutation_step = new_mutation_step
        self.genes = new_genes
        

        
class EvolutionStrategy:
    def __init__(self, generations=10000, population_size=30, sons_per_iter = 200, init_mutation_step=1000, mutation_constant=0.95, delta_mutation = 1, \
                 iter_to_adjust = 5, elitist_suvivor = False, mutation_type = "delta_exp", recombination_type="random", parent_sel = "global"):
        self.generations = generations
        self.population_size = population_size
        self.sons_per_iter = sons_per_iter
        self.population = []
        self.init_mutation_step = init_mutation_step
        self.mutation_constant = mutation_constant
        self.success_rate = .2
        self.verbose = 0
        self.parent_sel = parent_sel
        self.f_nba = NBA.runMLP()
        self.count_adjust = 0
        self.delta_mutation = delta_mutation
        self.iter_to_adjust = iter_to_adjust
        self.recombination_type = recombination_type
        
    def init_population(self):
        self.population = []
        self.count_adjust = 0
        for i in range(self.population_size):
            self.population.append(Chromossome(genes = 2*np.random.uniform(0,1), mutation_step = self.init_mutation_step*np.random.random()))

    def apply_mutation(self):
        for i in range(len(self.population)):
            self.population[i].mutation_delta_exp(delta_mutation = self.delta_mutation)
    
    def parent_selection(self):
        return random.sample(self.population, 2)
    
    def apply_recombination(self):
        if(self.recombination_type == "mean"):
            self.mean_recombination()
        elif(self.recombination_type == "random"):
            self.random_recombination()
            
    def mean_recombination(self):
        new_population = []
        for sons in range(self.sons_per_iter):
            if(self.parent_sel == "local"):
                parents = self.parent_selection()
            genes_son = []
            for i in range(2):
                if(self.parent_sel == "global"):
                    parents = self.parent_selection()
                    
                genes_son.append((parents[0].genes[i] + parents[1].genes[i])/2)

            
            if(self.parent_sel == "global"):
                parents = self.parent_selection()                
            if(self.mutation_type == "delta_exp"):
                mutation_step_son = (parents[0].mutation_step + parents[1].mutation_step)/2
                new_population.append(Chromossome(genes = genes_son, mutation_step = mutation_step_son))
            else:
                mutation_step_son = (parents[0].mutation_step + parents[1].mutation_step)/2
                num_mutations_son = parents[0].num_mutations
                num_successful_mutations_son = (parents[0].num_successful_mutations + parents[1].num_successful_mutations)/2
                new_population.append(Chromossome(genes = genes_son, mutation_step = mutation_step_son, \
                                                  num_successful_mutations = num_successful_mutations_son, \
                                                  num_mutations = num_mutations_son
                                                  ))
                
        if(self.elitist_suvivor):
            for chromossome in new_population:
                self.population.append(chromossome)
        else:
            self.population = new_population

               
    def random_recombination(self):
        new_population = []
        for sons in range(self.sons_per_iter):
            if(self.parent_sel == "local"):
                parents = self.parent_selection()
            genes_son = []
            for i in range(2):
                if(self.parent_sel == "global"):
                    parents = self.parent_selection()
                parent_select = np.random.randint(0,2)
                genes_son.append(parents[parent_select].genes[i])
            #endFor
            
            if(self.parent_sel == "global"):
                parents = self.parent_selection()                
            if(self.mutation_type == "delta_exp"):
                parent_select = np.random.randint(0,2)
                mutation_step_son = parents[parent_select].mutation_step
                new_population.append(Chromossome(genes = genes_son, mutation_step = mutation_step_son))
            else:
                mutation_step_son = parents[parent_select].mutation_step
                num_mutations_son = parents[0].num_mutations
                num_successful_mutations_son = parents[parent_select].num_successful_mutations
                new_population.append(Chromossome(genes = genes_son, mutation_step = mutation_step_son, \
                                                  num_successful_mutations = num_successful_mutations_son, \
                                                  num_mutations = num_mutations_son
                                                  ))
        #endFor
        if(self.elitist_suvivor):
            for chromossome in new_population:
                self.population.append(chromossome)
        else:
            self.population = new_population

            

    def apply_selection(self):
        self.population.sort(key=lambda chromossome : chromossome.fitness(), reverse=True)	
        self.population = self.population[:self.population_size]

class NBA:

    def __init__(self):
        pass
    
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

    def runNBA(self):
        es = EvolutionStrategy(generations=1000, population_size=30, sons_per_iter = 200,\
                               init_mutation_step=100, mutation_constant=0.95, delta_mutation = 1.8, \
                               iter_to_adjust = 5, elitist_suvivor = False,\
                               recombination_type="mean", parent_sel = "global")

    
        
def main():
    n = NBA()
    n.runNBA()
                
if __name__ == "__main__":
    main();
