
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import decisiontree as dt
import pandas as pd
import cPickle

class Classifier(object):
    """ Classe generique d'un classifieur
        Dispose de 3 méthodes :
            fit pour apprendre
            predict pour predire
            score pour evaluer la precision
    """
    def fit(self,data,y):
        raise NotImplementedError("fit non  implemente")
    def predict(self,data):
        raise NotImplementedError("predict non implemente")
    def score(self,data,y):
        return (self.predict(data)==y).mean()

def v2m(x):
    return x.reshape((x.size,1)) if len(x.shape)==1 else x

def mod_labels(y,mod):
    if(mod=='-1,1'):
        return y*2-1
    elif(mod=='0,1'):
        return (y+1)/2
    else:
        print 'Error! Wrong mod for error.'
        return None

def cross_validation(model,x,y,k):
    n=len(x)
    index_perm=np.random.permutation(range(n))
    scores=np.zeros((k))
    x_perm=x[[index_perm]]
    y_perm=y[[index_perm]]
    for index in range(k):        
        ik=int(float(index)*n/k)
        ikp1=int(float(index+1)*n/k)
        x_train=np.vstack((x_perm[:ik],x_perm[ikp1:]))
        y_train=np.concatenate((y_perm[:ik],y_perm[ikp1:]))
        x_test=x_perm[ik:ikp1]
        model.fit(x_train,y_train)        
        y_test=y_perm[ik:ikp1]
        scores[index]=model.score(x_test,y_test)
        #print('round '+str(index)+': '+str(scores[index])+'%.')
    return scores.mean()    




# In[1]:

def DecisionTree():
    return dt.DecisionTree()


# In[5]:


class Bayes(Classifier):
    def fit(self,x,y):
        self.nb_classes=len(np.unique(y))
        x_l=np.array([x[np.where(y==i)] for i in range(self.nb_classes)])
        mean=[[np.mean(x_l[j][:,i]) for i in range(x_l[j].shape[1])] for j in range(self.nb_classes)]
        std=[[np.std(x_l[j][:,i])+.1 for i in range(x_l[j].shape[1])] for j in range(self.nb_classes)]
        self.mean=np.array(mean)
        self.std=np.array(std)
    
    def predict(self,x):
        m,s=self.mean,self.std
        maxllog=np.zeros((x.shape[0]))
        for k in range(x.shape[0]):            
            maxllog[k]=np.argmax([np.sum([np.log(1/(np.sqrt(2*np.pi)*s[j][i])*np.exp((-0.5*(float(x[k][i]-m[j][i])/s[j][i])**2))) for i in range(x.shape[1])]) for j in range(m.shape[0])])
        return maxllog
    
    
    


# In[4]:

class KNN(Classifier):
    def __init__(self,k=3):
        self.k=k
        
    def fit(self,x,y):
        self.x=x
        self.y=y
        
    def predict(self,z):
        z_labels=np.zeros((len(z)))
        for index,j in enumerate(z):
            dist=np.array([np.linalg.norm(i-j) for i in self.x])
            arg_dist=np.argsort(dist)[:self.k]
            vote=self.y[arg_dist]
            vote = vote.astype(int)
            z_labels[index]=np.argmax(np.bincount(vote))
            
        return z_labels
    
    def hyperparam_plot(datax,datay,param_range=range(1,6)):
        plt.figure(figsize=(7,7))
        knn_score=np.zeros((len(param_range)))
        for k in param_range:
            knn=KNN(k=k)
            knn_score[k-1]=self.cross_validation(datax,datay,8)
        plt.plot(range(1,len(param_range)+1),knn_score,label='knn cross-val score')
        plt.legend()
        plt.show()
    


# In[6]:

def hinge(x,y,w):
    return np.maximum(0.,-(x.dot(w))*y)

def hinge_grad(x,y,w):
    return x*y

Fonction = namedtuple("Fonction",["f","grad","dim"]) #declaration de la structure
HINGE=Fonction(hinge,hinge_grad,6)

class Perceptron(Classifier):
    def __init__(self,loss=HINGE,max_iter=200,eps=0.00001):
        self.max_iter,self.eps=max_iter,eps
        self.w=None
        self.loss=loss
        
    def fit(self,datax,datay):
        datay=v2m(datay*2-1)
        self.w=np.random.random((len(datax[0]),1))-0.5
        self.max_iter=400
        for t in range(self.max_iter):
            hinge=self.loss[0](datax,datay,self.w)
            index_pos=np.where(hinge>0.)[0]
            grad=self.loss[1](datax,datay,self.w)
            self.w+=self.eps*np.array([np.sum(grad[index_pos],axis=0)]).T
        print 'hinge loss: '+str(np.sum(hinge))
        print 'fit score: '+str((1.-float(len(np.where(hinge>0.)[0]))/len(datax))*100)+'%.'
        
    def predict(self,datax):
         return (np.sign(datax.dot(self.w))+1)/2


# In[7]:

def sigmoid(x,l=1):
    return 2*(1.0/(1.+np.exp(-l*x))-.5)
    #return np.tanh(x)
def dSigmoid(x,l=1):
    return (l*np.exp(-l*x))/((1.0+np.exp(-l*x))**2)





class NN(Classifier):
    def __init__(self,layers,eps=.2,max_iter=100000): #layer = [2,2,1] ds exemple
        self.w = []
        for i in range(1, len(layers) - 1):
            weight = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.w.append(weight)
        weight = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.w.append(weight)
        self.maxIter = max_iter
        self.epsi = eps
    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)
        #biais ajouter au début sur les x
        uns = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((uns.T, x), axis=1)
        
        for k in range(self.maxIter):
            #batch
            i = np.random.randint(x.shape[0])
            a = [x[i]]

            for l in range(len(self.w)):
                    dot_value = np.dot(a[l], self.w[l])
                    activation = sigmoid(dot_value)
                    a.append(activation)
            
            # output layer
            error = y[i] - a[-1]
            
            deltas = [error * dSigmoid(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.w[l].T)*dSigmoid(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.w)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.w[i] += self.epsi*layer.T.dot(delta)

            #if k % 10000 == 0: print 'itération:', k

    def predict(self,x):
        a = np.concatenate((np.atleast_2d(np.ones(x.shape[0])).T, np.array(x)), axis=1)      
        for l in range(0, len(self.w)):
                    
            a = sigmoid(np.dot(a, self.w[l]))
            
        
        return (np.sign(a))

    def score(self,data,y):
        return (self.predict(data).T[0]==y).mean()


# In[ ]:



