#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import email
import email.policy
import random


# In[11]:


#get spam and ham emails data
PATH = 'ham-and-spam-dataset/hamnspam/'
ham_filenames = [name for name in os.listdir(PATH+'ham')]
spam_filenames = [name for name in os.listdir(PATH+'spam')]
random.shuffle(ham_filenames)  #shuffle first
random.shuffle(spam_filenames)


# In[13]:


#load emails
def load_email(is_spam,filename,path):
    '''
    Load email by its name and loacation
    
    @is_spam: whether email is spam or not
    @filename: email name
    @path: PATH 
    
    @return: email object
    '''
    if is_spam :
        path = path + 'spam/' + filename
    else :
        path = path + 'ham/' + filename
    with open(path,'rb') as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(False,name,PATH) for name in ham_filenames]
spam_emails = [load_email(True,name,PATH) for name in spam_filenames]

#remove not text type emails
ham_emails = [email for email in ham_emails if type(email.get_payload()) is str or len(email.get_payload())>1] 
spam_emails = [email for email in spam_emails if type(email.get_payload()) is str or len(email.get_payload())>1]

print('Number of spam emails:',len(spam_emails))
print('Number of ham emails:',len(ham_emails))


# In[72]:


#example
test_email = spam_emails[0]
print('Example email content:\n\n',test_email.get_payload()[:400])  #limit the size to 400 chars
print('\n')


# In[31]:


#stem words
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
ps = PorterStemmer()


# In[79]:


#process email's content
import re
import string
def process_email(content) :
    '''
    preprocesses the content of an email 
    
    and returns a dictionary with word as key and its frequency as value
    @content : email content (a string)
    @return : a counting dictionary 
    '''                                         
    if not isinstance(content,str) :       
        return {},''
    content = re.sub(r'<[^<>]+>', ' ',content)  ##strip all HTML
    content = str.lower(content) ##lower case
    
    #handle URLS with http:// or https://
    content = re.sub(r'(http|https)://[^\s]*','httpaddr ',content) 
    
    #handle email addresses
    #look for strings with @ in the middle
    content = re.sub(r'[^\s]+@[^\s]+','emailaddr',content)
    
    content = re.sub(r'[0-9]+', 'number ',content) #handle numbers
    content = re.sub(r'[$]+','dollar ',content) #handle $ sign 
    content = re.sub(r'[\n]+',' ',content) #remove \n
    #remove punctuaion
    content = re.sub(r'[{0}]'.format(string.punctuation),' ',content) 
    
    res = {}
    words = word_tokenize(content)
    for word in words :
        word = ps.stem(word)
        if len(word) > 11 :
            continue
        if len(word) <=1 :
            continue
        if not res.get(word):
            res[word] = 0
        res[word] += 1
    
    return res,content


# In[60]:


(res,content) = process_email(test_email.get_payload())
print('Email content becomes(only list 500 chars):\n\n',process_email(test_email.get_payload())[1][:500]) 
print('\n')
print('words(only list 5 example):\n',list(res.keys())[:5])


# In[83]:


#build vocabulary for spam emails
def build_vocab(emails) :
    '''
    build_vocab will build a vocabulary with words 
    appearing in the email content
    @emails : list of email
    @return : 
    '''
    assert isinstance(emails,list)
    dic = {}
    index = 0
    
    while index < len(emails) :
        email = emails[index]
        judge = email.get_payload()
        if type(judge) is not str:
            dict_email = {}
            for e in judge :
                dic_toadd = process_email(e.get_payload())
                for word in dic_toadd[0] :
                    if not dic_email.get(word):
                        dic_email[word] = 0
                    dic_email[word] += 1
        else :
            dic_email = process_email(judge)[0]
        for word in dic_email.keys() :
            if not dic.get(word) :
                dic[word] = 0
            dic[word] += dic_email[word]
        index+=1
    
    return dic
# print(test_email['Subject'])
vocab = build_vocab(spam_emails)
print('number of total words in spam emails:', len(vocab.keys()))
vocab = [word for word in vocab.keys() if vocab[word]>11]
print('number of words that with frequency than 11:', len(vocab))
n = len(vocab)


# In[85]:


#export vocabulary to a csv file
df_vo = pd.DataFrame(vocab)
df_vo = df_vo.rename(columns={0:'words'})
df_vo.to_csv('vocabulary.csv',index=False)


# In[86]:


#features


# In[197]:


import random
class emailToFeature:
    '''
    This is a class for building feature vectors
    '''
    def __init__(self,filename) :
        vocab = pd.read_csv(filename)
        vocab = list(vocab['words'])
        index = 0
        vocabulary = {}
        while index < len(vocab) :
            vocabulary[vocab[index]] = index
            index+=1
        self.d = len(vocab)
        self.vocab = vocabulary   
    
    def fea_vector(self,email) :
        '''
        return a numpy array(1Xn) representing the
        feature vector
        @email: input email can be both a string and email object
        '''
        if type(email) is str:
            judge = email
        else :
            judge = email.get_payload()
        if not type(judge) is str:
            dic_email = {}
            for e in judge :
                dic_toadd = process_email(e.get_payload())
                for word in dic_toadd[0] :
                    if not dic_email.get(word):
                        dic_email[word] = 0
                    dic_email[word] += 1
        else :
            dic_email = process_email(judge)[0]
            
        res = np.zeros((1,self.d))
        for word in dic_email.keys() :
            if not self.vocab.get(word):
                continue
            index = self.vocab[word]
            res[0,index] = 1
        return res
    
    def build_vectors(self,is_spam,emails) :
        '''
        build feature vectors
        
        @emails : list of ham or spam emails
        @return : numpy array representing feature vectors
        '''
        N = len(emails)  # N*d array
        fea_vectors = np.zeros((N,self.d+1))
        for i in range(N) :
            a = self.fea_vector(emails[i])
            fea_vectors[i,:-1] = a
        if is_spam :
            fea_vectors[:,self.d] = 1
        else :
            fea_vectors[:,self.d] = 0
        return fea_vectors       


# In[92]:


#construct feature vectors and export them to csv files
emailTof = emailToFeature('vocabulary.csv') #class object
spam_vectors = emailTof.build_vectors(True,spam_emails)
ham_vectors = emailTof.build_vectors(False,ham_emails)

index = list(range(n+1))
spam_df = pd.DataFrame(spam_vectors,columns=index)
spam_df.to_csv('spam_vectors.csv',index = False)
ham_df = pd.DataFrame(ham_vectors,columns=index)
ham_df.to_csv('ham_vectors.csv',index=False)

print('size of spam feature vectors is:',spam_vectors.shape)
print('size of ham feature vectors is:',ham_vectors.shape)


# In[102]:


##split the data to training set, cross-validation set and test set (60%,20%,20%)
ham_training = ham_vectors[:1530]
ham_validation,ham_test = ham_vectors[1530:2039],ham_vectors[2039:]

spam_training = spam_vectors[:274]
spam_validation,spam_test = spam_vectors[274:365],spam_vectors[365:]

training = np.concatenate((ham_training,spam_training))
np.random.shuffle(training)
pd.DataFrame(training,columns=index).to_csv('training.csv',index=False)

cval = np.concatenate((ham_validation,spam_validation))
test = np.concatenate((ham_test,spam_test))
pd.DataFrame(cval,columns=index).to_csv('cross-validation.csv',index=False)
pd.DataFrame(test,columns=index).to_csv('test.csv',index=False)


# In[94]:


##training svm (linear kernel) and logistic regression


# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[99]:


#svm training
from sklearn import svm
def linearKernel(x1,x2):
    '''
    A linear kernel function 
    return the value of similarity
    '''
    return x1.dot(x2.T)


# In[114]:


#import training data
training = pd.read_csv('training.csv')
X_training = training.loc[:,'0':str(n-1)]
y_training = training.loc[:,[str(n)]]
y_training = np.array(y_training)
y_training = y_training[:,0]
X_training = np.concatenate( (np.ones((X_training.shape[0],1)),X_training), 1)

#import cross-validation data
validation = pd.read_csv('cross-validation.csv')
X_vali = validation.loc[:,'0':str(n-1)]
y_vali = validation.loc[:,[str(n)]]
y_vali = np.array(y_vali)
y_vali = y_vali[:,0]
X_vali = np.concatenate( (np.ones((X_vali.shape[0],1)),X_vali), 1)

#import test data
test = pd.read_csv('test.csv')
X_test = test.loc[:,'0':str(n-1)]
y_test = test.loc[:,[str(n)]]
y_test = np.array(y_test)
y_test = y_test[:,0]
X_test = np.concatenate( (np.ones((X_test.shape[0],1)),X_test), 1)
n = n+1 #number of columns


# In[119]:


#training and predicting test set
clf = svm.SVC(kernel=linearKernel)
clf.fit(X_training,y_training)
y_test_p = clf.predict(X_test)


# In[117]:


#functions to compute precision, recall, f1-scrore and error
def com_precision(y,y_p):
    '''
    y is true classification
    y_p is predicted classification
    '''
    length = len(y)
    num_pre1 = np.sum(y_p)
    num_correct = 0
    for i in range(length) :
        if y_p[i] != 1:
            continue
        if y[i] == 1:
            num_correct += 1
    return num_correct / num_pre1

def com_recall(y,y_p) :
    '''
    y is true classification
    y_p is predicted classification
    '''
    actual1 = np.sum(y)
    length = len(y)
    num_correct = 0
    for i in range(length) :
        if y[i]!=1 :
            continue 
        if y_p[i] == 1:
            num_correct += 1
    return num_correct / actual1

def com_f1Score(precision,recall):
    '''
    Compute F1-score
    ''' 
    return 2*precision*recall/(precision+recall)

def com_error(y,y_p):
    '''
    compute error
    '''
    vali = np.ones(len(y))
    vali = vali[y == y_p]
    return 1-np.sum(vali) / len(y_p)


# In[178]:


#result for svm 
precision,recall = com_precision(y_test,y_test_p),com_recall(y_test,y_test_p)
f1_score = com_f1Score(precision,recall)
error = com_error(y_test,y_test_p)
print('precision for svm is:',precision)
print('recall for svm is:',recall)
print('F1-score for svm is:',f1_score)
print('error for svm is:',error)
print('percentage of correct prediction is:','{0:.4}'.format((1-error)*100)+'%')


# In[248]:


#learning curve for svm
m = X_training.shape[0]
t_size = range(m//20,m,m//20)
error_training_svm,error_vali_svm = [],[]
for size in t_size:
    X,y = X_training[:size],y_training[:size]
    clf.fit(X,y)
    y_p_vali = clf.predict(X_vali)
    y_p_training = clf.predict(X)
    error_training_svm.append(com_error(y,y_p_training))
    error_vali_svm.append(com_error(y_vali,y_p_vali))


# In[254]:


plt.plot(list(t_size),error_vali_svm)
plt.plot(list(t_size),error_training_svm)
# fig = plt.gcf()
# fig.set_size_inches(4., 5)
plt.ylabel('Error')
plt.xlabel('Index of lambas')
plt.legend(['cross-validation','training'])
plt.show()


# In[ ]:





# In[ ]:





# In[158]:


##logistic training (with regularization)
import scipy.optimize as op

def sigmoid(z):
    '''
    Compute sigmaoid of a matrix 
    '''
    return 1/(1 + np.exp(-z))

def gradient(theta,x,y,lam):
    '''
    Compute gradient
    '''
    m , n = x.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    
    grad += lam*theta/m 
    grad[0,0] -=  lam * theta[0,0] / m
    return grad.flatten()

def costFunction(theta,x,y,lam):
    '''
    compute the cost
    '''
    epsilon = 1e-10
    m,n = x.shape 
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    first = np.log(sigmoid(x.dot(theta))+epsilon)
    second = np.log(1-sigmoid(x.dot(theta))+epsilon)
    first = first.reshape((m,1))
    second = second.reshape((m,1))
    all_m = y * first + (1 - y) * second
    J = -((np.sum(all_m))/m)
    
    J += np.sum(np.transpose(theta).dot(theta) * lam / (2*m) - lam*theta[0,0]**2 / (2*m))
    return J


# In[159]:


#training by logistic regression
lambdas = [0,0.05,0.1,0.5,1,2,4,8,16,32]
error_training = []
error_vali = []
X = X_training 
m,n = X.shape
for i in range(len(lambdas)):   
    initial_theta = np.zeros(n)
    Result = op.minimize(fun = costFunction,x0 = initial_theta, args = (X, y_training,lambdas[i]),
                                 method = 'TNC',
                                 jac = Gradient)
    theta = Result.x
    theta = theta.reshape((X.shape[1],1))
    pre_vali = X_vali.dot(theta)[:,0] 
    pre_training = X_training.dot(theta)[:,0]
    for row in range(len(pre_vali)):
        if pre_vali[row] > 0:
            pre_vali[row] = 1
        else :
            pre_vali[row] = 0
    for row in range(len(pre_training)):
        if pre_training[row] > 0:
            pre_training[row] = 1
        else :
            pre_training[row] = 0
    error_vali.append(com_error(y_vali,pre_vali))
    error_training.append(com_error(y_training,pre_training))


# In[160]:


#visualization
plt.plot(list(range(len(lambdas))),error_vali)
plt.plot(list(range(len(lambdas))),error_training)
fig = plt.gcf()
fig.set_size_inches(4., 5)
plt.ylabel('Error')
plt.xlabel('Index of lambas')
plt.legend(['cross-validation','training'])
plt.show()
best_lam = lambdas[error_vali.index(min(error_vali))]


# In[164]:


#training with the best lambda
print('The best lambda is:',best_lam)
initial_theta = np.zeros(n)
Result = op.minimize(fun = costFunction, x0 = initial_theta, args = (X, y_training,best_lam),                                   
                                 method = 'TNC',
                                 jac = gradient)
theta = Result.x
theta = theta.reshape((X.shape[1],1))
y_test_LR = X_test.dot(theta)[:,0]
for row in range(len(y_test_LR)):
    if y_test_LR[row] > 0:
        y_test_LR[row] = 1
    else :
        y_test_LR[row] = 0


# In[177]:


precision,recall = com_precision(y_test,y_test_LR),com_recall(y_test,y_test_LR)
f1_score = com_f1Score(precision,recall)
error = com_error(y_test,y_test_LR)
print('precision for logistic regression is:',precision)
print('recall for logistic regression is:',recall)
print('F1-score for logistic regression is:',f1_score)
print('error for logistic regression is:',error)
print('percentage of correct prediction is:','{0:.4}'.format((1-error)*100)+'%')


# In[256]:


##learning curve for LR
m,n = X_training.shape
t_size = range(m//20,m,m//20)
error_training,error_vali = [],[]
for size in t_size:
    X,y = X_training[:size],y_training[:size]
    initial_theta = np.zeros(n)
    Result = op.minimize(fun = costFunction, x0 = initial_theta, args = (X, y,best_lam),                                   
                                 method = 'TNC',
                                 jac = gradient)
    theta = Result.x
    theta = theta.reshape((X.shape[1],1))
    pre_vali = X_vali.dot(theta)[:,0] 
    pre_training = X.dot(theta)[:,0]
    for row in range(len(pre_vali)):
        if pre_vali[row] > 0:
            pre_vali[row] = 1
        else :
            pre_vali[row] = 0
    for row in range(len(pre_training)):
        if pre_training[row] > 0:
            pre_training[row] = 1
        else :
            pre_training[row] = 0
    error_vali.append(com_error(y_vali,pre_vali))
    error_training.append(com_error(y,pre_training))


# In[257]:


#visualization
plt.plot(list(t_size),error_vali)
plt.plot(list(t_size),error_training)
# fig = plt.gcf()
# fig.set_size_inches(4., 5)
plt.ylabel('Error')
plt.xlabel('Size of Training Set')
plt.legend(['cross-validation','training'])
plt.show()


# In[221]:


##just for fun
##i pick some junk email from my email account
junk_email = '''Unfortunately the venue for tonight's GrAdvantage Career Night has flooded, 
and University Center has declared the space unsafe to use. 
The Career Night will therefore be CANCELLED until further notice; 
we are very sorry for any inconvenience. We hope to reschedule this event in the near future; 
n the meantime, if you are interested in Consulting Careers or Professional Development opportunities at UCSD, 
please check out:'''  
emailTof2 = emailToFeature('vocabulary.csv') #class object
f_vector = emailTof2.fea_vector(email1)
f_vector = np.concatenate((np.array([[1]]),f_vector),1)
y_email1 = f_vector.dot(theta)[:,0][0]
if y_email1>0:
    print('predict email1 to be a spam email')
else :
    print('predict email1 to be non-spam email')


# In[223]:


junk_email2 = '''It's been about a year since any spam passing through 
our mail hubs has been worthy of much comment, but Miss Postmaster 
feels the need to deconstruct this one for the benefit of readers 
who are new to the Internet. Basically, it is a cheesy pitch like 
the one the radio commentator Jean Shepherd fell for when he was a boy. 
For ten cents, you could buy a telescope that would let you see the bones in your hand! 
He imagined it would let him see through anything...anything! 
So, this spam tells you how, for $24.95 US, you can find out anything about anyone! 
Maybe you'll even be able the see the bones in your hand, too!'''
f_vector = emailTof2.fea_vector(junk_email2)
f_vector = np.concatenate((np.array([[1]]),f_vector),1)
y_email2 = f_vector.dot(theta)[:,0][0]
if y_email2>0:
    print('predict email2 to be a spam email')
else :
    print('predict email2 to be non-spam email')


# In[224]:


junk_email3 = '''How are you?

Would you be interested in building an app for your business? We are a company specialized in mobile app development.

We have a team of over 125+ Android, IOS and Hybrid App Developers and have developed more than 300 mobile applications.

Our Mobile Application Development team specializes in building Native, Cross Compiled, and Hybrid Apps for iOS and Android using Titanium and Phone gap platforms.

Please let me know if you are interested, I am more than happy to provide you with a detailed proposal. Also, we will be happy to showcase our portfolio and vast gallery of our happy clients over a quick chat as per your interest.'''
f_vector = emailTof2.fea_vector(junk_email3)
f_vector = np.concatenate((np.array([[1]]),f_vector),1)
y_email3 = f_vector.dot(theta)[:,0][0]
if y_email3>0:
    print('predict email3 to be a spam email')
else :
    print('predict email3 to be non-spam email')


# In[ ]:




