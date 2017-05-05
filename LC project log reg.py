# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:43:48 2017

@author: Perk
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import (cross_val_score, 
                                     StratifiedKFold, train_test_split)
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools

files, frames = [], []
letters = ['a','b','c','d']
qtrs = ['1','2','3','4']
for ltr in letters:
    files.append('LoanStats3{}_securev1.csv'.format(ltr))
for qtr in qtrs:
    files.append('LoanStats_securev1_2016Q{}.csv'.format(qtr))
for fi in files:
    with open(fi) as f:
        df = pd.read_csv(f,skiprows=1,low_memory=False)
        frames.append(df)

df = pd.concat(frames)
df['adj_annual_inc'] = df[['annual_inc','annual_inc_joint']].max(axis=1)
df['adj_dti'] = df[['dti','dti_joint']].min(axis=1)
df['mean_fico'] = df[['fico_range_low','fico_range_high']].mean(axis=1)
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'],format='%b-%Y')
df['issue_d'] = pd.to_datetime(df['issue_d'],format='%b-%Y')
df['credit_length'] = (df['issue_d'] - 
                          df['earliest_cr_line'] ) / np.timedelta64(1,'M')
df['loan_status'] = df['loan_status'].replace(
        'Does not meet the credit policy. Status:Fully Paid','Fully Paid')
df['loan_status'] = df['loan_status'].replace(
        'Does not meet the credit policy. Status:Charged Off','Charged Off')
df = df.loc[df['loan_status'].isin(['Fully Paid','Charged Off'])]
df = df.dropna(axis=0, thresh=df.shape[1]/2)
df['int_rate'] = df['int_rate'].replace('%','',regex=True).astype('float')/100
df['revol_util'] = df['revol_util'].replace('%','',
                                            regex=True).astype('float')/100
df['emp_length'] = df['emp_length'].replace('n/a',0)
df['emp_length'] = df['emp_length'].replace('< 1 year',0)
df['emp_length'] = df['emp_length'].replace('10+ years',10)
df['emp_length'] = df['emp_length'].replace([' years',' year'],'',regex=True)
df['emp_length'] = df['emp_length'].astype('float')
df = df.loc[df['home_ownership'].isin(['RENT','OWN','MORTGAGE'])]
df = df.loc[df['dti'] < 100]
df = df.loc[df['revol_util'] <= 1.0]
df = df.loc[df['dti'] > 0]
purpose_df = pd.get_dummies(df['purpose'],prefix='Purpose')
veri_df = pd.get_dummies(df['verification_status'],drop_first=True)
home_df = pd.get_dummies(df['home_ownership'],prefix='Home')
term_df = pd.get_dummies(df['term'],drop_first=True)
loan_stat = pd.get_dummies(df['loan_status'], drop_first=True)
df.drop(['id','member_id','loan_amnt','grade','sub_grade','emp_title',
         'installment','funded_amnt_inv','term','home_ownership','issue_d',
         'pymnt_plan','url','desc','title','zip_code','purpose','addr_state',
         'verification_status','loan_status','earliest_cr_line',
         'fico_range_low','fico_range_high','mths_since_last_delinq',
         'mths_since_last_record','initial_list_status','out_prncp',
         'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp',
         'total_rec_int','total_rec_late_fee','recoveries',
         'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
         'next_pymnt_d','last_credit_pull_d','last_fico_range_high',
         'last_fico_range_low','mths_since_last_major_derog','policy_code',
         'application_type','annual_inc','annual_inc_joint','dti','dti_joint',
         'verification_status_joint','mths_since_rcnt_il',
         'mths_since_recent_bc_dlq','mths_since_recent_revol_delinq',
         'mo_sin_old_rev_tl_op'],axis=1,inplace=True)
df = pd.concat([loan_stat,df,home_df,term_df,purpose_df,veri_df], axis=1)
df['pct_tl_nvr_dlq'] = df['pct_tl_nvr_dlq'].fillna(100.0)
df = df.fillna(0.0)

y = df['Fully Paid']
X = df.iloc[:,1:]

#logit = sm.Logit(y,X)
#result = logit.fit()
#print result.summary()

#logit = LogisticRegression(C=0.01, penalty ='l1',
#                           class_weight='balanced',max_iter=200).fit(X,y)
#model = SelectFromModel(logit, prefit=True)
#new_X = SelectKBest(chi2,k=10).fit_transform(X,y)
#selected_params = X[model.get_support(indices=True)].columns.values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, 
                                                    stratify=y)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier(loss='log', penalty="l1", n_jobs=-1,
                    class_weight={0:4}).fit(X_train, y_train)
sfm = SelectFromModel(clf, prefit=True)
X_train = sfm.transform(X_train)
X_test = sfm.transform(X_test)

#model2 = SGDClassifier(loss='log',penalty='l1',n_jobs=-1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
skf = StratifiedKFold(4)

print "F-beta score (B=0.5): {}".format(metrics.fbeta_score(y_test,
                                        predicted,beta=0.5))
scores = cross_val_score(clf, X, y, scoring='f1', cv=skf)
print "4-Fold cross val F-scores: {}".format(scores)
print "F-scores average: {}".format(scores.mean())
print metrics.classification_report(y_test, predicted,
                                    target_names=['Charged Off','Fully Paid'])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Charged Off','Fully Paid'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Charged Off','Fully Paid'], 
                      normalize=True,
                      title="Logistic regression, L1 penalty, default weight:4")

plt.show()