import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

PATH = 'Data/'
calculate_correlation = False

criminal_train_data = pd.read_csv(PATH + 'criminal_train.csv')
print criminal_train_data

#Header Columns
columns = list(criminal_train_data)
for key in columns:
    criminal_train_data = criminal_train_data[criminal_train_data[key] != -1]

print 'After Filtering the data for missing values (missing values represented by -1)'
print criminal_train_data

if calculate_correlation:
    corr = criminal_train_data.corr()
    print corr.head()
    corr.to_csv('correlation.csv')

train, test = train_test_split(criminal_train_data, test_size = 0.15)

isfather_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRHHSIZ2']
isfather_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRHHSIZ2']
child_u_18_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRKI17_2']
child_u_18_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRKI17_2']
iihh_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIHH65_2']
iihh_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIHH65_2']
proxy_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['PRXYDATA']
proxy_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['PRXYDATA']
tslhc_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['HLNVSOR']
tslhc_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['HLNVSOR']
hlrefused_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIMEDICR']
hlrefused_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIMEDICR']
poverty_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRCHMPUS']
poverty_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRCHMPUS']
cost_high_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IICHMPUS']
cost_high_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IICHMPUS']
noffer_employer_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRPRVHLT']
noffer_employer_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRPRVHLT']
hl_reported_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRINSUR4']
hl_reported_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRINSUR4']
hl_refused_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIINSUR4']
hl_refused_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIINSUR4']
impu_revised_child_u_18_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['OTHINS']
impu_revised_child_u_18_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['OTHINS']
hlinsu_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['CELLNOTCL']
hlinsu_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['CELLNOTCL']
imindi_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRFAMSVC']
imindi_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRFAMSVC']
cvhl_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIFAMSVC']
cvhl_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIFAMSVC']
imrc_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRWELMOS']
imrc_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRWELMOS']
ohl_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIWELMOS']
ohl_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIWELMOS']
ohl1_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRPINC3']
ohl1_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRPINC3']
ohl2_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IRFAMIN3']
ohl2_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IRFAMIN3']
ohl3_criminal = criminal_train_data[criminal_train_data['Criminal'] == 1]['IIPINC3']
ohl3_not_criminal = criminal_train_data[criminal_train_data['Criminal'] == 0]['IIPINC3']

#Decision Tree Classifier
c = DecisionTreeClassifier(min_samples_split=100)

#Features to use to train and test the model
features = ['IRHHSIZ2', 'IRKI17_2', 'IIHH65_2', 'PRXYDATA', 'HLNVSOR', 'IIMEDICR', 'IRCHMPUS', 'IICHMPUS', 'IRPRVHLT', \
            'IRINSUR4', 'IIINSUR4', 'OTHINS', 'CELLNOTCL', \
            'IRFAMSVC', 'IIFAMSVC', 'IRWELMOS', 'IIWELMOS', 'IRPINC3', 'IRFAMIN3', 'IIPINC3']

'''
X_train = train[features]
y_train = train['Criminal']
X_test = test[features]
y_test = test['Criminal']
'''


criminal_test_data = pd.read_csv(PATH + 'criminal_test.csv')
X_train = criminal_train_data[features]
y_train = criminal_train_data['Criminal']
X_test = criminal_test_data[features]
test_per_id = criminal_test_data['PERID']

dt = c.fit(X_train, y_train)
y_pred = c.predict(X_test)

#Save to file
print y_pred.astype(int)
print len(test_per_id)
np.savetxt('submission.csv', np.c_[test_per_id, y_pred], fmt="%d", delimiter=',')
'''
score = accuracy_score(y_test, y_pred) * 100
print 'Accuracy using decision tree: ', round(score, 1), '%'
print 'Precision Score (binary - default) is: ', precision_score(y_test, y_pred)
print 'R2 Score: ', r2_score(y_test, y_pred)
'''