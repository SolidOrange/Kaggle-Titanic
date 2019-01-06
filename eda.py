# EDA

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
fig = plt.figure(figsize=(18,400))
plt.subplots_adjust(hspace = 0.8)

plt.subplot2grid((5,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind='bar')
plt.title('Survived')

plt.subplot2grid((5,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title('Survived v. Age')

plt.subplot2grid((5,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind='bar')
plt.title('Class')

plt.subplot2grid((5,3),(1,0), colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind='kde')    
plt.title('Class by Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'))

plt.subplot2grid((5,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind='bar')
plt.title('Embarked')

plt.subplot2grid((5,3),(1,2))
df.Survived[df.Sex == 'male'].value_counts(normalize=True).plot(kind='bar')
plt.title('Male Survived')

plt.subplot2grid((5,3),(2,0))
df.Survived[df.Sex == 'female'].value_counts(normalize=True).plot(kind='bar')
plt.title('Female Survived')

plt.subplot2grid((5,3),(2,1))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind='bar')
plt.title('Survived by Sex')

plt.subplot2grid((5,3),(3,0), colspan=2)
for x in [1,2,3]:
    df.Survived[df.Pclass ==  x].plot(kind='kde')
plt.title('Survived by Class')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc=1)

plt.subplot2grid((5,3),(2,2))
df.Survived[(df.Sex == 'male') & df.Pclass == 1].value_counts(normalize=True).plot(kind='bar')
plt.title('Rich Men Survived')

plt.subplot2grid((5,3),(3,2))
df.Survived[(df.Sex == 'male') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind='bar')
plt.title('Poor Men Survived')

plt.subplot2grid((5,3),(4,0))
df.Survived[(df.Sex == 'female') & (df.Pclass == 1)].value_counts(normalize=True).plot(kind='bar')
plt.title('Rich Women Survived')

plt.subplot2grid((5,3),(4,1))
df.Survived[(df.Sex == 'female') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind='bar')
plt.title('Poor Women Survived')


plt.show()


















