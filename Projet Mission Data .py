#!/usr/bin/env python
# coding: utf-8

# Après nettoyage des datasets, nous nous sommes retrouvés avec la df "Data_Bonheur", une base de données nettoyée où les cinq datasets ont été mergés.

# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

file_path = "C:\\Users\\Ulysse\\Downloads\\data_bonheur.csv"
data = pd.read_csv(file_path)

display(data.head())

display(data.info())

numeric_data = data.select_dtypes(include=['float64', 'int64'])

display(numeric_data.head())

plt.figure(figsize=(10, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True,vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5)

plt.title('Heatmap des corrélations')
plt.show()


# In[26]:


plt.figure(figsize=(14, 12))
sns.clustermap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Clustermap des corrélations')
plt.show()


# In[27]:


print(data.columns)


# Grâce à la heatmap de corrélations, nous pouvons identifier les corrélations et donc les facteurs importants composant le bonheur dans cette base de données :
# 
# - economy_gdp_per_capita
# - family
# - health_life_expectancy
# - freedom
# - trust_government_corruption

# In[29]:


from sklearn.linear_model import LinearRegression

X = data[['economy_gdp_per_capita','family','health_life_expectancy','freedom','trust_government_corruption']]
y = data['happiness_score']

Bon_m = LinearRegression().fit(X, y)

print("coefficient :",Bon_m.coef_)
print("\nintercept :", Bon_m.intercept_)


# Nous pouvons voir que le coefficient (je finis après, constat habituel)

# In[39]:


from sklearn.model_selection import train_test_split

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

features = ['economy_gdp_per_capita', 'family', 'health_life_expectancy', 'freedom', 'trust_government_corruption']
target = 'happiness_score'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\ncoefficient :", model.coef_)
print("\nintercept :", model.intercept_)

data['predictions'] = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(data['year'], data['happiness_score'], color='royalblue', label='Happiness Score')
plt.scatter(data['year'], data['predictions'], color='red', label='Predictions')
plt.xlabel('Années')
plt.ylabel('Happiness Score')
plt.legend()
plt.title('Happiness Score et Prédictions')
plt.show()


# In[41]:


economy_gdp_per_capita = 1.340
family = 1.587
health_life_expectancy = 0.986
freedom= 0.596
Generosity = 0.153
trust_government_corruption = 0.393

feature_names = ['economy_gdp_per_capita', 'family', 'health_life_expectancy', 'freedom','trust_government_corruption']
my_data = pd.DataFrame([[economy_gdp_per_capita, family, health_life_expectancy, freedom, trust_government_corruption]], columns = feature_names) 
print(model.predict(my_data))


# In[ ]:





# In[ ]:




