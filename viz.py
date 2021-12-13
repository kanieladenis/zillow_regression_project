import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Show house vlaues create than mean by coorindates, shows a map
value_map = df[df.tax_value > df.tax_value.mean()]
sns.relplot(data=value_map, x='longitude', y='latitude', hue='tax_value')
plt.show()


# Shows house value distribution across counties
plt.figure(figsize=(10,5))
sns.displot(x=train.tax_value, hue=train.fips)
plt.xlabel('House Tax Value')
plt.show()


# Show heat map of features
corr_table = train.drop(columns=['fips']).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_table, cmap='Purples', annot=True, linewidth=0.5, mask= np.triu(corr_table))
plt.show()



# Shows that value increases with area & year_built
sns.relplot(x='year_built', y='area', data=train,  hue='tax_value', kind='scatter')
plt.xlabel('Year Built')
plt.ylabel('House Area Sqft')
plt.show()


#shows that increasd area tends to have increases bathrooms. Slight correlation to increased value.
sns.relplot(x='area', y='tax_value', data=train, hue='bathrooms', kind='scatter')
plt.xlabel('House Area in Sqft')
plt.ylabel('House Tax Value')
plt.show()

#show average tax value of homes across couty fips
sns.barplot(x=df.fips, y=df.tax_value)
plt.xlabel('Average Tax Amount')
plt.ylabel('County Identifer')
plt.show()



# Shows tax value distribution by fips
sns.displot(x=train.tax_value, hue=train.fips)
plt.xlabel('Home Tax Value')
plt.ylabel('Home Count')
plt.show()



# Shows increased area has year increased and more high areas have high value
sns.relplot(x='year_built', y='tax_value', data=train, hue='area', kind='scatter')
plt.xlabel('Year Built')
plt.ylabel('House Tax Value')
plt.show()

# compare area vs value by fips
sns.relplot(x='area', y='tax_value', data=train, hue='fips', kind='scatter')
plt.ylabel('House Tax Value')
plt.xlabel('House Area in Sqft')
plt.show()


# shows that 6037 has majority of properties, 6059 has more high value properties
sns.relplot(x='year_built', y='tax_value', data=train, hue='fips', kind='scatter')
plt.ylabel('House Tax Value')
plt.xlabel('Year Built')
plt.show()