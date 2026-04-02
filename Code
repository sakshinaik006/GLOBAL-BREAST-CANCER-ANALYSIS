import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('breast_cancer_by_country.csv')

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head(3))
print(df.isnull().sum())
print(df.duplicated().sum())

target_cols = ['Incidence_Rate_Per_100K', 'Mortality_Rate_Per_100K', 'Five_Year_Survival_Pct']
for col in target_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]['Country'].tolist()
    print(f"{col}: {outliers if outliers else 'None'}")

df['Screening'] = df['Screening_Program'].map({True: 'Has Screening', False: 'No Screening'})

key_metrics = [
    'Incidence_Rate_Per_100K', 'Mortality_Rate_Per_100K', 'Five_Year_Survival_Pct',
    'Stage_I_II_Pct', 'Mammography_Coverage_Pct', 'Treatment_Access_Score'
]

print(df[key_metrics].describe().round(2))

CONT_COLORS = {
    'Americas': '#1565C0', 'Europe': '#2E7D32', 'Asia': '#E65100',
    'Africa': '#C62828', 'Oceania': '#00695C'
}

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df['Five_Year_Survival_Pct'], bins=12, kde=True, color='#1565C0', edgecolor='white', alpha=0.8, ax=ax)
ax.axvline(df['Five_Year_Survival_Pct'].mean(), color='red', ls='--', lw=2)
ax.axvline(df['Five_Year_Survival_Pct'].median(), color='green', ls=':', lw=2)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
order = df.groupby('Continent')['Incidence_Rate_Per_100K'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Continent', y='Incidence_Rate_Per_100K', palette=CONT_COLORS, order=order, ax=ax)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
top15 = df.nlargest(15, 'Incidence_Rate_Per_100K')
ax.barh(top15['Country'], top15['Incidence_Rate_Per_100K'], color=[CONT_COLORS[c] for c in top15['Continent']])
ax.invert_yaxis()
plt.tight_layout()
plt.show()

for col in key_metrics:
    if col != 'Five_Year_Survival_Pct':
        print(f"{col}: {df[col].corr(df['Five_Year_Survival_Pct']):.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[key_metrics].corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.tight_layout()
plt.show()

def plot_regression(x_col, y_col):
    fig, ax = plt.subplots(figsize=(9, 4))
    for cont, grp in df.groupby('Continent'):
        ax.scatter(grp[x_col], grp[y_col], color=CONT_COLORS.get(cont, 'grey'), alpha=0.7)
    m, b = np.polyfit(df[x_col], df[y_col], 1)
    xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(xs, m*xs+b, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

plot_regression('Treatment_Access_Score', 'Five_Year_Survival_Pct')
plot_regression('Mammography_Coverage_Pct', 'Five_Year_Survival_Pct')
plot_regression('Stage_I_II_Pct', 'Five_Year_Survival_Pct')

df['Case_Fatality_Ratio'] = (df['Deaths_2022'] / df['New_Cases_2022'] * 100).round(2)
df['Access_Tier'] = pd.cut(df['Treatment_Access_Score'], bins=[0, 40, 75, 100], labels=['Low', 'Medium', 'High'])
df['Log_Mammo_Coverage'] = np.log1p(df['Mammography_Coverage_Pct'])
df['Treatment_Access_Norm'] = (df['Treatment_Access_Score'] - df['Treatment_Access_Score'].min()) / \
                              (df['Treatment_Access_Score'].max() - df['Treatment_Access_Score'].min())

print(df[['Country', 'Case_Fatality_Ratio', 'Access_Tier', 'Log_Mammo_Coverage', 'Treatment_Access_Norm']].head())
