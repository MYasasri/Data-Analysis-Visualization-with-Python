import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"D:\DATA SCIENCE TOOLBOX USING PYTHON PROGRAMMING\U.S._Chronic_Disease_Indicators.csv")
print("the first five columns of th eunicorn dataset \n")
print(df.head())
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(df.info())
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(df.describe)
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(df['StratificationCategoryID1'].unique())
print(df['StratificationCategoryID1'].nunique())
print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Missing Values:\n",df.isnull().sum())
print("Total Missing Values:\n",df.isnull().sum().sum())
print("------------------------------------ After Handling missimg values-----------------------------------------------------------------------------------------------------------------------------")
df['DataValue']=df['DataValue'].fillna(df['DataValue'].mean())
df['Response']=df['Response'].fillna("NA")
df['DataValueAlt']=df['DataValueAlt'].fillna(df['DataValueAlt'].mean())
df['DataValueFootnoteSymbol']=df['DataValueFootnoteSymbol'].fillna(df['DataValueFootnoteSymbol'].mode()[0])
df['DataValueFootnote']=df['DataValueFootnote'].fillna(df['DataValueFootnote'].mode()[0])
df['LowConfidenceLimit']=df['LowConfidenceLimit'].fillna(df['LowConfidenceLimit'].mean())
df['HighConfidenceLimit']=df['HighConfidenceLimit'].fillna(df['HighConfidenceLimit'].mean())
df['StratificationCategory2']=df['StratificationCategory2'].fillna("NA")
df['Stratification2']=df['Stratification2'].fillna("NA")
df['StratificationCategory3']=df['StratificationCategory3'].fillna("NA")
df['Stratification3']=df['Stratification3'].fillna("NA")
df['Geolocation']=df['Geolocation'].fillna(df['Geolocation'].mode()[0])
df['ResponseID']=df['ResponseID'].fillna("NA")
df['StratificationCategoryID2']=df['StratificationCategoryID2'].fillna("NA")
df['StratificationID2']=df['StratificationID2'].fillna("NA")
df['StratificationCategoryID3']=df['StratificationCategoryID3'].fillna("NA")
df['StratificationID3']=df['StratificationID3'].fillna("NA")
print("Missing Values:\n",df.isnull().sum())
print("Total Missing Values:\n",df.isnull().sum().sum())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
numerical_columns=df.select_dtypes(include=['float64','int64']).columns
Q1=df[numerical_columns].quantile(0.25)
Q3=df[numerical_columns].quantile(0.75)

iqr=Q3 - Q1 
lb=Q1 - 1.5 *iqr
ub=Q3 + 1.5 * iqr
outlier_iqr=((df[numerical_columns] < lb) | (df[numerical_columns] >ub))
print("Outliers :- ", outlier_iqr)
plt.figure(figsize=(12, 6))
df[numerical_columns].boxplot(rot=45)
plt.title("Box Plot for Outlier Detection")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()
##print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

subset = df[(df['Question'] == 'Obesity among adults') & 
            (df['LocationDesc'] == 'California')]


plt.figure(figsize=(10, 5))
sns.lineplot(data=subset, x='YearStart', y='DataValue', marker='o')
plt.title('Obesity Trend in California Over Time')
plt.xlabel('Year')
plt.ylabel('Obesity Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
##print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

state_avg = df.groupby('LocationDesc')['DataValue'].mean().reset_index()


state_avg = state_avg.sort_values(by='DataValue', ascending=False)


plt.figure(figsize=(12, 6))
sns.barplot(data=state_avg, x='LocationDesc', y='DataValue',hue='LocationDesc', palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Chronic Disease Data Value by State')
plt.xlabel('State')
plt.ylabel('Average Data Value')
plt.tight_layout()
plt.show()
##print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
filtered_df = df[
    (df["Topic"] == "Diabetes") &
    (df["StratificationCategory1"] == "Overall") &
    (df["DataValue"].notna())
]
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=filtered_df,
    x="YearStart",
    y="DataValue",
    hue="LocationDesc", 
    alpha=0.7,
    palette="tab10"
)

plt.title("Diabetes Prevalence Over Time by State")
plt.xlabel("Year")
plt.ylabel("Data Value (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='State')
plt.show()

##print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
filtered_df = df[
    (df["Topic"] == "Diabetes") &
    (df["StratificationCategory1"] == "Overall") &
    (df["DataValue"].notna())
]
state_avg = filtered_df.groupby("LocationDesc")["DataValue"].mean().sort_values(ascending=False)

top_states = state_avg.head(10)

plt.figure(figsize=(8, 8))
plt.pie(top_states, labels=top_states.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 10 States by Average Diabetes Prevalence")
plt.axis('equal')
plt.show()

print("-----------------------------------Skewness of Diabetes---------------------------------------------------------------------------------------------------------------------------------")

filtered_df = df[
    (df["Topic"] == "Diabetes") &
    (df["StratificationCategory1"] == "Overall") &
    (df["DataValue"].notna())
]

data_skew = filtered_df["DataValue"].skew()
print(f"Skewness of Diabetes DataValue: {data_skew:.2f}")

plt.figure(figsize=(8, 5))
sns.histplot(filtered_df["DataValue"], kde=True, bins=20)
plt.title(f"Distribution of Diabetes DataValue\n(Skewness = {data_skew:.2f})")
plt.xlabel("Data Value (%)")
plt.ylabel("Frequency")
plt.show()

##print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

filtered_df = df[df["DataValue"].notna()]

numeric_df = filtered_df[["DataValue", "LowConfidenceLimit", "HighConfidenceLimit"]]

corr = numeric_df.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Chronic Disease Metrics")
plt.show()

