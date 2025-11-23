# Data-Analysis-Visualization-with-Python
In my recent exploration of the U.S. Chronic Disease Indicators dataset, I dived deep into data cleaning, statistical analysis, and visual storytelling to uncover health trends across states, especially focusing on Diabetes and Obesity.

💡 Key Steps & Insights:

✅ Data Cleaning

Handled missing values using mean, mode, and placeholders depending on column type.
Addressed outliers with the Interquartile Range (IQR) method.
Improved data quality to ensure reliable visualizations and insights.

📊 Outlier Detection

Used log-scaled boxplots to highlight skewed numerical features and outliers in variables like DataValue, LowConfidenceLimit, and HighConfidenceLimit.

🌎 State-wise Health Patterns

Visualized Obesity trends in California showing changes over time.
Compared average chronic disease data across all U.S. states, revealing stark differences in disease prevalence.

💉 Diabetes-Focused Analysis

Tracked diabetes trends by state using scatter plots and line plots.
Identified the Top 10 states with the highest diabetes prevalence using a pie chart.
Found a slight positive skew (Skewness = ~0.6) in the diabetes DataValue, indicating more states with lower to moderate rates than extremely high rates.

🔗 Correlations & Confidence

Created a correlation heatmap for DataValue, LowConfidenceLimit, and HighConfidenceLimit, uncovering strong positive relationships – essential for understanding how survey confidence levels align with reported values.
🛠 Tools & Libraries Used:
 Pandas, NumPy, Matplotlib, Seaborn, Statsmodels
 
📌 This project not only refined my data wrangling and EDA (Exploratory Data Analysis) skills, but also emphasized the importance of clear visualization in communicating public health patterns.

📈 Whether you’re in data science or public policy, analyzing real-world datasets like these can power impactful decision-making and awareness.
