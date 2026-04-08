Task 2 - Data Cleaning & Missing Value Handling
# edutech_task2
Data Visualization: Utilized Seaborn heatmaps to visually identify missing data patterns.

Feature Dropping: Removed the 'deck' column entirely due to an excessively high rate (77%) of missing values.

Numerical Imputation: Handled missing 'age' values by imputing the median using Scikit-learn's SimpleImputer.

Categorical Imputation: Handled missing 'embarked' values by imputing the mode (most frequent category).

Outlier Treatment: Mitigated extreme values in the 'fare' column using the Interquartile Range (IQR) method, capping the maximum fare at 65.

Verification & Export: Validated the final dataset to confirm zero missing values and exported the pipeline's output as a clean CSV.
