X_columns = ["Attrition_Flag","Customer_Age","Gender","Dependent_count",
             "Education_Level","Marital_Status","Income_Category","Card_Category",
             "Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon",
             "Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal",
             "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt",
             "Total_Trans_Ct","Total_Ct_Chng_Q4_Q1"]

y_column = "Avg_Utilization_Ratio"

delete_columns = ['CLIENTNUM', 
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

mean_impute_columns = [
    'Customer_Age',
    'Months_on_book',
    'Total_Relationship_Count',
    'Credit_Limit',
    'Avg_Open_To_Buy',
    'Total_Trans_Amt',
    'Total_Trans_Ct'
    ]

mode_impute_columns = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
    'Dependent_count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
    ]

outlier_columns = [
    'Customer_Age',
    'Months_on_book',
    'Total_Relationship_Count',
    'Credit_Limit',
    'Avg_Open_To_Buy',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Dependent_count',
    'duration', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

scaling_columns = [
    "Customer_Age","Marital_Status","Card_Category",
    "Months_on_book","Credit_Limit","Total_Revolving_Bal",
    "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct"
]

