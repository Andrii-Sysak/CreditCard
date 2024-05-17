Business needs
  
    A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction
    
Requirements

    python 3.7

    matplotlib==3.8.3
    numpy==1.26.4
    pandas==2.2.1
    scipy==1.12.0
    seaborn==0.13.2
    scikit-learn==1.4.1.post1
    imbalanced-learn==0.12.0
    category-encoders==2.6.3
    patsy==0.5.6
    statsmodels==0.14.2
    sklearn==1.4.2

Running:

    To run the demo, execute:
        python predict.py

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'Avg_Utilization_Ratio' column with the result value.

    The input is expected  csv file in the same folder with a name <new_data.csv>. The file shoul have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction y.
    After running the script the "finalized_model.saw" will be created.
    Run the training script:
        python train.ipynb

    Model metrics are:
    Mean Squared Error: 0.0005230640917334247   
    Mean Absolute Error: 0.014197086138953714   
    R-squared: 0.9933248764319517
    Explained Variance Score: 0.9933333305788559
    There is no fraud check.