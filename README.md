"Predicting Work Functions of 2D Materials with Interpretable Machine Learning"

This is a small project where I tried to predict the work function of 2D materials using only composition-based features (Magpie-style features). 

What is inside:
1- 'work_function_prediction.py'  :  Main script. Loads the dataset, trains ML models (Random Forest + XGBoost), evaluates, and makes plots.
2- 'outputs/'  :  Where the script saves figures + models.

How to run:
I used Python 3.10.

Install packages:
''bash
pip install -r requirements.txt
''

Then run:
''bash
python work_function_prediction.py
''

Dataset (important note about C2DB): The dataset I used is derived from C2DB (work function values + composition features).  
Because of licensing/redistribution, I’m (not) putting the full dataset here by default.

The script expects:'Data/Dataset.csv'

So if you want to reproduce the run, you need to obtain the data properly (official source with permission), then put it there.


What results did I get:
From my report:
1- RF around R² ~ 0.806, MAE ~ 0.339 eV  
2- XGBoost around R² ~ 0.811, MAE ~ 0.335 eV

I also did feature importance + SHAP just to understand the trends (not claiming deep physics from it).

Files you probably don’t need:
If you just care about code, you can ignore:
.. 'outputs/models/*.pkl'
..'outputs/figures/*.png'
