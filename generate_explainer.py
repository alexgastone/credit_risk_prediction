import data_prep
import train_model
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

def main():
    classifier, _, X_test, _, y_test, _ = train_model.main()

    feature_descriptions = {
        "Code_Gender": "Gender of client",
        "Flag_Own_Car": "Has a car",
        "Flag_Own_Realty": "Has a property",
        "Cnt_Children": "Number of children",
        "Amt_Income_Total": "Annual income", 
        "Name_Income_Type": "Income category",
        "Name_Education_Type": "Education type",
        "Name_Family_Status": "Marital status",
        "Name_Housing_Type" : "Housing type",
        "Days_Birth": 'Days since birth',
        "Days_Employed": 'Days since employment',
        "Flag_Mobil": 'Has a mobile phone',
        "Flag_Work_Phone": 'Has a work phone',
        "Flag_Email": 'Has an email',
        "Cnt_Fam_Members": 'Family size'
    }

    explainer = ClassifierExplainer(classifier, X_test, y_test, shap='tree', \
        cats=[{"Name_Housing_Type":['House/appartment', 'With parents', 'Municipal apartment', 'Rented apartment',\
        "Office appartment", "Co-op appartment"]},{"Name_Income_Type": ['Working', 'Commercial Associate', \
            'Pensioner', 'State servant', 'Student']}, {"Name_Education_Type": ['Secondary', 'Higher education', \
                'Lower secondary', 'Academic degree']}, {"Name_Family_Status": ['Married', 'Single', 'Civil marriage',\
                    'Separated', 'Widow']}],\
                        descriptions=feature_descriptions, \
                            labels=['No loan/paid off', '1-29 days past due', '30-59 days past due', '60-89 days past due', \
                                '90-119 days past due', '120-149 days past due', 'overdue more than 150 days'], \
                                    target = "Credit Status")

    db = ExplainerDashboard(explainer, title="Credit Explainer", mode='external')
    
    explainer.dump("explainer.joblib")
    db.to_yaml("dashboard.yaml")
