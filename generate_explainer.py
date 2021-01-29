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

    # cats contains one-hot-encoded vectors
    explainer = ClassifierExplainer(classifier, X_test, y_test, shap='tree',\ 
                                    cats=[{"Name_Housing_Type":['Housing_Co-Op Apartment', 'Housing_House / Apartment','Housing_With Parents', 'Housing_Municipal Apartment', 'Housing_Rented Apartment',"Housing_Office Apartment"]},\
                                        {"Name_Income_Type": ['Income_Working', 'Income_Commercial Associate', 'Income_Pensioner', 'Income_State Servant', 'Income_Student']}, \
                                        {"Name_Education_Type": ['Education_Secondary / Secondary Special', 'Education_Higher Education', 'Education_Incomplete Higher','Education_Lower Secondary', 'Education_Academic Degree']}, \
                                        {"Name_Family_Status": ['Family_Married', 'Family_Single / Not Married', 'Family_Civil Marriage','Family_Separated', 'Family_Widow']}],\
                                    descriptions=feature_descriptions,\
                                    labels=['No loan/paid off', '1-29 days past due', '30-59 days past due', '60-89 days past due','90-119 days past due', '120-149 days past due', 'overdue more than 150 days'],\
                                    target = "Credit Status")

    db = ExplainerDashboard(explainer, title="Credit Explainer", mode='external', hide_pdp=True, \
        importances=True, model_summary=True, contributions=True, whatif=False,\
            shap_dependence=False, shap_interaction=False, decision_trees=False)
    
    explainer.dump("explainer.joblib")
    db.to_yaml("dashboard.yaml")


if __name__ == '__main__':
    main()