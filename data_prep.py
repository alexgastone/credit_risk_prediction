import numpy as np
import pandas as pd

def quantiles_transf(df, df_col):
    q_hi = df_col.quantile(0.999)
    q_low = df_col.quantile(0.001)
    df = df[(df_col>q_low) & (df_col<q_hi)]
    return df

def clean_app(df_app):
    df_app.drop_duplicates('ID', keep='last', inplace=True) 
    df_app.drop('OCCUPATION_TYPE', axis=1, inplace=True) 

    for col in ['CNT_CHILDREN','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS']:
        df_app = quantiles_transf(df_app, df_app[col])

    gender_dict = {'M':0, 'F':1}
    df_app['CODE_GENDER'] = df_app.CODE_GENDER.map(gender_dict)

    yn_dict = {'N':0, 'Y':1}
    for col in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df_app[col] = df_app[col].map(yn_dict)

    df_app = pd.get_dummies(df_app, prefix=['HOUSING', 'INCOME', 'EDUCATION', 'FAMILY'], 
                     columns=['NAME_HOUSING_TYPE','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', \
                         'NAME_FAMILY_STATUS'])
    return df_app

def clean_credit(df_credit):
    df_credit['STATUS'].replace({'X': 0, 'C' : 0, '0' : 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6},\
        inplace=True)
    
    # group by ID, keep most recent
    df_credit_grouped = df_credit.groupby('ID').agg(max).reset_index()    

    return df_credit_grouped

def join_df(df_app, df_credit):
    df = df_app.join(df_credit.set_index('ID'), on='ID', how='inner')
    df.drop(['MONTHS_BALANCE'], axis=1, inplace=True)
    df = df.set_index('ID')

    return df

def main():
    df_app = pd.read_csv("application_record.csv")
    df_credit = pd.read_csv("credit_record.csv")

    df_app = clean_app(df_app)
    df_credit = clean_credit(df_credit)

    df = join_df(df_app, df_credit)

    return df

if __name__ == '__main__':
    main()