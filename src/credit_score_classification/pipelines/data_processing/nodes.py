import pandas as pd

def map_bool(series: pd.Series) -> pd.Series:
    return series.map({'Yes': 1, 'No': 0})

def map_credit_mix(series: pd.Series) -> pd.Series:
    return series.map({'Good': 2, 'Standard': 1, 'Bad': 0})

def map_calendar(series: pd.Series) -> pd.Series:
    return series.map({'January': 1,
                       'February': 2,
                       'March': 3,
                       'April': 4,
                       'May': 5,
                       'June': 6,
                       'July': 7,
                       'August': 8,
                       'September': 9,
                       'October': 10,
                       'November': 11,
                       'December': 12})

def payment_mapping(series: pd.Series) -> pd.Series:
    return series.map({
        'High_spent_Large_value_payments': 6,#Successfully managing large debts provides the most positive contribution to the credit score.
        'Low_spent_Small_value_payments': 1,  # may limit the credit history and provide minimal contribution to the credit score
        'High_spent_Small_value_payments': 4, #Small payments can negatively affect the credit score if debts accumulate over time.
        'Low_spent_Large_value_payments': 3, #shows quick financial responsibility, positively affecting the credit score.
        'Low_spent_Medium_value_payments': 2, #contributes positively to the credit score by demonstrating debt management.
        'Low_spent_Small_value_payments': 1 #may limit the credit history and provide minimal contribution to the credit score
    }).fillna(0)  # Default value for unmapped entries
    
def count_loan_types(loan: pd.DataFrame) -> pd.DataFrame:
    # List of unique loan values
    unique_loan_types = ['Auto Loan',
                         'Credit-Builder Loan',
                         'Debt Consolidation Loan',
                         'Home Equity Loan',
                         'Mortgage Loan',
                         'No Loan',
                         'Not Specified',
                         'Payday Loan',
                         'Personal Loan',
                         'Student Loan']
    
    # Adding a new column for each unique loan type and checking how many times it appears
    for loan_type in unique_loan_types:
        cleaned_loan_type = loan_type.replace(' ', '_').replace('-', '_').lower()
        loan[cleaned_loan_type] = loan['type_of_loan'].apply(lambda x: x.count(loan_type) if isinstance(x, str) else 0)
    
    return loan


def preprocess_loan(loan: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the loan data.

    Args:
        loan: Raw data.
    Returns:
        Preprocessed data, with `credit_mix`, `payment_of_min_amount` and `month` converted to numeric, and
        `type_of_loan` column expanded into count columns. Columns not in further use were dropped as appropriate.
    """
    
    loan = count_loan_types(loan)
    
    loan.drop(["id", "customer_id", "name", "ssn", "type_of_loan"], axis=1, inplace=True)
    
    loan['payment_behaviour'] = payment_mapping(loan['payment_behaviour'])
    loan['payment_behaviour'] = pd.to_numeric(loan['payment_behaviour'], downcast='integer')
    
    loan['credit_mix'] = map_credit_mix(loan['credit_mix'])
    loan['credit_mix'] = pd.to_numeric(loan['credit_mix'], downcast='integer')
    
    loan['payment_of_min_amount'] = map_bool(loan['payment_of_min_amount'])
    loan['payment_of_min_amount'] = pd.to_numeric(loan['payment_of_min_amount'], downcast='integer')
    
    loan = pd.get_dummies(loan, columns=['occupation'], drop_first=True)
    
    loan['month'] = map_calendar(loan['month'])
    loan['month'] = pd.to_numeric(loan['month'], downcast='integer')
    
    return loan
