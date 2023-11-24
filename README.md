# Credit_card_default_prediction
Objective of the project-
To build a predictive model that leverages both the demographic characteristics of credit card owners and their payment history to estimate the likelihood of credit default. 

# Workflow
1) Data understanding
2) Data preprocessing
3) EDA
4) Data Preparation for model building
5) Model Building

# Dataset overview
ID: A unique identifier for each credit card holder.

LIMIT_BAL: The credit limit of the cardholder.

SEX: Gender of the cardholder (1 = male, 2 = female).

EDUCATION: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others).

MARRIAGE: Marital status (1 = married, 2 = single, 3 = others).

AGE: Age of the cardholder.

PAY_0 to PAY_6: Payment status for the last 6 months. It seems to represent the repayment status, with -1 meaning payment delay for one month, -2 meaning payment delay for two months, and so on.

BILL_AMT1 to BILL_AMT6: The bill amount for the last 6 months.

PAY_AMT1 to PAY_AMT6: The amount paid for the last 6 months.

default.payment.next.month: Whether the cardholder will default next month (1 = yes, 0 = no).
