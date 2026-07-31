curl -X POST "http://churn-prediction-alb-654855468.us-east-1.elb.amazonaws.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "Yes",
      "tenure": 48,
      "PhoneService": "Yes",
      "MultipleLines": "Yes",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "Yes",
      "DeviceProtection": "Yes",
      "TechSupport": "Yes",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Two year",
      "PaperlessBilling": "No",
      "PaymentMethod": "Bank transfer (automatic)",
      "MonthlyCharges": 59.4,
      "TotalCharges": 2851.2
    }'