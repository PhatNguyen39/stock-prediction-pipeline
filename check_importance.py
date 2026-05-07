from src.models.xgboost_model import XGBoostModel

model = XGBoostModel()
model.load('models/saved/latest_model.pkl')

imp = model.get_feature_importance()
print(imp.to_string(index=False))