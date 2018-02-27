import xgboost as xgb


bst = xgb.Booster({'nthread':4}) #init model
bst.load_model("0002.model") # load data
