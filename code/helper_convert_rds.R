library(readr)
#run from vscode terminal: 
# cd /Users/emilyguan/Downloads/EndoScribe/pep_prediction/AlbertCodeFiles/pep_risk-master/pep_risk_app
# Rscript ../code/helper_convert_rds.r

# Training data
write_csv(readRDS("data/train_impute.rds"), "data/train_impute.csv")
write_csv(readRDS("data/train_new.rds"), "data/train_new.csv")

# Variable names (labels)
write_csv(readRDS("data/var_names.rds"), "data/var_names.csv")