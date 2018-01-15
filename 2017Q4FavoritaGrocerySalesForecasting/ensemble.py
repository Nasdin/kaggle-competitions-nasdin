import numpy as np 
import pandas as pd 

###  Ensemble of several models



filelist = ['output/lgb.csv',
             'output/LGBM.csv',
             'output/aggr.csv',
	'output/Median-based.csv',
'output/decisiontree.csv',
'output/elasticnet.csv',
'output/extratrees1.csv',
'output/extratrees2.csv',
'output/gradientboosting1.csv',
'output/gradientboosting2.csv',
'output/gradientboosting3.csv',
'output/hubermethod.csv',
'output/linear_model.csv',
'output/RandomForest01.csv',
'output/RandomForest02.csv',
'output/Ridge_method.csv',



]

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)

concat_df.columns = [str(x) for x in range(1,len(filelist)+1)]
print(concat_df.columns)
concat_df["unit_sales"] = (0.625*concat_df['1'] + 0.625*concat_df['2']+ 0.625*concat_df['3']+ 0.625*concat_df['4']+ 0.625*concat_df['5']+ 0.625*concat_df['6']+ 0.625*concat_df['7']+ 0.625*concat_df['8']+ 0.625*concat_df['9']+ 0.625*concat_df['10']+ 0.625*concat_df['11']+ 0.625*concat_df['12']+ 0.625*concat_df['13']+ 0.625*concat_df['14']+ 0.625*concat_df['15']+0.625*concat_df['16'] )
concat_df[["unit_sales"]].to_csv("output/final_ensemble.csv")
