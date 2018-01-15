import numpy as np 
import pandas as pd 

###  Ensemble of several models



filelist = ['lgb.csv',
             '../input/ensemble/lgb.csv',
             '../input/catboost-starter-lb-0-517/cat1.csv']

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = ['1', '2','3']

concat_df["unit_sales"] = (0.2*concat_df['1'] + 0.45*concat_df['2'] + 0.35*concat_df['3'])
concat_df[["unit_sales"]].to_csv("final_ensemble.csv")
