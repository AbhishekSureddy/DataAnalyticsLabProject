
import numpy as np
import pandas as pd

from missingpy import MissForest

col_no = np.arange(3,13)
col_no1 = np.arange(15,19)
cat_col_no = np.append(col_no, col_no1)

le_df = pd.read_csv('./uk-road-safety-accidents-and-vehicles/le_df.csv')
cols = le_df.columns


imputer = MissForest(random_state= 1)

for i in range(1,70):
    data_imputed = imputer.fit_transform(le_df[i*len(le_df) // 70: (i+1)*len(le_df)//70], cat_vars=cat_col_no)
    df = pd.DataFrame(data_imputed, columns=cols)
    df.to_csv('mf_'+str(i)+'.csv', index = False)
    print("progress :",i)