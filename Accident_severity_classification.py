# importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# reading the dataset
acc_info = pd.read_csv('Accident_Information.csv', low_memory = False, encoding = 'latin')
vehicle_info = pd.read_csv('Vehicle_Information.csv', low_memory = False, encoding = 'latin')

# getting preliminary insights from accident data
acc_info.shape
acc_info.head()
acc_info.info()
acc_info.describe()

# getting preliminary insights from vehicle data
vehicle_info.shape
vehicle_info.head()
vehicle_info.info()
vehicle_info.describe()

# merging the two datasets 
acc_vehicle_info = pd.merge(vehicle_info, acc_info, how = 'inner', on = 'Accident_Index')

mod_acc_vehicle_info = acc_vehicle_info.drop(['1st_Road_Class', '1st_Road_Number', '2nd_Road_Class', '2nd_Road_Number','Carriageway_Hazards', 'Date', 'Day_of_Week', 'Did_Police_Officer_Attend_Scene_of_Accident','Latitude','Local_Authority_(District)', 'Local_Authority_(Highway)','Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude','LSOA_of_Accident_Location','Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', 'Police_Force','Special_Conditions_at_Site', 'Speed_limit','Year_y', 'InScotland','Driver_Home_Area_Type', 'Driver_IMD_Decile','Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'Journey_Purpose_of_Driver','make', 'model','Propulsion_Code','Vehicle_Leaving_Carriageway','Vehicle_Location.Restricted_Lane','Vehicle_Reference', 'Vehicle_Type', 'Was_Vehicle_Left_Hand_Drive','Year_x'], axis = 1)
## dropping bad rows and columns
def drop_useless_rows(df,threshold = 0.8, target = None):
    bad_rows = df.loc[df.isnull().sum(axis = 1)/mod_acc_vehicle_info.shape[1]>= threshold]
    mod_df = df.drop(bad_rows.index, axis = 0)
    if target != None:
        class0 = len(bad_rows[bad_rows[target] == 0])
        class1 = bad_rows[target].value_counts().values[1]
        class0_pc, class1_pc = class0/len(df), class1/len(df)
        print("class0 dropped :", class0)
        print('class0 drop %.3f '% (class0_pc*100)+' %')
        print("class1 dropped :", class1)
        print('class1 drop %.3f ' % (class1_pc*100)+' %')
    return mod_df

#def drop_useless_cols(df, threshold = 0.4):
#    columns = df.columns
#    bad_cols = []
#    for col in columns:
#        if(df[col].isnull().sum()/len(df) >= threshold):
#            bad_cols.append(col)
#    mod_df = df.drop(bad_cols,axis = 1)
#    print('Dropped %d columns out of %d '%(len(bad_cols), len(columns)))
#    return mod_df
#    

import time
start = time.time()
miss_tag = ['Data missing or out of range','Unclassified','Not known','Unallocated']
col_names = mod_acc_vehicle_info.columns

for tag in miss_tag:
    for col in col_names:
        mod_acc_vehicle_info.loc[mod_acc_vehicle_info[col]== tag, col] = np.nan
       
end = time.time()

print("exec time :", end - start)

mod_acc_vehicle_info_2 = drop_useless_rows(mod_acc_vehicle_info, threshold = 0.5, target = None)

#getting the amount of Nan values in each columns
miss_plot = mod_acc_vehicle_info_2.isnull().sum(axis = 0).sort_values(axis = 0).plot(kind = 'barh', figsize = (20,15), fontsize = 10)
# plotting the mising data as bar plot
miss_plot.set_alpha(2)
miss_plot.set_title('Distribution of Nan values in processed features')
miss_plot.set_ylabel('Features')

#for i in mod_acc_vehicle_info.columns:
#    print(mod_acc_vehicle_info[i].value_counts(dropna = False))
    
severity_counts = acc_vehicle_info['Accident_Severity'].value_counts().plot(kind = 'bar',figsize = (8,5), fontsize = 10, rot = 0)
severity_counts.set_alpha(0.5)
severity_counts.set_title('Distribution of the dataset on the basis of Accident severity')
severity_counts.set_ylabel('% frequency')
totals = []
for i in severity_counts.patches:
    totals.append(i.get_height())    
total = sum(totals)
for i in severity_counts.patches:
    severity_counts.text(i.get_x()+0.12, i.get_height()+10,str(round((i.get_height()/total)*100, 2))+'%', fontsize=10)