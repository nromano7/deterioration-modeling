for i,x in enumerate(raw_data.dtypes.values):
    if x == object:
        col = data.columns[i] 
        print(col)

for i,x in enumerate(data.dtypes.values):
    if x == object:
        col = data.columns[i] 
        print(col)

  
## Preprocess ## 

# create dataframe
raw_data = pd.read_csv('./archive/data_export.csv')

# drop unwated columns
raw_data.drop(['8 - Structure Number'],axis=1,inplace=True)

# drop columns with nan
raw_data.dropna(axis=1,how='all',inplace=True)

# factorize categorical variables and create dict of preprocessed data
categorical_data = {}
for i,x in enumerate(raw_data.dtypes.values): 
    col = raw_data.columns[i] 
    if x == object:        
        categorical_data[col] = pd.factorize(raw_data[col])
        data_dict[col] = categorical_data[col][0]
    else:
        data_dict[col] = raw_data[col]
      
data = pd.DataFrame(data_dict)
        
        