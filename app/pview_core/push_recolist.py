import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np

# Set up the database connection
engine = create_engine('mysql+pymysql://root:@localhost:3306/dev')

'''
### 추천테이블 삽입 ###
df = pd.read_excel("/home/ubuntu/PviewServer/app/pview_core/recolist.xlsx", index_col=0)
df['created_at'] = datetime.now()

# Insert the data from the DataFrame into the database
df.to_sql(name='recommandation', con=engine, if_exists='append', index=False)

'''

### 화장품테이블 삽입 ###
df = pd.read_excel("/home/ubuntu/PviewServer/app/pview_core/cosdf_to_db.xlsx", index_col=0)
df['created_at'] = datetime.now()

df.insert(0, 'id', np.arange(1, len(df)+1))

# Insert the data from the DataFrame into the database
df.to_sql(name='cosmetic', con=engine, if_exists='append', index=False)