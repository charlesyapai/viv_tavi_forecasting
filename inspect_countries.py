
import pandas as pd
df = pd.read_csv("simulation_run_v1/data/un_population_data/Population Projections - Male, Korea.csv")
print(df.iloc[:, 2].unique())
