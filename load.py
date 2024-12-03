# %%
from darts.models import NBEATSModel

# %%
nbeats = NBEATSModel.load("./nbeats_model.pt")

# %%
df = nbeats.predict(15)

# %%
import pandas as pd

# %%
print(df.all_values())


# %%
