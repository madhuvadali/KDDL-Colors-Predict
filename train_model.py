import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Prepare the data
data = {
    "Power (W)": [300, 270, 240, 210, 180, 150, 120],
    "10 mm/s": [
        "(255,255,255)", "(126,125,126)", "(129,124,118)", "(122,119,114)",
        "(122,120,115)", "(117,117,116)", "(108,110,115)"
    ],
    "20 mm/s": [
        "(121,117,115)", "(136,128,123)", "(142,135,130)", "(139,136,131)",
        "(129,130,128)", "(108,117,129)", "(105,90,103)"
    ],
    "30 mm/s": [
        "(128,124,119)", "(148,137,131)", "(156,146,136)", "(149,149,142)",
        "(140,148,146)", "(81,76,112)", "(137,107,79)"
    ],
    "40 mm/s": [
        "(147,133,126)", "(159,145,133)", "(163,159,144)", "(152,158,151)",
        "(130,146,157)", "(121,92,91)", "(154,129,90)"
    ],
    "50 mm/s": [
        "(162,147,129)", "(169,158,138)", "(159,164,154)", "(137,150,159)",
        "(116,128,151)", "(148,119,86)", "(164,140,95)"
    ],
    "60 mm/s": [
        "(166,157,136)", "(169,168,148)", "(150,164,162)", "(133,149,161)",
        "(115,107,130)", "(160,137,97)", "(169,152,119)"
    ],
    "70 mm/s": [
        "(165,165,148)", "(153,167,164)", "(134,154,166)", "(124,133,150)",
        "(136,116,114)", "(161,136,91)", "(175,157,122)"
    ],
    "80 mm/s": [
        "(155,165,158)", "(142,164,169)", "(123,144,163)", "(122,126,145)",
        "(142,121,107)", "(165,145,110)", "(175,160,132)"
    ],
    "90 mm/s": [
        "(143,161,161)", "(124,151,169)", "(116,130,154)", "(120,114,133)",
        "(148,116,76)", "(169,149,111)", "(177,165,141)"
    ],
    "100 mm/s": [
        "(136,158,164)", "(108,137,165)", "(117,121,145)", "(125,113,127)",
        "(158,135,101)", "(174,155,119)", "(178,167,145)"
    ],
}

df = pd.DataFrame(data)

# Reshape the data
df_long = df.melt(id_vars=["Power (W)"], var_name="Scan Speed (mm/s)", value_name="RGB")
df_long["Scan Speed (mm/s)"] = df_long["Scan Speed (mm/s)"].str.extract("(\d+)").astype(int)
df_long[["R", "G", "B"]] = df_long["RGB"].str.strip("()").str.split(",", expand=True).astype(int)
df_long = df_long.drop(columns=["RGB"])

# Inputs and Outputs
X = df_long[["Power (W)", "Scan Speed (mm/s)"]]
y = df_long[["R", "G", "B"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("Model trained and saved as 'model.pkl'")
