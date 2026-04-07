from load_data import load_data
df = load_data("transactions.csv")

from sklearn.preprocessing import StandardScaler

def preprocess(df):

    sc = StandardScaler()
    df["Amount"] = sc.fit_transform(df[["Amount"]])
    df.drop(columns= "Time", inplace = True)
    return df







