import pandas as pd

def load_data():
    df = pd.read_csv("data/data.csv")
    return df

def clean_data(df):
    df["director"] = df["director"].fillna("Unknown_director")
    df["cast"] = df["cast"].fillna("Unknown_cast")
    df["country"] = df["country"].fillna("Unknown_country")
    df.to_csv("data/cleaned_data.csv", index = False)
    return df

def filter_data(df):
    movies_df = df[df["type"] == "Movie"]
    release_year_df = df[df["release_year"] > 2015]
    movies_after_2015_df = df[(df["type"] == "Movie") & (df["release_year"] > 2015)]
    return movies_df, release_year_df, movies_after_2015_df

def run():
    df = load_data()
    df = clean_data(df)
    movies_df, release_year_df, movies_after_2015_df = filter_data(df)
    print("Dataset shape", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print(movies_df.head())
    print(release_year_df.head())
    print(movies_after_2015_df.head())

if __name__ == "__main__":
    run()