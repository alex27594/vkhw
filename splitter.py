from sklearn.model_selection import train_test_split


def split(df, test_size, random_state):
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=True,
                                     stratify=df.group, random_state=random_state)
    return df_train, df_test