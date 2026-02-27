from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_prepration_cat_con(X):

    cat = list(X.select_dtypes(include='category').columns) # include='str'
    con = list(X.select_dtypes(include='number').columns)

    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    cat_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    )

    pre = ColumnTransformer([
        ('cat',cat_pipe,cat),
        ('con',num_pipe,con)
    ]).set_output(transform='pandas')

    X_pre = pre.fit_transform(X)
    return X_pre

def data_prepration_cat(X):

    cat_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    ).set_output(transform='pandas')

    X_pre = cat_pipe.fit_transform(X)
    return X_pre

def data_prepration_con(X):

    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    ).set_output(transform='pandas')

    X_pre = num_pipe.fit_transform(X)
    return X_pre
