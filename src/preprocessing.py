# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_paysim(path):

    df=pd.read_csv(path)

    df=df.sample(
        n=10000,
        random_state=42
    )

    encoder=LabelEncoder()

    df['type']=encoder.fit_transform(
        df['type']
    )

    scaler=StandardScaler()

    df['amount']=scaler.fit_transform(
        df[['amount']]
    )

    processed=pd.DataFrame({

        'source':df['nameOrig'],
        'target':df['nameDest'],
        'amount':df['amount'],
        'timestamp':df['step'],
        'label':df['isFraud']

    })

    return processed


def preprocess_elliptic(
    feature_path,
    class_path,
    edge_path
):

    features=pd.read_csv(
        feature_path,
        header=None
    )

    classes=pd.read_csv(
        class_path
    )

    edges=pd.read_csv(
        edge_path
    )

    features=features.rename(
        columns={
            0:'txId',
            1:'timestamp'
        }
    )

    merged=features.merge(
        classes,
        on='txId'
    )

    merged=edges.merge(
        merged,
        left_on='txId1',
        right_on='txId'
    )

    merged['class']=merged[
        'class'
    ].replace({

        'unknown':0,
        '1':1,
        '2':0

    })

    processed=pd.DataFrame({

        'source':merged['txId1'],
        'target':merged['txId2'],
        'amount':1,
        'timestamp':merged['timestamp'],
        'label':merged['class']

    })

    return processed