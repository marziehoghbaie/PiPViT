import pandas as pd
import numpy as np


def process_oct5K(csv_path):
    """

    Args:
        csv_path:

    Returns:
    keep only the drusen annotations
    """
    df = pd.read_csv(csv_path)
    print(df['class'].unique())
    drusen_classes = ['Softdrusen', 'Reticulardrusen', 'Harddrusen', 'SoftdrusenPED' ]
    drusen_df = df[df['class'].isin(drusen_classes)]
    print(drusen_df['class'].unique())
    drusen_df.to_csv('/all_bounding_boxes_only_drusen.csv', index=False)
    print(len(drusen_df))
    print(len(df['image'].unique()))


if __name__ == '__main__':
    csv_path = '/OCTDrusen/all_bounding_boxes.csv'
    process_oct5K(csv_path)