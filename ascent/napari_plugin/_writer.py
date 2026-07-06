import pandas as pd
import numpy as np

from typing import Any


def save_csv_tracks(path: str, data: Any, meta: dict) -> str:
    """
    Saves napari Points layer data to a CSV file.
    Maps internal feature names back to Ascent-style headers.
    """
    # 1. Extract coordinate data (N, 4) -> [t, z, y, x]
    coords = data
    
    # 2. Extract features using the keys defined in your readers
    features = meta.get('features', {})
    
    # Use .get() with fallbacks to prevent errors if features are missing
    track_ids = features.get('track_id', np.zeros(len(coords)))
    object_ids = features.get('object_id', np.arange(len(coords)))
    
    # 3. Create DataFrame with the specific headers from your save_csv() logic
    if coords.shape[1] == 4: # 3D + t
        df = pd.DataFrame(coords, columns=['t', 'z', 'y', 'x'])
    elif coords.shape[1] == 3: # 2D + t
        df = pd.DataFrame(coords, columns=['t','y','x'])

    df['TrackID'] = track_ids
    df['ObjectID'] = object_ids
    
    # Optional: Include provenance if it exists in the layer
    if 'provenance' in features:
        df['provenance'] = features['provenance']

    # 4. Save to disk
    # path is provided by the napari save dialog
    df.to_csv(path, index=False)
    
    return path