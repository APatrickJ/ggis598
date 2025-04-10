from pathlib import Path

import pandas as pd


def read_labels(path: Path) -> pd.DataFrame:
    """Reads the LFMC labels CSV file.

    From the Globe-LFMC-2.0 paper:

    "For remote sensing applications, it is recommended to average the LFMC measurements taken on
    the same date and located within the same pixel of the product employed in the study. The
    choice of which functional type to include in the average can be guided by the land cover type
    of that pixel. For example, in open canopy forests, both trees and shrubs (or grass) could be
    included."
    """
    data = pd.read_csv(path)
    grouped = data.groupby(
        [
            "latitude",
            "longitude",
            "sampling_date",
        ],
        as_index=False,
    ).agg(
        {
            "site_name": "first",
            "sorting_id": "first",
            "lfmc": "mean",
            "state_region": "first",
            "country": "first",
        }
    )
    grouped["sampling_date"] = pd.to_datetime(grouped["sampling_date"]).dt.date
    return grouped
