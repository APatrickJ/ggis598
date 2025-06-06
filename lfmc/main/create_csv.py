import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from lfmc.core.const import CONUS_STATES, LABELS_PATH, Column

_SHEET_NAME = "LFMC data"

_COLUMNS = {
    "Sorting ID": Column.SORTING_ID,
    "Contact": Column.CONTACT,
    "Site name": Column.SITE_NAME,
    "Country": Column.COUNTRY,
    "State/Region": Column.STATE_REGION,
    "Latitude (WGS84, EPSG:4326)": Column.LATITUDE,
    "Longitude (WGS84, EPSG:4326)": Column.LONGITUDE,
    "Sampling date (YYYYMMDD)": Column.SAMPLING_DATE,
    "Protocol": Column.PROTOCOL,
    "LFMC value (%)": Column.LFMC_VALUE,
    "Species collected": Column.SPECIES_COLLECTED,
    "Species functional type": Column.SPECIES_FUNCTIONAL_TYPE,
}


def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def create_csv(
    input_excel_path: Path,
    output_csv_path: Path,
    start_date: datetime,
):
    logging.info("Reading the Excel file")
    df = pd.read_excel(input_excel_path, sheet_name=_SHEET_NAME, usecols=list(_COLUMNS.keys()))

    logging.info("Renaming the columns")
    df = df.rename(columns=_COLUMNS)

    logging.info("Filtering the DataFrame by date and location")
    df = df[df[Column.SAMPLING_DATE] >= start_date]
    logging.info("After filtering by date, the DataFrame has %d rows", len(df))
    df = df[(df[Column.COUNTRY] == "USA") & (df[Column.STATE_REGION].isin(CONUS_STATES))]
    logging.info("After filtering by location, the DataFrame has %d rows", len(df))

    logging.info("Writing the CSV file")
    df.to_csv(output_csv_path, index=False)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--input_excel_path",
        type=Path,
        default=Path("data/external/Globe-LFMC-2.0 final.xlsx"),
    )
    parser.add_argument(
        "--output_csv_path",
        type=Path,
        default=Path(LABELS_PATH),
    )
    parser.add_argument(
        "--start_date",
        type=parse_datetime,
        default=datetime(2017, 1, 1),
    )
    args = parser.parse_args()
    create_csv(
        args.input_excel_path,
        args.output_csv_path,
        args.start_date,
    )


if __name__ == "__main__":
    main()
