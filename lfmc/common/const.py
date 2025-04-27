from enum import StrEnum
from pathlib import Path

from frozendict import frozendict

CONUS_STATES = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

LABELS_PATH = Path("data/labels/lfmc_data_conus.csv")

MAX_LFMC_VALUE = 327  # 99th percentile of the LFMC values


class Column(StrEnum):
    SORTING_ID = "sorting_id"
    CONTACT = "contact"
    SITE_NAME = "site_name"
    COUNTRY = "country"
    STATE_REGION = "state_region"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SAMPLING_DATE = "sampling_date"
    PROTOCOL = "protocol"
    LFMC_VALUE = "lfmc_value"
    SPECIES_COLLECTED = "species_collected"
    SPECIES_FUNCTIONAL_TYPE = "species_functional_type"
    LANDCOVER = "landcover"
    ELEVATION = "elevation"


class FileSuffix(StrEnum):
    TIF = "tif"
    H5 = "h5"


WGS84_EPSG = 4326


WORLD_COVER_CLASS_MAP = frozendict(
    {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }
)
