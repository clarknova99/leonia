"""Central configuration for the Leonia traffic framework.

All paths are relative to the repository root. Geometries use WGS84
(EPSG:4326) longitude/latitude unless otherwise noted.

The data directory can be overridden via the ``LEONIA_DATA_DIR``
environment variable, which is required for Docker deploys where
``data/`` lives outside the source tree (e.g. mounted as a volume
or baked into a different image layer than ``leonia_traffic/``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Polygon, box

REPO_ROOT = Path(__file__).resolve().parents[1]

STREETLIGHT_DIR = REPO_ROOT / "streetlight"
DATA_DIR = Path(os.environ.get("LEONIA_DATA_DIR", REPO_ROOT / "data"))
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_NETWORK_DIR = DATA_DIR / "network"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_FIG_DIR = REPORTS_DIR / "figures"

for _p in (DATA_PROCESSED_DIR, DATA_NETWORK_DIR, REPORTS_FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# Leonia bounding box with a ~1-mile buffer, chosen to include:
#   - the GWB approach in Fort Lee (east)
#   - Englewood Cliffs and the Palisade ridge (east)
#   - Palisades Park and Ridgefield (south)
#   - Overpeck Park and Ridgefield Park (west)
#   - Englewood (north)
LEONIA_BBOX_WGS84 = (-74.030, 40.852, -73.960, 40.895)  # (minx, miny, maxx, maxy)

LEONIA_BBOX_POLYGON = box(*LEONIA_BBOX_WGS84)


# Actual Borough of Leonia administrative boundary (from OSM via
# `osmnx.geocode_to_gdf("Leonia, Bergen County, New Jersey, USA")`).
# This is the jurisdictional polygon used to filter recommendations to
# streets the borough can actually act on. The broader ``LEONIA_BBOX``
# above remains in use for data-collection scope.
LEONIA_BOROUGH_BOUNDARY = DATA_NETWORK_DIR / "leonia_borough.geojson"


@lru_cache(maxsize=1)
def load_leonia_polygon():
    """Return the Borough of Leonia boundary as a shapely polygon (WGS84).

    Cached after first load. Raises ``FileNotFoundError`` if the
    ``data/network/leonia_borough.geojson`` file does not exist; the file
    is checked into the repo, so this should only fire if the data dir
    has been wiped.
    """
    import geopandas as gpd

    if not LEONIA_BOROUGH_BOUNDARY.exists():
        raise FileNotFoundError(
            f"Leonia borough boundary not found at {LEONIA_BOROUGH_BOUNDARY}. "
            "Regenerate via: `osmnx.geocode_to_gdf('Leonia, Bergen County, "
            "New Jersey, USA').to_file(...)`."
        )
    gdf = gpd.read_file(LEONIA_BOROUGH_BOUNDARY).to_crs(4326)
    return gdf.geometry.union_all()


# Road classes / OSM way name fragments that are state- or federally-
# owned even where they cross Leonia. The borough has no authority over
# these and recommendations should not target them.
NON_BOROUGH_ROAD_NAMES = (
    "New Jersey Turnpike",
    "NJ Turnpike",
    "Garden State Parkway",
    "I 95",
    "I-95",
    "US 1",
    "US 9",
    "US 46",
    "George Washington Bridge",
    "GWB",
    "NJ 4",
    "Route 4",
    "Mackay Highway",  # NJ 4 spur
)

# OSM ``highway`` classes that represent state/federal facilities — even
# inside Leonia these are not under municipal jurisdiction.
NON_BOROUGH_ROAD_CLASSES = (
    "Motorway",
    "On/Off Ramp",
)


# County- and state-owned arterials that physically pass through Leonia
# but are governed by Bergen County / NJDOT. The borough **cannot**
# divert, modify, or restrict access to these roads — but they ARE the
# preferred channels for through-traffic. Recommendations targeting
# these names should be downgraded to "monitor / petition county"; new
# rules should explicitly encourage routing traffic *toward* them.
#
# Source: NJDOT Straight-Line Diagrams + Bergen County roadway inventory:
#   * Broad Avenue   — CR 1 (Bergen County)
#   * Grand Avenue   — CR 17 / CR 49 (Bergen County)
#   * Fort Lee Road  — CR 9 (Bergen County). Within Leonia the road is
#     signed locally as "Main Street" along part of its length — the
#     two names refer to the same county-owned corridor.
COUNTY_STATE_ARTERIALS = (
    "Broad Avenue",
    "Broad Ave",
    "Grand Avenue",
    "Grand Ave",
    "Fort Lee Road",
    "Fort Lee Rd",
    "Main Street",
    "Main St",
)


# GWB upper-deck approach: rough polygon covering Bruce Reynolds Blvd,
# Center Ave, and the toll plaza in Fort Lee. Refined later from OSM.
GWB_APPROACH_POLYGON = Polygon([
    (-73.974, 40.852),
    (-73.974, 40.860),
    (-73.962, 40.860),
    (-73.962, 40.852),
])


# Names of streets we suspect carry cut-through traffic. Used to
# highlight rows in exploratory reports and to seed scenario design.
# NB: county arterials (Broad Ave, Grand Ave, Fort Lee Rd) are
# intentionally omitted — they are the desired channel for through-
# traffic, not targets for local intervention.
SUSPECTED_CUTTHROUGH_STREETS = (
    "Hillside Avenue",
    "Hillside Ave",
    "Christie Heights Street",
    "Christie Heights St",
    "Glenwood Avenue",
    "Glenwood Ave",
    "Beechwood Place",
    "Beechwood Pl",
    "Lakeview Avenue",
    "Lakeview Ave",
    "Park Avenue",
    "Park Ave",
)


# Cities/places to keep when filtering the StreetLight data. Anything
# outside this set is dropped to keep maps focused on Leonia + its
# immediate cut-through approaches.
STUDY_AREA_CITIES = (
    "Leonia, Bergen, New Jersey",
    "Fort Lee, Bergen, New Jersey",
    "Englewood, Bergen, New Jersey",
    "Englewood Cliffs, Bergen, New Jersey",
    "Palisades Park, Bergen, New Jersey",
    "Ridgefield Park, Bergen, New Jersey",
    "Bogota, Bergen, New Jersey",
    "Edgewater, Bergen, New Jersey",
    "Unincorporated, Bergen, New Jersey",
)


@dataclass(frozen=True)
class StreetLightSourceLabels:
    """Canonical labels for the StreetLight export sources on disk."""

    all_days: str = "all_days"
    weekdays: str = "weekdays"
    weekend: str = "weekend"


STREETLIGHT_LABELS = StreetLightSourceLabels()


# Mapping from on-disk subfolder name to canonical source label. The
# loader uses this for any folder it recognizes; unrecognized folders
# fall back to using the folder name as the label.
STREETLIGHT_FOLDER_TO_LABEL = {
    "": STREETLIGHT_LABELS.all_days,           # root streetlight/
    "weekdays": STREETLIGHT_LABELS.weekdays,
    "weekend": STREETLIGHT_LABELS.weekend,
}


@dataclass(frozen=True)
class SimulationDefaults:
    """Default UXsim simulation parameters."""

    deltan: int = 5              # platoon size
    tmax_seconds: int = 4 * 3600 # 4-hour simulation horizon by default
    default_jam_density: float = 0.2
    coef_degree_to_meter: float = 111_000.0
    random_seed: int = 0


SIM_DEFAULTS = SimulationDefaults()
