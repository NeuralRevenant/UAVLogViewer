"""
UAV telemetry parser
----------------------------------------

 - MAVLink streams / DataFlash :  .tlog   .bin   .log
 - PX4 ULog                    :  .ulg    .ulog   (needs *pyulog*)

Output
------
dict{ message_type : pandas.DataFrame }   — each DF is indexed by a *unique*
`timestamp` in UTC and all numeric columns are in coherent SI units.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from pymavlink import mavutil
import pandas as pd
import numpy  as np
import os, time, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1.  Messages we actually need                                               #
# --------------------------------------------------------------------------- #
ESSENTIAL: Dict[str, List[str]] = {
    # Navigation
    "GLOBAL_POSITION_INT": ["time_boot_ms", "lat", "lon", "alt",
                            "relative_alt", "vx", "vy", "vz", "hdg"],
    "VFR_HUD":             ["airspeed", "groundspeed", "heading",
                            "throttle", "alt", "climb"],
    "ATTITUDE":            ["time_boot_ms", "roll", "pitch", "yaw",
                            "rollspeed", "pitchspeed", "yawspeed"],

    # Power
    "SYS_STATUS":          ["voltage_battery", "current_battery",
                            "battery_remaining"],
    "BATTERY_STATUS":      ["voltages", "current_battery",
                            "battery_remaining"],

    # GPS
    "GPS_RAW_INT":         ["time_usec", "fix_type", "lat", "lon",
                            "alt", "eph", "epv", "satellites_visible"],

    # RC / RSSI
    "RC_CHANNELS":         ["time_boot_ms", "chan1_raw", "chan2_raw",
                            "chan3_raw", "chan4_raw", "chan5_raw",
                            "chan6_raw", "chan7_raw", "chan8_raw", "rssi"],
}

# --------------------------------------------------------------------------- #
# 2.  Raw-unit  →  SI conversion factors                                      #
# --------------------------------------------------------------------------- #
UNIT: Dict[str, float] = {
    # lat / lon  (1e7 → deg)
    "GLOBAL_POSITION_INT.lat": 1e-7,
    "GLOBAL_POSITION_INT.lon": 1e-7,
    "GPS_RAW_INT.lat":         1e-7,
    "GPS_RAW_INT.lon":         1e-7,

    # altitudes  (mm → m)
    "GLOBAL_POSITION_INT.alt": 1e-3,
    "GPS_RAW_INT.alt":         1e-3,

    # battery voltage (mV → V)  – other cells added at run-time
    "SYS_STATUS.voltage_battery": 1e-3,

    # current (0.01 A units → A)
    "SYS_STATUS.current_battery":     0.01,
    "BATTERY_STATUS.current_battery": 0.01,
}

MAX_VOLT_CELLS      = 10       # ArduCopter sends at most ten cell voltages
SHIFT_DUPLICATES_US = 1        # shift duplicate timestamps by ±1 µs steps


class TelemetryParser:
    # ------------------------------------------------------------------- #
    def __init__(self, file_path: str) -> None:
        self.fp  = file_path
        self.ext = os.path.splitext(file_path)[1].lower()
        if self.ext not in (".tlog", ".bin", ".log", ".ulg", ".ulog"):
            raise ValueError("Supported: .tlog  .bin  .log  .ulg  .ulog")

        self.is_ulog = self.ext in (".ulg", ".ulog")
        if self.is_ulog:
            try:
                import pyulog    # noqa: F401
            except ImportError as e:
                raise ImportError("*.ulg parsing needs the *pyulog* package") from e

    # ------------------------------------------------------------------- #
    # Public entry-point                                                  #
    # ------------------------------------------------------------------- #
    def parse(self) -> Dict[str, pd.DataFrame]:
        t0   = time.time()
        raw  = self._read_ulog() if self.is_ulog else self._read_mavlink()
        dfs  = self._to_dataframes(raw)
        rows = sum(len(df) for df in dfs.values())
        log.info("✓ parsed %d msg-types – %d rows – %.2fs",
                 len(dfs), rows, time.time() - t0)
        return dfs

    # ------------------------------------------------------------------- #
    # 3.  MAVLink / DataFlash reader (.tlog / .bin / .log)                #
    # ------------------------------------------------------------------- #
    def _read_mavlink(self) -> Dict[str, List[Dict[str, Any]]]:
        mlog = mavutil.mavlink_connection(self.fp, dialect="common")

        data: Dict[str, List[Dict[str, Any]]] = {}
        utc_anchor: Optional[float] = None   # first trusted wall-clock
        boot_off : Optional[float] = None   # continuously refined wall − boot

        wanted = list(ESSENTIAL.keys())

        while True:
            msg = mlog.recv_match(type=wanted, blocking=False)
            if msg is None:
                break

            kind = msg.get_type()
            d    = msg.to_dict()

            # ---- raw clocks -------------------------------------------
            wall = getattr(msg, "_timestamp", None)
            wall = float(wall) if wall and wall > 1e9 else None  # sanity guard

            boot = d.get("time_boot_ms")
            if boot is not None:
                boot /= 1e3
            elif "time_usec" in d:
                boot = d["time_usec"] / 1e6
            else:
                boot = None

            if wall and not utc_anchor:
                utc_anchor = wall
            if wall and boot is not None:
                boot_off = wall - boot        # refine whenever possible

            ts = boot + boot_off if (boot is not None and boot_off is not None) else \
                 wall or utc_anchor or 0.0

            d["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
            data.setdefault(kind, []).append(d)

        mlog.close()
        return data

    # ------------------------------------------------------------------- #
    # 4.  PX4 ULog reader (.ulg / .ulog)                                  #
    # ------------------------------------------------------------------- #
    def _read_ulog(self) -> Dict[str, List[Dict[str, Any]]]:
        from pyulog import ULog
        u = ULog(self.fp)

        data: Dict[str, List[Dict[str, Any]]] = {}
        topics = {
            "vehicle_local_position":  "GLOBAL_POSITION_INT",
            "vehicle_global_position": "GLOBAL_POSITION_INT",
            "vehicle_attitude":        "ATTITUDE",
            "battery_status":          "BATTERY_STATUS",
            "vehicle_gps_position":    "GPS_RAW_INT",
            "actuator_outputs":        "RC_CHANNELS",
            "vehicle_air_data":        "VFR_HUD",
        }

        for d in u.data_list:
            mtype = topics.get(d.name)
            if mtype not in ESSENTIAL:
                continue

            boot_usec = d.data["timestamp"]
            if not len(boot_usec):
                continue

            boot0  = boot_usec[0] / 1e6
            wall0  = u.start_timestamp / 1e6
            offset = wall0 - boot0          # wall − boot  (same logic as MAVLink)

            rows = []
            for i in range(len(boot_usec)):
                row = {f: d.data[f][i] for f in d.field_names if f != "timestamp"}
                ts  = (boot_usec[i] / 1e6) + offset
                row["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
                rows.append(row)

            data.setdefault(mtype, []).extend(rows)

        return data

    # ------------------------------------------------------------------- #
    # 5.  Raw dicts → tidy DataFrames                                     #
    # ------------------------------------------------------------------- #
    def _to_dataframes(self, raw: Dict[str, List[Dict[str, Any]]]
                       ) -> Dict[str, pd.DataFrame]:

        dfs: Dict[str, pd.DataFrame] = {}
        for mtype, rows in raw.items():
            if not rows:
                continue

            df = pd.DataFrame(rows).set_index("timestamp", drop=True).sort_index()

            # -- duplicate timestamps → shift by ±1 μs so NOTHING is lost --
            if not df.index.is_unique:
                dup = df.groupby(level=0).cumcount()
                df.index = df.index + pd.to_timedelta(dup * SHIFT_DUPLICATES_US, "us")

            # -- expand BATTERY_STATUS.voltages[N] ------------------------
            for col in list(df.columns):
                if isinstance(df[col].iloc[0], list):
                    max_len = min(df[col].map(len).max(), MAX_VOLT_CELLS)
                    for i in range(max_len):
                        new = f"{col}[{i}]"
                        df[new] = df[col].apply(
                            lambda lst: np.nan
                            if not lst or lst[i] in (0, 65535)
                            else lst[i]
                        )
                        UNIT.setdefault(new.replace(":", "."), 1e-3)  # add SI factor
                    df.drop(columns=col, inplace=True)

            # -- gentle gap-fill (numeric only, max 50 samples) -----------
            if mtype not in ("GLOBAL_POSITION_INT", "GPS_RAW_INT"):
                df = df.ffill(limit=50)

            # -- apply SI conversions ------------------------------------
            for col in df.columns:
                factor = UNIT.get(f"{mtype}.{col}")
                if factor:
                    df[col] = df[col] * factor

            # basic GPS sanity guard
            if mtype in ("GLOBAL_POSITION_INT", "GPS_RAW_INT"):
                df.loc[(df["lat"].abs() > 90) | (df["lon"].abs() > 180), :] = np.nan
                if "alt" in df.columns:
                    df.loc[df["alt"].abs() > 1e5, "alt"] = np.nan

            # co-locate GPS_RAW_INT with GLOBAL_POSITION_INT ≫
            if mtype in ("GLOBAL_POSITION_INT", "GPS_RAW_INT"):
                # resample onto a 10 ms grid so the two GPS streams line up
                df = (
                    df.resample("10ms")
                    .mean(numeric_only=True) # only average numeric columns
                    .ffill(limit=20)    # allow up to 200 ms of forward‐fill
                )

            dfs[mtype] = df.astype("float32", errors="ignore")

        return dfs


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
# parser = TelemetryParser("flight_log.tlog") # .tlog / .bin / .log / .ulg / .ulog
# telemetry_data = parser.parse()
# analyzer = TelemetryAnalyzer(telemetry_data)