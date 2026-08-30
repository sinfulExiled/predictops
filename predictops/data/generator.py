"""Deterministic synthetic plant: telemetry, degradation, failures, labels.

Models a mid-sized water / wastewater treatment works -- pumps, aeration
blowers, drive motors and sludge conveyors -- because that fleet mix makes the
failure economics concrete. The physics is equipment-level (bearing wear,
cavitation, overheating, pressure loss, electrical faults), so nothing here is
specific to water; the same generator would serve any rotating-equipment site.

Design notes
------------
The generator is built so that a *threshold* rule cannot win.  Every symptom a
failure produces is also produced, transiently and innocently, by something
else: load surges raise vibration/current/temperature, heatwaves raise
temperature, sensor glitches produce isolated extremes.  What separates a real
degradation from a nuisance is the *joint temporal pattern* (e.g. vibration and
current rising while load stays flat), which is exactly what a sequence model
can learn and a static threshold cannot.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..config import DATA_DIR, MACHINE_MIX, GeneratorConfig, HORIZON_HOURS
from .schemas import (
    ARCHETYPES,
    FAILURE_MODES,
    FailureEvent,
    MachineArchetype,
    MaintenanceEvent,
)

N_SITES = 4

# Per-channel glitch magnitude (multiplicative, sign preserving).
SPIKE_MAGNITUDE = {
    "temperature": (1.15, 1.45),
    "vibration": (1.8, 3.2),
    "pressure": (1.4, 2.4),
    "current": (1.5, 2.6),
    "voltage": (1.02, 1.08),
    "rpm": (1.05, 1.20),
}

# Transducer output ranges.
SENSOR_RANGE = {
    "temperature": (-30.0, 200.0),
    "vibration": (0.01, 60.0),
    "pressure": (0.0, 25.0),
    "rpm": (0.0, 6000.0),
    "current": (0.0, 300.0),
    "voltage": (300.0, 480.0),
    "load": (0.0, 1.3),
    "humidity": (0.0, 100.0),
}


# --------------------------------------------------------------------------
# small signal helpers
# --------------------------------------------------------------------------
def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """First-order lag -> thermal inertia."""
    zi = [x[0] * (1.0 - alpha)]
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)[0]


def _smooth_noise(rng: np.random.Generator, n: int, scale: float,
                  span: int = 12) -> np.ndarray:
    """Low-frequency wander (a slow drift, not white noise)."""
    span = max(int(span), 1)
    raw = rng.normal(0.0, 1.0, n + span)
    kernel = np.ones(span) / span
    out = np.convolve(raw, kernel, mode="valid")[:n]
    return out * (scale / (out.std() + 1e-9))


def _piecewise_setpoint(rng: np.random.Generator, n: int, steps_per_hour: int,
                        mean: float, sd: float, lo: float, hi: float,
                        min_h: float = 2.0, max_h: float = 8.0) -> np.ndarray:
    """Production setpoints hold for hours, then step to a new value."""
    out = np.empty(n)
    i = 0
    while i < n:
        hold = max(int(rng.uniform(min_h, max_h) * steps_per_hour), 1)
        out[i:i + hold] = float(np.clip(rng.normal(mean, sd), lo, hi))
        i += hold
    return out


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
@dataclass
class GeneratedDataset:
    telemetry: pd.DataFrame
    machines: pd.DataFrame
    failures: pd.DataFrame
    maintenance: pd.DataFrame
    manifest: dict


class PlantGenerator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.steps_per_hour = 60 // cfg.resolution_minutes
        self.n_steps = int(cfg.days * 24 * self.steps_per_hour)
        self.index = pd.date_range(
            start=cfg.start, periods=self.n_steps,
            freq=f"{cfg.resolution_minutes}min",
        )
        self.hours = np.arange(self.n_steps) / self.steps_per_hour

    def _step(self, hours: float) -> int:
        """Snap a float hour offset onto the sampling grid."""
        return int(np.clip(round(hours * self.steps_per_hour), 0, self.n_steps))

    def _ts(self, step: int):
        """Timestamp for a grid step (clamped to the last sample)."""
        return self.index[int(np.clip(step, 0, self.n_steps - 1))]

    # -- site-level environment -------------------------------------------
    def _sites(self, rng: np.random.Generator) -> dict:
        sites: dict = {}
        n_heat_days = int(round(self.cfg.days * self.cfg.heatwave_days_fraction))
        for s in range(N_SITES):
            mean_t = rng.uniform(14.0, 24.0)
            diurnal = 6.5 * np.sin(2 * np.pi * (self.hours - 15.0) / 24.0)
            drift = _smooth_noise(rng, self.n_steps, 2.2, span=6 * self.steps_per_hour)
            ambient = mean_t + diurnal + drift

            heat = np.zeros(self.n_steps)
            if n_heat_days > 0:
                start_days = rng.choice(max(self.cfg.days - 2, 1),
                                        size=n_heat_days, replace=False)
                for d in start_days:
                    a = int(d * 24 * self.steps_per_hour)
                    b = min(a + int(rng.uniform(18, 34) * self.steps_per_hour),
                            self.n_steps)
                    if b <= a:
                        continue
                    ramp = np.sin(np.linspace(0, np.pi, b - a))
                    heat[a:b] += ramp * rng.uniform(9.0, 16.0)
            ambient = ambient + heat

            humidity = np.clip(
                58 + 12 * np.sin(2 * np.pi * (self.hours - 3.0) / 24.0)
                + _smooth_noise(rng, self.n_steps, 5.0,
                                span=4 * self.steps_per_hour),
                20, 95,
            )
            sites[s] = {"ambient": ambient, "humidity": humidity,
                        "heatwave": (heat > 4.0).astype(np.int8)}
        return sites

    # -- per-machine load --------------------------------------------------
    def _load_profile(self, rng: np.random.Generator):
        base = _piecewise_setpoint(
            rng, self.n_steps, self.steps_per_hour,
            mean=rng.uniform(0.62, 0.78), sd=0.09, lo=0.35, hi=0.95,
        )
        shift = 0.06 * np.sin(2 * np.pi * (self.hours - 6.0) / 24.0)
        load = base + shift + _smooth_noise(rng, self.n_steps, 0.025, span=6)

        surge = np.zeros(self.n_steps, dtype=np.int8)
        n_surges = rng.poisson(self.cfg.load_surge_per_machine_day * self.cfg.days)
        for _ in range(int(n_surges)):
            a = int(rng.integers(0, self.n_steps))
            b = min(a + int(rng.uniform(1.0, 5.0) * self.steps_per_hour), self.n_steps)
            if b <= a:
                continue
            ramp = np.sin(np.linspace(0, np.pi, b - a)) ** 0.5
            load[a:b] += ramp * rng.uniform(0.18, 0.33)
            surge[a:b] = 1
        return np.clip(load, 0.25, 1.15), surge

    # -- failure scheduling -------------------------------------------------
    def _schedule_events(self, rng: np.random.Generator, machine_id: str,
                         arch: MachineArchetype):
        n_events = int(rng.poisson(self.cfg.failures_per_machine))
        events: list[FailureEvent] = []
        maint: list[MaintenanceEvent] = []
        total_h = self.cfg.days * 24.0

        # Failures are spread uniformly over the whole observation window, not
        # accumulated from t=0 -- otherwise every event lands in the training
        # period and the held-out splits contain no positives to score.
        candidates = sorted(float(h) for h in
                            rng.uniform(6.0, total_h - 2.0, n_events))
        prev_repair_h = -1e9

        for fail_h in candidates:
            ftype = str(rng.choice(arch.failure_modes))
            mode = FAILURE_MODES[ftype]
            sudden = bool(rng.random() < self.cfg.sudden_failure_fraction)
            if sudden:
                deg_h = float(rng.uniform(0.8, 2.5))
            else:
                deg_h = float(rng.uniform(*mode.typical_hours))
            start_h = fail_h - deg_h
            # keep events apart: degradation may not start before the previous
            # repair has finished
            if start_h < prev_repair_h + 4.0 or start_h < 1.0:
                continue
            repair_h = fail_h + float(rng.uniform(1.5, 8.0))
            prev_repair_h = repair_h

            subtle = bool(rng.random() < self.cfg.subtle_failure_fraction)
            if subtle:
                scale = float(rng.uniform(0.38, 0.62))
            else:
                scale = float(rng.uniform(0.85, 1.15))
            atypical = bool(rng.random() < self.cfg.atypical_failure_fraction)

            d_step = self._step(start_h)
            f_step = self._step(fail_h)
            r_step = self._step(repair_h)
            if f_step <= d_step:
                continue
            events.append(FailureEvent(
                machine_id=machine_id, failure_type=ftype,
                degradation_start=self._ts(d_step),
                failure_time=self._ts(f_step),
                repair_time=self._ts(r_step),
                severity_scale=scale, degradation_hours=deg_h,
                is_sudden=sudden, is_atypical=atypical,
                suppressed_channel=mode.signature if atypical else "",
                deg_start_step=d_step, fail_step=f_step, repair_step=r_step,
            ))
            maint.append(MaintenanceEvent(
                machine_id=machine_id, timestamp=self._ts(r_step),
                kind="corrective", failure_type=ftype, step=r_step,
            ))

        for _ in range(int(rng.poisson(0.7))):
            t_h = float(rng.uniform(0, total_h))
            k = self._step(t_h)
            maint.append(MaintenanceEvent(
                machine_id=machine_id, timestamp=self._ts(k),
                kind="preventive", step=k,
            ))
        maint.sort(key=lambda m: m.step)
        return events, maint

    # -- latent severity track ---------------------------------------------
    def _severity(self, rng: np.random.Generator, events: list):
        sev = np.zeros(self.n_steps)
        active = np.zeros(self.n_steps, dtype=np.int8)
        for ev in events:
            a, b = ev.deg_start_step, ev.fail_step
            if b <= a:
                continue
            gamma = float(rng.uniform(1.4, 3.0))
            u = np.linspace(0.0, 1.0, b - a)
            curve = u ** gamma
            wobble = _smooth_noise(rng, b - a, 0.035,
                                   span=max(3, self.steps_per_hour))
            sev[a:b] = np.clip(curve + wobble, 0.0, 1.05) * ev.severity_scale
            active[a:b] = 1
        return sev, active

    # -- one machine --------------------------------------------------------
    def _machine_frame(self, rng, machine_id, arch, site, events, maint):
        n = self.n_steps
        load, surge = self._load_profile(rng)
        sev, deg_active = self._severity(rng, events)

        vib_c = np.zeros(n)
        temp_c = np.zeros(n)
        cur_c = np.zeros(n)
        prs_c = np.zeros(n)
        vlt_c = np.zeros(n)
        jit_c = np.zeros(n)
        ftype_track = np.full(n, "", dtype=object)
        for ev in events:
            a, b = ev.deg_start_step, ev.fail_step
            if b <= a:
                continue
            m = FAILURE_MODES[ev.failure_type]
            damp = {c: 1.0 for c in ("vibration", "temperature", "current",
                                     "pressure", "voltage")}
            if ev.is_atypical and ev.suppressed_channel in damp:
                damp[ev.suppressed_channel] = 0.12
            vib_c[a:b] = m.vib_mult * damp["vibration"]
            temp_c[a:b] = m.temp_add * damp["temperature"]
            cur_c[a:b] = m.current_mult * damp["current"]
            prs_c[a:b] = m.pressure_mult * damp["pressure"]
            vlt_c[a:b] = m.voltage_mult * damp["voltage"]
            jit_c[a:b] = m.rpm_jitter_mult
            ftype_track[a:b] = ev.failure_type

        ambient = site["ambient"]
        humidity = site["humidity"]

        vib_offset = rng.uniform(0.85, 1.2)
        vibration = (arch.vib_base * vib_offset * (0.72 + 0.46 * load)
                     * (1.0 + vib_c * sev)
                     + _smooth_noise(rng, n, 0.06, span=4)
                     + rng.normal(0, 0.05, n))

        temp_target = (ambient + arch.temp_base * (0.34 + 0.66 * load)
                       + temp_c * sev)
        temperature = _ema(temp_target, arch.thermal_alpha) + rng.normal(0, 0.35, n)

        current = (arch.current_base * (0.28 + 0.74 * load)
                   * (1.0 + cur_c * sev)
                   + _smooth_noise(rng, n, 0.20, span=4)
                   + rng.normal(0, 0.18, n))

        voltage = (arch.voltage_nominal * (1.0 + vlt_c * sev)
                   + _smooth_noise(rng, n, 1.6, span=8)
                   + rng.normal(0, 0.9, n))

        jitter = rng.normal(0, 1.0, n) * (2.5 + 6.0 * jit_c * sev)
        rpm = arch.rpm_nominal * (1.0 - 0.022 * load) + jitter

        if arch.pressure_base > 0:
            pressure = (arch.pressure_base * (0.55 + 0.5 * load)
                        * (1.0 + prs_c * sev)
                        + _smooth_noise(rng, n, 0.05, span=5)
                        + rng.normal(0, 0.03, n))
        else:
            pressure = np.zeros(n)

        power = np.sqrt(3.0) * voltage * current * arch.power_factor / 1000.0

        op_hours = np.arange(n) / self.steps_per_hour + rng.uniform(0, 400)
        for mv in maint:
            if mv.step < n:
                op_hours[mv.step:] -= op_hours[mv.step]

        df = pd.DataFrame({
            "machine_id": machine_id,
            "timestamp": self.index,
            "temperature": temperature,
            "vibration": np.maximum(vibration, 0.02),
            "pressure": np.maximum(pressure, 0.0),
            "rpm": rpm,
            "current": np.maximum(current, 0.0),
            "voltage": voltage,
            "power": power,
            "load": load,
            "humidity": humidity,
            "ambient_temp": ambient,
            "operating_hours": np.maximum(op_hours, 0.0),
            "severity": sev,
            "degradation_active": deg_active,
            "load_surge": surge,
            "heatwave": site["heatwave"],
            "latent_failure_type": ftype_track,
        })

        # --- downtime -------------------------------------------------------
        is_down = np.zeros(n, dtype=np.int8)
        for ev in events:
            is_down[ev.fail_step:ev.repair_step] = 1
        sensor_cols = ["temperature", "vibration", "pressure", "rpm",
                       "current", "voltage", "power", "load"]
        down = is_down.astype(bool)
        if down.any():
            for c in sensor_cols:
                if c == "temperature":
                    df.loc[down, c] = df.loc[down, "ambient_temp"] + 4.0
                else:
                    df.loc[down, c] = 0.0
        df["is_downtime"] = is_down

        # --- nuisances: glitches then dropouts ------------------------------
        for c, (lo_mag, hi_mag) in SPIKE_MAGNITUDE.items():
            hit = rng.random(n) < self.cfg.spike_rate
            if hit.any():
                k = int(hit.sum())
                mag = rng.uniform(lo_mag, hi_mag, k)
                # a stuck-high or stuck-low reading, never a sign flip
                stuck_low = rng.random(k) < 0.5
                factor = np.where(stuck_low, 1.0 / mag, mag)
                df.loc[hit, c] = df.loc[hit, c].to_numpy() * factor

        drop = np.zeros(n, dtype=bool)
        n_gaps = rng.poisson(self.cfg.missing_rate * n)
        for _ in range(int(n_gaps)):
            a = int(rng.integers(0, n))
            b = min(a + int(rng.uniform(1, 2.0 * self.steps_per_hour)), n)
            drop[a:b] = True
        drop &= ~down
        if drop.any():
            for c in sensor_cols + ["humidity"]:
                df.loc[drop, c] = np.nan
        df["sensor_dropout"] = drop.astype(np.int8)

        # transducers have physical ranges; clip after all nuisances
        for c, (lo, hi) in SENSOR_RANGE.items():
            df[c] = df[c].clip(lo, hi)
        return df

    # -- labels -------------------------------------------------------------
    def _labels(self, df: pd.DataFrame, events: list) -> pd.DataFrame:
        n = len(df)
        label = np.zeros(n, dtype=np.int8)
        ttf = np.full(n, np.inf)
        htype = np.full(n, "", dtype=object)
        steps = np.arange(n)
        horizon_steps = HORIZON_HOURS * self.steps_per_hour
        for ev in events:
            f = ev.fail_step
            lo = max(f - horizon_steps, 0)
            label[lo:f] = 1
            htype[lo:f] = ev.failure_type
            gap = (f - steps[:f]) / self.steps_per_hour
            ttf[:f] = np.fmin(ttf[:f], gap)
        ttf[np.isinf(ttf)] = np.nan
        df["label"] = label
        df["time_to_failure_h"] = ttf
        df["horizon_failure_type"] = htype
        df.loc[df["is_downtime"] == 1, "label"] = 0
        return df

    # -- driver --------------------------------------------------------------
    def generate(self) -> GeneratedDataset:
        master = np.random.default_rng(self.cfg.seed)
        site_rng = np.random.default_rng(self.cfg.seed + 1)
        sites = self._sites(site_rng)

        # Lay the fleet out to the configured mix, largest class first, so the
        # counts are exact rather than approximate.
        kinds: list[str] = []
        for kind, share in sorted(MACHINE_MIX.items(), key=lambda kv: -kv[1]):
            kinds += [kind] * int(round(share * self.cfg.n_machines))
        while len(kinds) < self.cfg.n_machines:
            kinds.append(max(MACHINE_MIX, key=MACHINE_MIX.get))
        kinds = kinds[:self.cfg.n_machines]

        frames, machine_rows = [], []
        all_failures: list = []
        all_maint: list = []

        for i in range(self.cfg.n_machines):
            rng = np.random.default_rng(self.cfg.seed * 1000 + i)
            kind = kinds[i]
            arch = ARCHETYPES[kind]
            machine_id = f"{kind}-{i:03d}"
            site = int(master.integers(0, N_SITES))
            events, maint = self._schedule_events(rng, machine_id, arch)
            df = self._machine_frame(rng, machine_id, arch, sites[site],
                                     events, maint)
            df = self._labels(df, events)
            df["machine_type"] = kind
            df["site"] = site
            frames.append(df)
            machine_rows.append({
                "machine_id": machine_id, "machine_type": kind, "site": site,
                "rpm_nominal": arch.rpm_nominal,
                "install_age_h": float(df["operating_hours"].iloc[0]),
                "n_failures": len(events),
            })
            all_failures.extend(events)
            all_maint.extend(maint)

        telemetry = pd.concat(frames, ignore_index=True)
        telemetry = telemetry.sort_values(
            ["timestamp", "machine_id"]).reset_index(drop=True)

        machines = pd.DataFrame(machine_rows)
        failures = pd.DataFrame([e.to_dict() for e in all_failures])
        maintenance = pd.DataFrame([m.to_dict() for m in all_maint])

        manifest = {
            "config": self.cfg.to_dict(),
            "n_rows": int(len(telemetry)),
            "n_machines": int(self.cfg.n_machines),
            "n_failures": int(len(failures)),
            "horizon_hours": HORIZON_HOURS,
            "positive_rate": float(telemetry["label"].mean()),
            "downtime_rate": float(telemetry["is_downtime"].mean()),
            "missing_rate": float(telemetry["vibration"].isna().mean()),
            "start": str(self.index[0]),
            "end": str(self.index[-1]),
            "failure_type_counts": (
                failures["failure_type"].value_counts().to_dict()
                if len(failures) else {}),
        }
        return GeneratedDataset(telemetry, machines, failures, maintenance,
                                manifest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_dataset(ds: GeneratedDataset, out_dir: Path = DATA_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": out_dir / "telemetry.parquet",
        "machines": out_dir / "machines.parquet",
        "failures": out_dir / "failures.parquet",
        "maintenance": out_dir / "maintenance.parquet",
    }
    ds.telemetry.to_parquet(paths["telemetry"], index=False)
    ds.machines.to_parquet(paths["machines"], index=False)
    if len(ds.failures):
        ds.failures.to_parquet(paths["failures"], index=False)
    if len(ds.maintenance):
        ds.maintenance.to_parquet(paths["maintenance"], index=False)

    manifest = dict(ds.manifest)
    manifest["checksums"] = {k: _sha256(p) for k, p in paths.items()
                             if p.exists()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_dataset(out_dir: Path = DATA_DIR) -> GeneratedDataset:
    tel = pd.read_parquet(out_dir / "telemetry.parquet")
    mac = pd.read_parquet(out_dir / "machines.parquet")
    fal = pd.read_parquet(out_dir / "failures.parquet")
    mnt = pd.read_parquet(out_dir / "maintenance.parquet")
    man = json.loads((out_dir / "manifest.json").read_text())
    return GeneratedDataset(tel, mac, fal, mnt, man)
