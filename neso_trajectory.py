"""
neso_trajectory.py
=======================
Fetches a moon's X,Y,Z position vectors
relative to Neptune's centre (id=799) from JPL Horizons via astroquery,
then renders an animated GIF of the 3-D trajectory timelapse.

Nereid's and Triton's orbits are drawn per-frame from JPL Horizons vectors
(one orbital period per epoch, cached to nereid_orbits.csv / triton_orbits.csv).
This shows the true long-term secular evolution of both orbits over time.

Usage
-----
    python neso_trajectory.py                            # fetch + animate
    python neso_trajectory.py --no-fetch                 # reuse cached CSV, re-animate
    python neso_trajectory.py --cache my.csv             # custom cache filename
    python neso_trajectory.py --fps 50 --dpi 40          # tweak output
    python neso_trajectory.py --no-fetch --stride 10      # skip every 10th frame (10× faster)
    python neso_trajectory.py --no-fetch --figsize 5     # smaller figure (faster render)
    python neso_trajectory.py --no-fetch --mp4           # also export MP4 via ffmpeg

Speed tips
----------
    --stride N   Render every Nth frame  (e.g. --stride 10 → ~10× faster, smaller file)
    --fps N      GIF frame delay in ms; GIF viewers cap at ~50 fps in practice
    --figsize N  Square figure side in inches (default 7); smaller = faster + smaller file
    --mp4        Convert finished GIF to MP4 with ffmpeg for true 60 fps playback

Output
------
    neso_cache.csv          — raw Horizons vectors (auto-reused on reruns)
    neso_trajectory.gif     — animated GIF timelapse
    neso_trajectory.mp4     — MP4 (only with --mp4 and ffmpeg installed)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
from PIL import Image
from io import BytesIO


# ─────────────────────────────────────────────
# 1.  Horizons query
# ─────────────────────────────────────────────

def fetch_horizons(cache_csv: str) -> pd.DataFrame:
    """Query JPL Horizons for moon's state vectors; cache result to CSV."""
    from astroquery.jplhorizons import Horizons

    print("Querying JPL Horizons for Neso …")
    obj = Horizons(
        id="Neso",                 # Neso
        location="500@899",       # origin = Neptune body centre
        epochs={
            "start": "1601-Jan-01",
            "stop":  "2399-Jan-01",
            "step":  "20d",       # 50-day timestep
        },
    )
    tbl = obj.vectors()

    df = tbl[["datetime_jd", "datetime_str", "x", "y", "z"]].to_pandas()

    # Extract calendar year for labelling
    df["year"] = df["datetime_str"].str.extract(r"(\d{4})-").astype(int)

    # Convert AU → 10^6 km  (1 AU = 149.597870700 × 10^6 km)
    AU_TO_MKM = 149.597870700
    for col in ("x", "y", "z"):
        df[col] = df[col].astype(float) * AU_TO_MKM

    df.to_csv(cache_csv, index=False)
    print(f"  → {len(df)} epochs fetched and cached to '{cache_csv}'")
    return df


def load_cache(cache_csv: str) -> pd.DataFrame:
    print(f"Loading cached data from '{cache_csv}' …")
    df = pd.read_csv(cache_csv)
    print(f"  → {len(df)} epochs loaded  (year range {df['year'].iloc[0]}–{df['year'].iloc[-1]})")
    return df


# ─────────────────────────────────────────────
# 2.  Per-frame orbit queries — Nereid & Triton
# ─────────────────────────────────────────────
#
# To show long-term orbital evolution we query one full orbital period
# for each moon at every rendered epoch (i.e. every stride-th data point).
# Results are cached to CSV so subsequent runs are instant.
#
# Nereid  (Neptune II)  — period = 360.133 d  → 180 pts at 2d steps
# Triton  (Neptune I)   — period =   5.877 d  →  24 pts at 6h steps
#
# Cache CSV columns: epoch_jd, pt_index, x, y, z  (one row per orbit point)

NEREID_PERIOD_DAYS = 360.133039
TRITON_PERIOD_DAYS = 5.876854

NEREID_STEP   = "2d"          # ~180 points per orbit
TRITON_STEP   = "6h"          # ~24 points per orbit

NEREID_CACHE_DEFAULT = "nereid_orbits.csv"
TRITON_CACHE_DEFAULT = "triton_orbits.csv"

AU_TO_MKM = 149.597870700


def _fetch_moon_orbits(
    moon_id: str,
    period_days: float,
    step: str,
    step_hours: float,
    epoch_jds: list,
    cache_csv: str,
) -> dict:
    """
    For each JD in epoch_jds, query one orbital period of `moon_id`
    relative to Neptune centre (500@899).  Returns a dict:
        { jd_float: (x_arr, y_arr, z_arr) }
    Results are cached to cache_csv.
    """
    from astropy.time import Time
    import erfa

    n_pts = int(round(period_days * 24 / step_hours))

    # ── load cache if present ────────────────────────────────────────
    if os.path.exists(cache_csv):
        print(f"Loading {moon_id} orbit cache '{cache_csv}' …")
        cdf = pd.read_csv(cache_csv)
        orbits = {}
        for jd, grp in cdf.groupby("epoch_jd"):
            orbits[jd] = (grp["x"].values, grp["y"].values, grp["z"].values)
        # If all epochs are present, return immediately
        missing = [jd for jd in epoch_jds if jd not in orbits]
        if not missing:
            print(f"  → {len(orbits)} {moon_id} orbits loaded from cache.")
            return orbits
        print(f"  → {len(missing)} epochs missing — querying Horizons for those …")
    else:
        orbits  = {}
        missing = epoch_jds
        cdf     = pd.DataFrame(columns=["epoch_jd", "pt_index", "x", "y", "z"])

    from astroquery.jplhorizons import Horizons

    new_rows = []
    for k, jd in enumerate(missing):
        # Suppress ERFA "dubious year" warnings — expected for dates outside
        # 1900–2100 and harmless for our purposes (JPL handles the ephemeris).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=erfa.ErfaWarning)
            t_start = Time(jd,                                    format="jd")
            t_stop  = Time(jd + period_days + step_hours / 24.0, format="jd")
            start_str = t_start.iso.replace(" ", "T")[:19]
            stop_str  = t_stop.iso.replace(" ", "T")[:19]

            obj = Horizons(
                id=moon_id,
                location="500@899",
                epochs={"start": start_str, "stop": stop_str, "step": step},
            )
            tbl = obj.vectors()
        df  = tbl[["x", "y", "z"]].to_pandas().iloc[: n_pts + 1].copy()
        for col in ("x", "y", "z"):
            df[col] = df[col].astype(float) * AU_TO_MKM

        # Close loop
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        xs, ys, zs = df["x"].values, df["y"].values, df["z"].values
        orbits[jd] = (xs, ys, zs)

        for idx, (x, y, z) in enumerate(zip(xs, ys, zs)):
            new_rows.append({"epoch_jd": jd, "pt_index": idx, "x": x, "y": y, "z": z})

        if (k + 1) % 10 == 0 or k == len(missing) - 1:
            print(f"  {moon_id}: {k+1}/{len(missing)} epochs queried …",
                  end="\r", flush=True)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cdf    = pd.concat([cdf, new_df], ignore_index=True)
        cdf.to_csv(cache_csv, index=False)
        print(f"\n  → {moon_id} orbits saved to '{cache_csv}'")

    return orbits


def fetch_reference_orbits(epoch_jds: list,
                           nereid_cache: str = NEREID_CACHE_DEFAULT,
                           triton_cache: str = TRITON_CACHE_DEFAULT) -> tuple:
    """
    Fetch per-epoch orbit loops for Nereid and Triton.
    Returns (nereid_orbits_dict, triton_orbits_dict).
    """
    print("\nFetching Nereid orbits (one period per rendered epoch) …")
    nereid = _fetch_moon_orbits(
        moon_id="Nereid", period_days=NEREID_PERIOD_DAYS,
        step=NEREID_STEP, step_hours=48.0,
        epoch_jds=epoch_jds, cache_csv=nereid_cache,
    )
    print("\nFetching Triton orbits (one period per rendered epoch) …")
    triton = _fetch_moon_orbits(
        moon_id="Triton", period_days=TRITON_PERIOD_DAYS,
        step=TRITON_STEP, step_hours=6.0,
        epoch_jds=epoch_jds, cache_csv=triton_cache,
    )
    return nereid, triton


# ─────────────────────────────────────────────
# 3.  Plotting helpers
# ─────────────────────────────────────────────

NEPTUNE_RADIUS_MKM = 0.024764

BG            = "#000000"
TRAJ_COLOR    = "#ff6464"    # red — retrograde
HEAD_COLOR    = "#ffffff"    # current-position dot
NEREID_COLOR  = "#ff00ff"    # magenta — Nereid orbit
TRITON_COLOR  = "#ff00ff"    # magenta — Triton orbit
TAIL_ALPHA    = 1.0
TAIL_LEN      = 500          # frames of fading tail (time length: frames x timestep)


def draw_neptune(ax):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    r = NEPTUNE_RADIUS_MKM
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#00ffff", alpha=1,
                    linewidth=0, antialiased=True)


def make_frame(i: int, df: pd.DataFrame,
               xlim, ylim, zlim,
               nereid_orbits: dict, triton_orbits: dict,
               epoch_jd: float,
               dpi: int, total_frames: int, figsize=(7, 7)) -> Image.Image:

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    # ── axes styling ─────────────────────────────────────────────────
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("black")  # pane border color
    
    grid_alpha = 0.3
    ax.xaxis.gridlines.set_alpha(grid_alpha)
    ax.yaxis.gridlines.set_alpha(grid_alpha)
    ax.zaxis.gridlines.set_alpha(grid_alpha)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, grid_alpha))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, grid_alpha))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, grid_alpha))

    ax.tick_params(colors="white", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
    ax.set_xlabel("X (10⁶ km)", labelpad=4, fontsize=9)
    ax.set_ylabel("Y (10⁶ km)", labelpad=4, fontsize=9)
    ax.set_zlabel("Z (10⁶ km)", labelpad=4, fontsize=9)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)

    # ── Nereid orbit (current epoch) ────────────────────────────────
    if epoch_jd in nereid_orbits:
        nx, ny, nz = nereid_orbits[epoch_jd]
        ax.plot(nx, ny, nz, color=NEREID_COLOR, alpha=0.9,
                linewidth=1.0, linestyle="-", zorder=1)

    # ── Triton orbit (current epoch) ────────────────────────────────
    if epoch_jd in triton_orbits:
        tx, ty, tz = triton_orbits[epoch_jd]
        ax.plot(tx, ty, tz, color=TRITON_COLOR, alpha=0.9,
                linewidth=1.0, linestyle="-", zorder=1)

    # ── Fixed 2D legend-style labels for reference orbits ───────────
    '''
    ax.text2D(0.5, 0.92, "Nereid", transform=ax.transAxes,
              color=NEREID_COLOR, fontsize=8, alpha=0.9, va="top")
    ax.text2D(0.51, 0.5, "Triton", transform=ax.transAxes,
              color=TRITON_COLOR, fontsize=8, alpha=0.9, va="top")
    '''

    # ── Neptune sphere ────────────────────────────────────────────────
    draw_neptune(ax)

    # ── ghost full orbit (very dim) ──────────────────────────────────
    ax.plot(df["x"], df["y"], df["z"],
            color=TRAJ_COLOR, alpha=0.1, linewidth=0.75, zorder=2)

    # ── fading tail ──────────────────────────────────────────────────
    tail_start = max(0, i - TAIL_LEN)
    seg        = df.iloc[tail_start : i + 1]
    n_seg      = len(seg)
    if n_seg > 1:
        for j in range(n_seg - 1):
            alpha = TAIL_ALPHA * ((j + 1) / n_seg) ** 1.8
            ax.plot(
                seg["x"].iloc[j : j + 2],
                seg["y"].iloc[j : j + 2],
                seg["z"].iloc[j : j + 2],
                color=TRAJ_COLOR, alpha=alpha, linewidth=1.25, zorder=3,
            )

    # ── current position dot ─────────────────────────────────────────
    row = df.iloc[i]
    ax.scatter([row["x"]], [row["y"]], [row["z"]],
               color=HEAD_COLOR, s=18, zorder=6, depthshade=False)

    # ── date label ───────────────────────────────────────────────────
    lbl = ax.text2D(
        0.50, 0.97,
        f"Neso (Neptune XIII) - {int(row['year'])}-Jan-01",
        transform=ax.transAxes,
        ha="center", va="top",
        color="white", fontsize=16, fontweight="bold",
        #fontfamily="monospace",
    )
    lbl.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])

    # ── slowly rotating camera ───────────────────────────────────────
    ax.view_init(elev=28, azim=225)  # default oblique
    #ax.view_init(elev=90, azim=180)  # polar

    plt.tight_layout(pad=0.2)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────
# 4.  GIF builder
# ─────────────────────────────────────────────

def build_gif(df: pd.DataFrame, out_path: str,
              fps: int, dpi: int, stride: int = 150, figsize=(6, 6),
              nereid_cache: str = NEREID_CACHE_DEFAULT,
              triton_cache: str = TRITON_CACHE_DEFAULT):

    n          = len(df)
    indices    = list(range(0, n, stride))
    n_frames   = len(indices)
    pad        = 0.10

    # Collect the JD for each rendered epoch so we can look up the right orbit
    epoch_jds = df["datetime_jd"].iloc[indices].tolist()

    # Fetch per-epoch orbits for Nereid and Triton
    nereid_orbits, triton_orbits = fetch_reference_orbits(
        epoch_jds, nereid_cache=nereid_cache, triton_cache=triton_cache
    )

    # Axis limits: encompass Neso + Nereid + Triton
    all_ref_x = np.concatenate([v[0] for v in nereid_orbits.values()] +
                                [v[0] for v in triton_orbits.values()])
    all_ref_y = np.concatenate([v[1] for v in nereid_orbits.values()] +
                                [v[1] for v in triton_orbits.values()])
    all_ref_z = np.concatenate([v[2] for v in nereid_orbits.values()] +
                                [v[2] for v in triton_orbits.values()])

    def lims(col, ref_vals):
        lo = min(df[col].min(), ref_vals.min())
        hi = max(df[col].max(), ref_vals.max())
        rng = max(hi - lo, 1e-6)
        return lo - pad * rng, hi + pad * rng

    xlim = lims("x", all_ref_x)
    ylim = lims("y", all_ref_y)
    zlim = lims("z", all_ref_z)

    print(f"\nRendering {n_frames} frames (stride={stride}, {n} total epochs) "
          f"at {dpi} dpi  ({fps} fps) …")
    frames = []
    for k, i in enumerate(indices):
        if k % 100 == 0 or k == n_frames - 1:
            print(f"  frame {k+1:5d}/{n_frames}  ({100*(k+1)/n_frames:.0f}%)",
                  end="\r", flush=True)
        jd = df["datetime_jd"].iloc[i]
        frames.append(
            make_frame(i, df, xlim, ylim, zlim,
                       nereid_orbits, triton_orbits, jd,
                       dpi=dpi, total_frames=n, figsize=figsize)
        )

    print(f"\nSaving → '{out_path}' …")
    delay_ms = max(2, int(1000 / fps))   # GIF minimum ~2 cs = 20 ms
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=delay_ms,
        optimize=False,
    )
    mb = os.path.getsize(out_path) / 1e6
    print(f"Done!  {out_path}  ({mb:.1f} MB,  {n_frames} frames @ {fps} fps)")
    return out_path


def convert_to_mp4(gif_path: str, fps: int) -> None:
    """Convert a GIF to MP4 using ffmpeg (must be installed)."""
    import subprocess
    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "gif", "-i", gif_path,
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path,
    ]
    print(f"Converting to MP4 → '{mp4_path}' …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr)
    else:
        mb = os.path.getsize(mp4_path) / 1e6
        print(f"MP4 saved!  {mp4_path}  ({mb:.1f} MB)")


# ─────────────────────────────────────────────
# 5.  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animated GIF of moon's 3D trajectory around Neptune."
    )
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip Horizons query; load existing --cache CSV.")
    parser.add_argument("--cache",  default="moon_cache.csv",
                        help="CSV cache path for moon (default: moon_cache.csv).")
    parser.add_argument("--nereid-cache", default=NEREID_CACHE_DEFAULT,
                        help=f"CSV cache path for Nereid orbits (default: {NEREID_CACHE_DEFAULT}).")
    parser.add_argument("--triton-cache", default=TRITON_CACHE_DEFAULT,
                        help=f"CSV cache path for Triton orbits (default: {TRITON_CACHE_DEFAULT}).")
    parser.add_argument("--out",    default="neso_trajectory_1600-2400.gif",
                        help="Output GIF path (default: neso_trajectory_1600-2400.gif).")
    parser.add_argument("--fps",    type=int, default=60,
                        help="Frames per second (default: 60). GIF viewers cap at ~50 fps.")
    parser.add_argument("--dpi",    type=int, default=120,
                        help="Figure DPI (default: 120).")
    parser.add_argument("--stride", type=int, default=150,
                        help="Render every Nth frame — e.g. --stride 150 gives 150× speedup "
                             "and 10× smaller file (default: 1 = all frames).")
    parser.add_argument("--figsize", type=float, default=6.0,
                        help="Square figure side in inches (default: 6). "
                             "Smaller = faster render + smaller file.")
    parser.add_argument("--mp4", action="store_true",
                        help="Also convert the finished GIF to MP4 via ffmpeg "
                             "for true high-fps playback.")
    args = parser.parse_args()

    # ── data ────────────────────────────────────────────────────────
    if args.no_fetch:
        if not os.path.exists(args.cache):
            sys.exit(f"Cache '{args.cache}' not found — run without --no-fetch first.")
        df = load_cache(args.cache)
    elif os.path.exists(args.cache):
        print(f"Cache '{args.cache}' found — skipping Horizons query.")
        print("  (Delete the file to force a fresh fetch.)")
        df = load_cache(args.cache)
    else:
        df = fetch_horizons(args.cache)

    for col in ("x", "y", "z", "year"):
        if col not in df.columns:
            sys.exit(f"CSV missing column '{col}'.")

    print(f"\nTrajectory: {len(df)} epochs  "
          f"({df['year'].iloc[0]}–{df['year'].iloc[-1]})")
    print(f"  X  {df['x'].min():.3f} → {df['x'].max():.3f}  (10⁶ km)")
    print(f"  Y  {df['y'].min():.3f} → {df['y'].max():.3f}  (10⁶ km)")
    print(f"  Z  {df['z'].min():.3f} → {df['z'].max():.3f}  (10⁶ km)")

    gif_path = build_gif(df, out_path=args.out, fps=args.fps, dpi=args.dpi,
                         stride=args.stride, figsize=(args.figsize, args.figsize),
                         nereid_cache=args.nereid_cache,
                         triton_cache=args.triton_cache)

    if args.mp4:
        convert_to_mp4(gif_path, fps=args.fps)


if __name__ == "__main__":
    main()