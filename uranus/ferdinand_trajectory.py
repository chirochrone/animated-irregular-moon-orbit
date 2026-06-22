"""
ferdinand_trajectory.py
=======================
Fetches a moon's X,Y,Z position vectors
relative to Uranus's centre (id=799) from JPL Horizons via astroquery,
then renders an animated GIF of the 3-D trajectory timelapse.

Oberon's orbit is drawn from one orbital period of JPL Horizons vectors
(~13.46 days at 6-hour steps, cached to oberon_cache.csv).

Usage
-----
    python ferdinand_trajectory.py                            # fetch + animate
    python ferdinand_trajectory.py --no-fetch                 # reuse cached CSV, re-animate
    python ferdinand_trajectory.py --cache my.csv             # custom cache filename
    python ferdinand_trajectory.py --fps 50 --dpi 40          # tweak output
    python ferdinand_trajectory.py --no-fetch --stride 10      # skip every 10th frame (10× faster)
    python ferdinand_trajectory.py --no-fetch --figsize 5     # smaller figure (faster render)
    python ferdinand_trajectory.py --no-fetch --mp4           # also export MP4 via ffmpeg

Speed tips
----------
    --stride N   Render every Nth frame  (e.g. --stride 10 → ~10× faster, smaller file)
    --fps N      GIF frame delay in ms; GIF viewers cap at ~50 fps in practice
    --figsize N  Square figure side in inches (default 7); smaller = faster + smaller file
    --mp4        Convert finished GIF to MP4 with ffmpeg for true 60 fps playback

Output
------
    ferdinand_cache.csv          — raw Horizons vectors (auto-reused on reruns)
    ferdinand_trajectory.gif     — animated GIF timelapse
    ferdinand_trajectory.mp4     — MP4 (only with --mp4 and ffmpeg installed)
"""

import argparse
import os
import sys

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

    print("Querying JPL Horizons for Ferdinand …")
    obj = Horizons(
        id="Ferdinand",
        location="500@799",       # origin = Uranus body centre
        epochs={
            "start": "1601-Jan-01",
            "stop":  "2399-Jan-01",
            "step":  "20d",       # 20-day timestep
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
# 2.  Oberon orbit — one period from JPL Horizons
# ─────────────────────────────────────────────
#
# Oberon (Uranus IV) NAIF id = 702.
# Orbital period = 13.4632 days (IAU).
# We query one period at 6-hour steps starting 2000-Jan-01,
# giving ~54 points that faithfully trace the true orbit shape.
# Result is cached to oberon_cache.csv so subsequent runs are instant.

OBERON_PERIOD_DAYS = 13.4632
OBERON_CACHE_DEFAULT = "oberon_cache.csv"


def fetch_oberon(cache_csv: str = OBERON_CACHE_DEFAULT) -> tuple:
    """
    Query JPL Horizons for one full orbital period of Oberon
    relative to Uranus centre (500@799), cache to CSV, and return
    (x_arr, y_arr, z_arr) in 10^6 km.
    """
    if os.path.exists(cache_csv):
        print(f"Oberon cache '{cache_csv}' found — loading.")
        df = pd.read_csv(cache_csv)
    else:
        from astroquery.jplhorizons import Horizons
        print("Querying JPL Horizons for Oberon (id=702, one orbital period) …")
        # Use a fixed, well-behaved epoch; the orbit shape is stable.
        start = "2000-Jan-01"
        # stop = start + period + a small buffer to close the loop
        stop  = "2000-Jan-15"   # comfortably covers 13.46 days
        obj = Horizons(
            id="Oberon",
            location="500@799",
            epochs={"start": start, "stop": stop, "step": "6h"},
        )
        tbl = obj.vectors()
        df  = tbl[["x", "y", "z"]].to_pandas()

        # Keep only one period's worth of points
        n_period = int(round(OBERON_PERIOD_DAYS * 24 / 6))  # steps of 6 h
        df = df.iloc[: n_period + 1].copy()

        AU_TO_MKM = 149.597870700
        for col in ("x", "y", "z"):
            df[col] = df[col].astype(float) * AU_TO_MKM

        df.to_csv(cache_csv, index=False)
        print(f"  → {len(df)} Oberon points cached to '{cache_csv}'")

    # Close the loop so the plotted curve joins back to the start
    row0 = df.iloc[[0]]
    df   = pd.concat([df, row0], ignore_index=True)
    return df["x"].values, df["y"].values, df["z"].values


# ─────────────────────────────────────────────
# 3.  Plotting helpers
# ─────────────────────────────────────────────

URANUS_RADIUS_MKM = 0.025559

BG            = "#000000"
TRAJ_COLOR    = "#ff6464"    # red — retrograde
HEAD_COLOR    = "#ffffff"    # current-position dot
OBERON_COLOR  = "#ff00ff"    # soft violet — Oberon reference ring
TAIL_ALPHA    = 1.0
TAIL_LEN      = 140          # frames of fading tail; (orbital period / timestep)


def draw_uranus(ax):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    r = URANUS_RADIUS_MKM
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#00ffff", alpha=1,
                    linewidth=0, antialiased=True)


def make_frame(i: int, df: pd.DataFrame,
               xlim, ylim, zlim,
               oberon_xyz,
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

    # ── Oberon reference ring ────────────────────────────────────────
    ox, oy, oz = oberon_xyz
    ax.plot(ox, oy, oz, color=OBERON_COLOR, alpha=1,
            linewidth=1., linestyle="-", zorder=1)
    # tiny label at one point on the ring
    # ax.text(ox[0], oy[0], oz[0], " Oberon", color=OBERON_COLOR, fontsize=8, alpha=0.75)

    # ── Uranus sphere ────────────────────────────────────────────────
    draw_uranus(ax)

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
        f"Ferdinand (Uranus XXIV) - {int(row['year'])}-Jan-01",
        transform=ax.transAxes,
        ha="center", va="top",
        color="white", fontsize=16, fontweight="bold",
        #fontfamily="monospace",
    )
    lbl.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])

    # ── slowly rotating camera ───────────────────────────────────────
    ax.view_init(elev=60, azim=45)

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
              fps: int, dpi: int, stride: int = 180, figsize=(6, 6),
              oberon_cache: str = OBERON_CACHE_DEFAULT):

    n          = len(df)
    indices    = list(range(0, n, stride))
    n_frames   = len(indices)
    pad        = 0.10

    def lims(col, oberon_vals):
        lo = min(df[col].min(), oberon_vals.min())
        hi = max(df[col].max(), oberon_vals.max())
        rng = max(hi - lo, 1e-6)
        return lo - pad * rng, hi + pad * rng

    oberon_xyz = fetch_oberon(oberon_cache)
    ox, oy, oz = oberon_xyz

    xlim = lims("x", ox); ylim = lims("y", oy); zlim = lims("z", oz)

    print(f"Rendering {n_frames} frames (stride={stride}, {n} total epochs) "
          f"at {dpi} dpi  ({fps} fps) …")
    frames = []
    for k, i in enumerate(indices):
        if k % 100 == 0 or k == n_frames - 1:
            print(f"  frame {k+1:5d}/{n_frames}  ({100*(k+1)/n_frames:.0f}%)",
                  end="\r", flush=True)
        frames.append(
            make_frame(i, df, xlim, ylim, zlim, oberon_xyz,
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
        description="Animated GIF of moon's 3D trajectory around Uranus."
    )
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip Horizons query; load existing --cache CSV.")
    parser.add_argument("--cache",  default="moon_cache.csv",
                        help="CSV cache path for moon (default: moon_cache.csv).")
    parser.add_argument("--oberon-cache", default=OBERON_CACHE_DEFAULT,
                        help=f"CSV cache path for Oberon (default: {OBERON_CACHE_DEFAULT}).")
    parser.add_argument("--out",    default="ferdinand_trajectory_1600-2400.gif",
                        help="Output GIF path (default: ferdinand_trajectory_1600-2400.gif).")
    parser.add_argument("--fps",    type=int, default=60,
                        help="Frames per second (default: 60). GIF viewers cap at ~50 fps.")
    parser.add_argument("--dpi",    type=int, default=120,
                        help="Figure DPI (default: 120).")
    parser.add_argument("--stride", type=int, default=180,
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
                         oberon_cache=args.oberon_cache)

    if args.mp4:
        convert_to_mp4(gif_path, fps=args.fps)


if __name__ == "__main__":
    main()
