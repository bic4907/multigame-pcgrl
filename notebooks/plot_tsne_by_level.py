"""
notebooks/plot_tsne_by_level.py

encoder/export_tsne.py (혹은 notebooks/view_encoder_tsne.py)가 만들어낸
embeddings_state_text.npz(차원 축소 전 raw 임베딩)를 읽어서, 한 modality(기본 state,
--modality text로 전환 가능)의 임베딩만 게임별로 서브샘플링한 뒤 t-SNE를 직접
다시 돌려 그린다.

t-SNE는 항상 모든(선택된 reward_enum의) 게임을 함께 넣어 계산한다 — --only-game은
계산이 끝난 뒤 "어느 게임만 그릴지"를 고르는 순수 렌더링 옵션이다. 그래야 게임별
개별 figure가 전체 figure와 동일한 좌표계를 공유해서 서로 비교 가능하다 (한 게임만
따로 넣어 t-SNE를 다시 돌리면 그때마다 기하가 완전히 달라져서 비교가 불가능하다).
같은 (run-dir, dim, reward_enum, max_per_game, perplexity, early_exaggeration, seed)
조합은 run-dir 안에 캐시해서 여러 번 다시 계산하지 않는다.

--unseen-games로 지정한 게임은 별표(*) 마커 + red로, 나머지 seen 게임은 원(o)
마커 + 고유 색(blue/green/orange/purple)으로 그린다.

각 게임마다 "낮은 레벨 대표점 → 높은 레벨 대표점"을 잇는 화살표 하나를 그려서,
서로 다른 게임의 semantic embedding이 레벨 변화에 따라 비슷한 방향으로 움직이는지
(domain alignment) 한눈에 보여준다. 정확한 경로가 아니라 "느낌"을 전달하는 것이
목적이라 중간 bin은 표시하지 않는다.

encoder/visualization/ 쪽 export 파이프라인은 건드리지 않는, 별개의
독립 스크립트다 (raw t-SNE export 결과 자체는 그대로 pending 상태로 둔다).

기본값: --dim 3d, --unseen-games dm, --reward-enum 1, --max-per-game 250.
--only-game을 주면 해당 게임 포인트만 그린다 (게임별 개별 figure용).

Usage:
    python notebooks/plot_tsne_by_level.py --run-dir <run_dir> --out notebooks/tsne_by_level/embed_visualize_3d.png
    python notebooks/plot_tsne_by_level.py --run-dir <run_dir> --only-game zelda --out notebooks/tsne_by_level/embed_visualize_3d_zelda.png
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

_PRETENDARD_PATH = Path.home() / ".local" / "share" / "fonts" / "Pretendard-Regular.otf"
if _PRETENDARD_PATH.exists():
    import matplotlib.font_manager as fm

    fm.fontManager.addfont(str(_PRETENDARD_PATH))
    plt.rcParams["font.family"] = "Pretendard"
else:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 15

ROOT = Path(__file__).resolve().parents[1]


class Arrow3D(FancyArrowPatch):
    """3D 좌표를 받아 화면에 투영해서 그리는 FancyArrowPatch.
    mplot3d에는 3D용 FancyArrowPatch가 없어서, 매 프레임 3D→2D 투영 좌표를
    다시 계산해 2D patch로 그리는 표준적인 우회법이다 (ax.quiver의 원뿔형
    화살촉이 시점에 따라 뒤틀려 보이는 문제를 피할 수 있다)."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        from mpl_toolkits.mplot3d.proj3d import proj_transform

        x, y, z = proj_transform(*self._verts3d, self.axes.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return min(z)

GAME_BASE_COLORS = {
    "dungeon": "#d62728",  # red
    "pokemon": "#2ca02c",  # green
    "doom": "#ff7f0e",     # orange (unseen일 땐 red로 덮어씀)
    "sokoban": "#1f77b4",  # blue
    "zelda": "#9467bd",    # purple
}
UNSEEN_COLOR = "#d62728"  # red — unseen 게임은 항상 이 색
# 범례에 표시할 게임 순서 (여기 없는 게임은 뒤에 알파벳순으로 붙는다).
LEGEND_GAME_ORDER = ["doom", "pokemon", "sokoban", "dungeon", "zelda"]
ABBR_TO_GAME = {
    "dg": "dungeon",
    "pk": "pokemon",
    "sk": "sokoban",
    "dm": "doom",
    "zd": "zelda",
}


def game_color(game: str, unseen_games: set[str]) -> str:
    if game in unseen_games:
        return UNSEEN_COLOR
    return GAME_BASE_COLORS.get(game, "#7f7f7f")


def parse_unseen_games(abbr: str | None) -> set[str]:
    if not abbr:
        return set()
    games = set()
    for i in range(0, len(abbr), 2):
        chunk = abbr[i : i + 2]
        games.add(ABBR_TO_GAME.get(chunk, chunk))
    return games


def subsample_per_game(games: np.ndarray, max_per_game: int, seed: int = 0) -> np.ndarray:
    """게임별로 최대 max_per_game개까지만 남기는 인덱스를 뽑는다. max_per_game<=0이면 전부 사용."""
    if max_per_game <= 0:
        return np.arange(len(games))
    rng = np.random.default_rng(seed)
    keep = []
    for game in sorted(set(games.tolist())):
        idx = np.where(games == game)[0]
        if len(idx) > max_per_game:
            idx = rng.choice(idx, size=max_per_game, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def axis_limits(values: np.ndarray, percentile: float, pad_frac: float) -> tuple[float, float]:
    """percentile<100이면 중심 percentile% 구간만 잡아 축을 좁힌다 (바깥 점은 잘려서 그려짐)."""
    if percentile >= 100:
        lo, hi = float(values.min()), float(values.max())
    else:
        half = (100.0 - percentile) / 2.0
        lo, hi = float(np.percentile(values, half)), float(np.percentile(values, 100.0 - half))
    margin = (hi - lo) * pad_frac or 1.0
    return lo - margin, hi + margin


def compute_level_endpoints(
    coords: np.ndarray, games: np.ndarray, quantized_bins: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """게임별로 (가장 낮은 bin의 평균 좌표, 가장 높은 bin의 평균 좌표) 를 반환한다."""
    result = {}
    for game in sorted(set(games.tolist())):
        game_mask = games == game
        bins_here = sorted(set(quantized_bins[game_mask].tolist()))
        if len(bins_here) < 2:
            continue
        lo_b, hi_b = bins_here[0], bins_here[-1]
        lo = coords[game_mask & (quantized_bins == lo_b)].mean(axis=0)
        hi = coords[game_mask & (quantized_bins == hi_b)].mean(axis=0)
        result[game] = (lo, hi)
    return result


def draw_direction_arrow(
    ax, endpoints_by_game: dict[str, tuple[np.ndarray, np.ndarray]], unseen_games: set[str], is_3d: bool,
    depth_alpha_fn=None, depth_zorder_fn=None,
) -> None:
    """게임별로 낮은 레벨 대표점 → 높은 레벨 대표점을 잇는 화살표 하나를 그린다 ("arrow" 모드).

    ax.quiver의 원뿔형 화살촉은 시점에 따라 뒤틀려 보이는 경우가 많아
    (mplot3d가 화살촉 자체를 3D로 그리기 때문), 대신 매 프레임 3D→2D로
    투영해서 그리는 FancyArrowPatch(Arrow3D)를 쓴다 — 항상 깔끔한 2D 화살표
    모양을 유지한다. arrowstyle="simple"은 화살표 전체(몸통+머리)가 하나의
    채워진 도형이라, facecolor/edgecolor로 진짜 "감싸는" 테두리가 생긴다
    (-|> 는 몸통이 단순 선이라 edgecolor를 줘도 테두리처럼 안 보였다).

    depth_alpha_fn이 주어지면 화살표 중점의 우하단 축(t-SNE dim 2) 값으로 알파를
    계산해서, scatter 포인트와 같은 규칙으로 뒤로 갈수록 옅어지게 한다.
    depth_zorder_fn이 주어지면 같은 중점 기준으로 zorder도 매겨서, 화살표가
    자기보다 앞쪽에 있는 점/화살표에 항상 가려지도록 한다.

    화살표 길이(=낮은/높은 레벨 대표점 사이 거리)가 짧을수록 옅은(흰색에 가까운)
    색, 길수록 게임 고유 색 그대로 진하게 나오도록 색 자체를 블렌딩한다
    (투명도 대신 색으로 표현 — 뒤에 겹친 점들이 비쳐 보이는 걸 피하기 위함).
    """
    import matplotlib.colors as mcolors

    lengths = {
        game: float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
        for game, (start, end) in endpoints_by_game.items()
    }
    len_lo, len_hi = min(lengths.values()), max(lengths.values())
    min_blend = 0.35  # 가장 짧은 화살표도 완전히 흰색이 되지 않도록 최소 채도 유지

    arrow_style = "simple,head_length=1.05,head_width=0.85,tail_width=0.24"
    # jitter 화살표: 각 게임의 실제 화살표와 같은 출발점에서, 방향을 살짝 흩뜨린 짧은
    # 회색 화살표 몇 개를 함께 그려서 "여러 방향이 대체로 한 방향으로 정렬된다"는
    # 느낌을 준다 (본 화살표보다 뒤/아래 zorder로 깔아둔다).
    jitter_style = "simple,head_length=1.0,head_width=0.8,tail_width=0.22"
    jitter_frac = 0.5    # 본 화살표 길이 대비 짧게
    # 게임별로 jitter 방향을 특정 각도(도 단위, xy 평면 회전)로 고정한다. 순수 난수로 두면
    # 던전 화살표가 다른 클러스터로 뻗어 모양을 깨서, 게임마다 깔끔한 방향을 직접 지정한다.
    jitter_deg = {"doom": -10.0, "dungeon": 55.0, "pokemon": -45.0, "sokoban": 20.0, "zelda": -55.0}

    for game, (start, end) in endpoints_by_game.items():
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        direction = end - start
        seg_len = float(np.linalg.norm(direction)) or 1.0
        base_color = np.array(mcolors.to_rgb(game_color(game, unseen_games)))
        t = (lengths[game] - len_lo) / (len_hi - len_lo) if len_hi > len_lo else 1.0
        blend = min_blend + (1.0 - min_blend) * t
        color = tuple(1.0 - blend * (1.0 - base_color))  # white -> base_color 보간
        mid_y = (start[1] + end[1]) / 2.0
        alpha = depth_alpha_fn(mid_y) if depth_alpha_fn is not None else 1.0
        zorder = depth_zorder_fn(mid_y) if depth_zorder_fn is not None else 50
        # doom 화살표를 다른 화살표들보다 항상 맨 앞에 오게 zorder를 올린다.
        if game == "doom":
            zorder = zorder + 5

        # --- jitter 화살표 (회색, 짧게, 본 화살표와 동일한 시작점) ---
        # 본 화살표 방향을 xy 평면에서 지정 각도만큼 회전시켜 jitter 방향을 만든다.
        theta = np.deg2rad(jitter_deg.get(game, 90.0))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        j_dir = direction.copy()
        j_dir[0] = direction[0] * cos_t - direction[1] * sin_t
        j_dir[1] = direction[0] * sin_t + direction[1] * cos_t
        j_end = start + (j_dir / (np.linalg.norm(j_dir) or 1.0)) * seg_len * jitter_frac
        j_kwargs = dict(
            mutation_scale=15, arrowstyle=jitter_style,
            facecolor="#8a8a8a", edgecolor="#000000", linewidth=0.6,
            shrinkA=0, shrinkB=0,  # tail을 start에 정확히 고정 (기본 shrink=2 때문에 벌어지던 문제 해결)
            zorder=zorder - 1, clip_on=False, alpha=0.65 * alpha,
        )
        if is_3d:
            jarr = Arrow3D([start[0], j_end[0]], [start[1], j_end[1]], [start[2], j_end[2]], **j_kwargs)
            ax.add_artist(jarr)
        else:
            jarr = FancyArrowPatch((start[0], start[1]), (j_end[0], j_end[1]), **j_kwargs)
            ax.add_patch(jarr)

        kwargs = dict(
            mutation_scale=16, arrowstyle=arrow_style,
            facecolor=color, edgecolor="black", linewidth=0.8,
            shrinkA=0, shrinkB=0,  # 본 화살표 tail도 start에 정확히 고정
            zorder=zorder, clip_on=False, alpha=alpha,
        )
        if is_3d:
            arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)
            ax.add_artist(arrow)
        else:
            arrow = FancyArrowPatch((start[0], start[1]), (end[0], end[1]), **kwargs)
            ax.add_patch(arrow)


def compute_bin_centroids(
    coords: np.ndarray, games: np.ndarray, quantized_bins: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """게임별로 bin 오름차순 (bin 배열, 그 bin에 속한 점들의 평균 좌표) 를 반환한다."""
    result = {}
    for game in sorted(set(games.tolist())):
        game_mask = games == game
        bins_here = sorted(set(quantized_bins[game_mask].tolist()))
        centroids = np.array([
            coords[game_mask & (quantized_bins == b)].mean(axis=0) for b in bins_here
        ])
        result[game] = (np.array(bins_here), centroids)
    return result


def draw_bin_cluster_path(
    ax, centroids_by_game: dict[str, tuple[np.ndarray, np.ndarray]], unseen_games: set[str], is_3d: bool
) -> None:
    """bin 오름차순 centroid를 선으로 잇고, 원 안에 bin 번호를 적는다 ("cluster" 모드)."""
    text_kwargs = dict(color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=100)
    for game, (bins_here, centroids) in centroids_by_game.items():
        if len(centroids) < 2:
            continue
        color = game_color(game, unseen_games)
        if is_3d:
            ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], color=color, linewidth=2.2, zorder=5)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       color=color, s=280, edgecolors="black", linewidths=0.8, zorder=6)
            for b, c in zip(bins_here, centroids):
                ax.text(c[0], c[1], c[2], str(int(b)), **text_kwargs)
        else:
            ax.plot(centroids[:, 0], centroids[:, 1], color=color, linewidth=2.2, zorder=5)
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       color=color, s=280, edgecolors="black", linewidths=0.8, zorder=6)
            for b, c in zip(bins_here, centroids):
                ax.text(c[0], c[1], str(int(b)), **text_kwargs)


def find_representative_index(
    coords: np.ndarray, games: np.ndarray, quantized_bins: np.ndarray, game: str, target_bin: int
) -> int | None:
    """game과 quantized_bin이 일치하는 점들 중 centroid에 가장 가까운 점 하나를 대표로 고른다."""
    mask = (games == game) & (quantized_bins == target_bin)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    pts = coords[idx]
    centroid = pts.mean(axis=0)
    return int(idx[np.argmin(np.sum((pts - centroid) ** 2, axis=1))])


# game -> (axes-fraction 위치, 앵커 자기 방향) — 큐브와 안 겹치게 충분히 바깥/위로 두고, 세로 간격도
# 이미지 크기(zoom)보다 넉넉하게 벌린다.
# 좌우로 나누지 않고, 큐브 위쪽에 부채꼴(fan)로 5개를 펼쳐 놓는다 (img.png 참고).
# x는 왼쪽→오른쪽으로 (예전보다 촘촘하게) 벌리고, y는 가운데(unseen 게임)가 가장 높고
# 가장자리로 갈수록 낮아지는 완만한 호를 그린다.
LEVEL_THUMBNAIL_SLOTS: dict[str, tuple[float, float]] = {
    "zelda": (-0.25, 0.65),
    "dungeon": (0.0, 1.17),
    "pokemon": (0.53, 1.22),
    "sokoban": (1.08, 1.17),
    "doom": (1.31, 0.65),
}


def draw_level_thumbnails(
    fig, ax, coords: np.ndarray, games: np.ndarray, quantized_bins: np.ndarray,
    unseen_games: set[str], game_bin_map: dict[str, int], levels_dir: Path, is_3d: bool,
    zoom: float = 0.45,
) -> None:
    """게임별 대표 레벨 썸네일(notebooks/render_level_thumbnails.py로 미리 렌더링된 PNG)을
    큐브 바깥에 배치하고, 해당 (game, quantized_bin)의 대표 임베딩 포인트와 선으로 잇는다 —
    "이 임베딩에서 이런 형태의 맵이 나온다"를 보여주기 위함.

    라벨은 render_level_thumbnails.py가 captions.json에 저장한 그 대표 샘플의
    instruction 텍스트를 쓴다 (img.png 예시처럼 자연어 설명을 이미지에 붙인다).
    captions.json이 없으면 "game (bin N)"으로 폴백한다."""
    import json
    import textwrap

    import matplotlib.image as mpimg
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    captions_path = levels_dir / "captions.json"
    captions = json.loads(captions_path.read_text(encoding="utf-8")) if captions_path.exists() else {}

    for game, target_bin in game_bin_map.items():
        idx = find_representative_index(coords, games, quantized_bins, game, target_bin)
        if idx is None:
            print(f"[level-thumbnail] skip: no point for game={game} bin={target_bin}")
            continue
        img_path = levels_dir / f"{game}_bin{target_bin}.png"
        if not img_path.exists():
            print(f"[level-thumbnail] skip: missing image {img_path}")
            continue
        img = mpimg.imread(img_path)
        point = coords[idx]

        if is_3d:
            from mpl_toolkits.mplot3d.proj3d import proj_transform

            x2, y2, _ = proj_transform(point[0], point[1], point[2], ax.get_proj())
            xy = (x2, y2)
        else:
            xy = (point[0], point[1])

        line_color = UNSEEN_COLOR if game in unseen_games else "black"
        box_xy = LEVEL_THUMBNAIL_SLOTS.get(game, (1.22, 0.5))
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox, xy, xycoords="data",
            xybox=box_xy, boxcoords="axes fraction",
            frameon=False, pad=0.0,
            # 연결선의 zorder를 방향 화살표(n_points_zorder_max+10)보다 낮게 둬서,
            # 화살표가 항상 연결선 위에 그려지게 한다 (이미지 박스 자체는 zorder=200 유지).
            arrowprops=dict(arrowstyle="-", color=line_color, linewidth=1.3, shrinkA=0, shrinkB=2, zorder=15),
            annotation_clip=False, zorder=200,
        )
        ax.add_artist(ab)
        # 라벨: 대표 샘플의 instruction 텍스트 (없으면 game/bin 폴백). 이미지 좌우 폭을
        # 넘지 않도록, 이미지 표시 폭(px * zoom ≈ point)에 맞춰 줄바꿈 폭을 정한다.
        fontsize = 17
        img_width_pt = img.shape[1] * zoom
        wrap_width = max(12, int(img_width_pt / (fontsize * 0.32)))
        caption = captions.get(f"{game}_bin{target_bin}", f"{game} (bin {target_bin})")
        # 단어 중간이 잘리면 보기 나쁘므로, 폭을 넘더라도 단어 단위는 유지한다.
        caption = "\n".join(textwrap.wrap(caption, width=wrap_width, break_long_words=False, break_on_hyphens=False))
        # 이미지 크기(zoom)에 안 흔들리도록 "포인트" 단위 오프셋으로 이미지 위에 붙인다.
        ax.annotate(
            caption, xy=box_xy, xycoords="axes fraction",
            xytext=(0, 62), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize, color="black", fontweight="bold",
            fontstyle="italic", fontfamily="Arial",
            annotation_clip=False, zorder=201,
        )


def run_tsne(
    embeddings: np.ndarray,
    n_components: int,
    perplexity: float,
    early_exaggeration: float,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(float(perplexity), max(5.0, (len(embeddings) - 1) / 3.0))
    kwargs = {
        "n_components": int(n_components),
        "perplexity": perplexity,
        "early_exaggeration": float(early_exaggeration),
        "init": "pca",
        "learning_rate": "auto",
        "metric": "cosine",
        "random_state": int(seed),
        "verbose": 1,
    }
    tsne_params = inspect.signature(TSNE).parameters
    kwargs["max_iter" if "max_iter" in tsne_params else "n_iter"] = int(max_iter)
    return TSNE(**kwargs).fit_transform(embeddings)


def compute_tsne_data(
    embeddings_npz_path: Path,
    dim: str,
    max_per_game: int,
    perplexity: float,
    early_exaggeration: float,
    max_iter: int,
    seed: int,
    reward_enum: int | None,
    modality: str = "state",
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모든 게임을 함께 넣어 t-SNE를 계산한다 (coords, games, quantized_bins).
    modality는 "state" 또는 "text" — 어느 임베딩을 볼지 고른다.

    같은 파라미터 조합은 embeddings_npz_path 옆에 캐시해서, 게임별 figure를 여러 번
    뽑을 때마다 t-SNE를 다시 돌리지 않는다.
    """
    cache_path = embeddings_npz_path.parent / (
        f"tsne_cache_{modality}_{dim}_mpg{max_per_game}_re{reward_enum}_"
        f"p{perplexity:g}_ee{early_exaggeration:g}_it{max_iter}_s{seed}.npz"
    )
    if use_cache and cache_path.exists():
        c = np.load(cache_path, allow_pickle=True)
        print(f"loaded cached t-SNE: {cache_path}")
        return c["coords"], c["games"], c["quantized_bins"]

    d = np.load(embeddings_npz_path, allow_pickle=True)
    embeddings = d["embeddings"]
    games = d["games"]
    types = d["types"]
    quantized_bins = d["quantized_bins"].astype(int)
    reward_enums = d["reward_enums"].astype(int)

    mask = types == modality
    if reward_enum is not None:
        mask = mask & (reward_enums == reward_enum)
    embeddings, games, quantized_bins = embeddings[mask], games[mask], quantized_bins[mask]
    print(f"{modality} points before subsample (reward_enum={reward_enum}): {len(games)} "
          f"({', '.join(f'{g}={c}' for g, c in zip(*np.unique(games, return_counts=True)))})")

    keep = subsample_per_game(games, max_per_game, seed=seed)
    embeddings, games, quantized_bins = embeddings[keep], games[keep], quantized_bins[keep]
    print(f"{modality} points after subsample (max {max_per_game}/game): {len(games)}")

    n_components = 3 if dim == "3d" else 2
    print(f"running t-SNE: n={len(embeddings)}, dim={n_components}, "
          f"perplexity={perplexity}, early_exaggeration={early_exaggeration}")
    coords = run_tsne(embeddings, n_components, perplexity, early_exaggeration, max_iter, seed)

    if use_cache:
        np.savez_compressed(cache_path, coords=coords, games=games, quantized_bins=quantized_bins)
        print(f"cached t-SNE: {cache_path}")
    return coords, games, quantized_bins


def render_plot(
    coords: np.ndarray,
    games: np.ndarray,
    quantized_bins: np.ndarray,
    out_path: Path,
    title: str,
    unseen_games: set[str],
    axis_percentile: float,
    show_trajectory: bool = True,
    trajectory_style: str = "cluster",
    only_game: str | None = None,
    depth_fade: bool = True,
    save_pdf: bool = True,
    flip_depth_axis: bool = False,
    level_thumbnails: dict[str, int] | None = None,
    levels_dir: Path | None = None,
) -> None:
    if only_game is not None:
        game_mask = games == only_game
        coords, games, quantized_bins = coords[game_mask], games[game_mask], quantized_bins[game_mask]

    if flip_depth_axis:
        # 우하단(depth) 축 자체를 뒤집는다 — 값이 클수록 앞으로 오게. t-SNE 축의 부호는
        # 원래 임의라서, 이 축 값을 negate해도 임베딩 구조(상대적 거리)는 그대로 보존된다.
        coords = coords.copy()
        coords[:, 1] = -coords[:, 1]

    is_3d = coords.shape[1] == 3
    # dungeon을 범례 맨 앞에 두고 나머지는 알파벳순.
    unique_games = sorted(set(games.tolist()), key=lambda g: (g != "dungeon", g))
    bin_min, bin_max = int(quantized_bins.min()), int(quantized_bins.max())

    # 레벨 썸네일을 붙일 땐 좌우 여백이 훨씬 많이 필요해서 figure를 넓힌다 —
    # 그래야 axes-fraction 오프셋이 실제 인치 단위로도 캡션/이미지를 담을 만큼 벌어진다.
    # 세로로 긴 느낌을 줄이고 정사각형에 가깝게 — 썸네일이 붙을 때 높이를 낮춘다.
    figsize = (9.5, 8.2) if level_thumbnails else (6.4, 6.8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    if is_3d:
        # 위쪽 여백을 확보해서 3D 큐브 상단이 캔버스 밖으로 잘리지 않게 한다.
        # (레벨 썸네일이 붙을 땐 부채꼴을 큐브 바로 위 촘촘하게 두므로 여백을 더 줄인다.)
        fig.subplots_adjust(top=0.88 if level_thumbnails else 0.8)
        # mplot3d는 기본적으로 아티스트별 평균 depth로 zorder를 자동 계산해서
        # 나중에 그린 아티스트가 먼저 그린 것 뒤에 가려질 수 있다. 수동 zorder를 강제한다.
        ax.computed_zorder = False

    # 축 범위를 scatter를 그리기 "전에" 먼저 확정한다. depth-fade가 최종 투영행렬(ax.get_proj())을
    # 써야 카메라 기준 원근이 정확하기 때문이다 (축 범위/줌이 나중에 바뀌면 투영도 달라진다).
    pad = 0.015
    arrow_points = (
        np.array([p for pair in compute_level_endpoints(coords, games, quantized_bins).values() for p in pair])
        if show_trajectory and trajectory_style == "arrow" and bin_max > bin_min
        else None
    )
    for axis, set_lim in zip(range(coords.shape[1]), [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if is_3d else [])):
        values = coords[:, axis]
        if arrow_points is not None:
            values = np.concatenate([values, arrow_points[:, axis]])
        lo, hi = axis_limits(values, axis_percentile, pad)
        if axis == 1:
            # 반전 넣은 depth 축(우하단) 범위를 -10~10으로 좁힌다.
            lo, hi = max(lo, -10.0), min(hi, 10.0)
        set_lim(lo, hi)
    if is_3d:
        ax.set_box_aspect((1, 1, 1), zoom=1.25)

    # 우하단 축(axis index 1, t-SNE dim 2)을 기준으로 값이 클수록(=뒤쪽) 옅어지게 한다.
    depth_min_alpha = 0.2
    depth_lo, depth_hi = float(coords[:, 1].min()), float(coords[:, 1].max())
    n_bins = 24 if is_3d and depth_fade else 1
    n_points_zorder_max = n_bins - 1

    def _depth_face_colors(pts: np.ndarray, base_color: str) -> np.ndarray:
        import matplotlib.colors as mcolors

        rgba = np.tile(mcolors.to_rgba(base_color), (len(pts), 1))
        if not depth_fade or len(pts) == 0 or depth_hi - depth_lo < 1e-9:
            return rgba
        t = (pts[:, 1] - depth_lo) / (depth_hi - depth_lo)
        rgba[:, 3] = 1.0 - (1.0 - depth_min_alpha) * t
        return rgba

    def _depth_alpha(y: float) -> float:
        if not depth_fade or depth_hi - depth_lo < 1e-9:
            return 1.0
        t = float(np.clip((y - depth_lo) / (depth_hi - depth_lo), 0.0, 1.0))
        return 1.0 - (1.0 - depth_min_alpha) * t

    def _depth_zorder(y: float) -> float:
        """점들과 같은 depth 스케일로 zorder를 매긴다 (뒤=낮음, 앞=높음)."""
        if not (is_3d and depth_fade) or depth_hi - depth_lo < 1e-9:
            return n_points_zorder_max
        t = float(np.clip((y - depth_lo) / (depth_hi - depth_lo), 0.0, 1.0))
        return (1.0 - t) * n_points_zorder_max

    # 점을 뒤(큰 y) → 앞(작은 y) 순으로 나눠서 그린다. zorder를 그 순서에 맞춰 매겨서
    # (뒤=낮은 zorder, 앞=높은 zorder) 항상 앞쪽 점이 뒤쪽 점 위에 그려지게 한다 —
    # 위에서 준 알파(뒤=옅게)와 실제 렌더링 순서가 서로 어긋나지 않도록.
    # marker가 게임별(원/별)로 달라서 같은 depth 구간 안에서도 게임별로 나눠 부른다.
    order = np.argsort(-coords[:, 1])
    bins = np.array_split(order, n_bins)
    for depth_rank, idx_bin in enumerate(bins):
        if len(idx_bin) == 0:
            continue
        for game in unique_games:
            sel = idx_bin[games[idx_bin] == game]
            if len(sel) == 0:
                continue
            is_unseen = game in unseen_games
            color = game_color(game, unseen_games)
            pts = coords[sel]
            if is_unseen:
                kwargs = {"marker": "*", "s": 145, "linewidths": 0.5, "edgecolors": "white"}
            else:
                kwargs = {"marker": "o", "s": 52, "linewidths": 0.5, "edgecolors": "white"}
            # 축 범위를 타이트하게 잡아도(percentile crop) 그 밖으로 나가는 점이 잘리지 않고 보이게 한다.
            kwargs["clip_on"] = False
            kwargs["zorder"] = depth_rank
            if is_3d:
                kwargs["c"] = _depth_face_colors(pts, color)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], depthshade=False, **kwargs)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.85, **kwargs)

    if show_trajectory and bin_max > bin_min:
        if trajectory_style == "arrow":
            endpoints_by_game = compute_level_endpoints(coords, games, quantized_bins)
            # 화살표는 depth-fade 대상에서 빼고 항상 맨 앞(최고 zorder)에 불투명하게 그린다.
            draw_direction_arrow(ax, endpoints_by_game, unseen_games, is_3d,
                                  depth_alpha_fn=None, depth_zorder_fn=lambda y: n_points_zorder_max + 10)
        else:
            centroids_by_game = compute_bin_centroids(coords, games, quantized_bins)
            draw_bin_cluster_path(ax, centroids_by_game, unseen_games, is_3d)

    if level_thumbnails:
        draw_level_thumbnails(
            fig, ax, coords, games, quantized_bins, unseen_games,
            level_thumbnails, levels_dir or (ROOT / "notebooks" / "tsne_by_level" / "levels"), is_3d,
        )

    game_handles = [
        mlines.Line2D([], [], color=game_color(g, unseen_games),
                      marker="*" if g in unseen_games else "o", linestyle="None",
                      markersize=18 if g in unseen_games else 15,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=g.capitalize())
        for g in sorted(
            unique_games,
            key=lambda g: (LEGEND_GAME_ORDER.index(g) if g in LEGEND_GAME_ORDER else len(LEGEND_GAME_ORDER), g),
        )
    ]
    # Type은 큐브 좌하단 빈 공간에, Game은 그래프의 실제 오른쪽(우하단 바깥)에 배치한다.
    seen_unseen_handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=15,
                      markeredgecolor="white", markeredgewidth=0.8, label="Seen"),
        mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=18,
                      markeredgecolor="white", markeredgewidth=0.8, label="Unseen"),
    ]
    type_legend = ax.legend(handles=seen_unseen_handles, title="Type", loc="lower left",
                             bbox_to_anchor=(1.08, -0.15), ncol=1, frameon=True,
                             handletextpad=0.4, fontsize=18, title_fontsize=19, markerscale=1.1)
    ax.add_artist(type_legend)

    ax.legend(handles=game_handles, title="Game", loc="lower right", bbox_to_anchor=(0.0, -0.15),
              ncol=1, frameon=True, handletextpad=0.4, fontsize=18, title_fontsize=19, markerscale=1.1)

    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    if is_3d:
        ax.zaxis.set_major_locator(MultipleLocator(5))
        # 축 눈금 숫자가 축 선에서 너무 떨어져 있어서 여백처럼 보이던 걸 줄인다.
        ax.xaxis.set_tick_params(pad=1)
        ax.yaxis.set_tick_params(pad=1)
        ax.zaxis.set_tick_params(pad=1)

    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight"는 실제 그려진 영역에 딱 맞춰 잘라내므로, 커진 썸네일 위쪽에
    # 숨 쉴 여백을 주려면 pad_inches를 키워야 한다 (subplots_adjust top은 tight crop
    # 앞에서 무시된다).
    pad_inches = 0.2 if level_thumbnails else 0.1
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=pad_inches)
    print(f"saved: {out_path}")
    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad_inches)
        print(f"saved: {pdf_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="raw 임베딩에서 text만 다시 t-SNE 돌려 시각화")
    parser.add_argument("--run-dir", default=None,
                         help="export_tsne 결과 폴더 (embeddings_state_text.npz 를 찾는다)")
    parser.add_argument("--npz", default=None, help="--run-dir 대신 embeddings_state_text.npz 경로를 직접 지정")
    parser.add_argument("--dim", choices=["2d", "3d"], default="3d")
    parser.add_argument("--unseen-games", default="dm",
                         help="unseen 게임 약어 (예: dm, pkzddm). 해당 게임은 별표(*) 마커 + red 색상으로 표시 (기본: dm)")
    parser.add_argument("--only-game", default=None,
                         help="이 게임(정식 이름, 예: zelda) 포인트만 그린다 (t-SNE는 항상 전체로 계산, 렌더링만 필터링)")
    parser.add_argument("--per-game", action="store_true",
                         help="--only-game 대신, t-SNE에 등장하는 모든 게임에 대해 개별 figure를 한 번에 뽑는다"
                              " (파일명: <out의 stem>_<game>.png)")
    parser.add_argument("--max-per-game", type=int, default=600,
                         help="게임별 최대 표시(및 t-SNE 입력) 포인트 수 (기본 600, 0이면 전부 사용)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--early-exaggeration", type=float, default=20.0,
                         help="클수록 클러스터 간 간격(퍼짐)이 커진다 (sklearn 기본 12.0)")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis-percentile", type=float, default=94.0,
                         help="축 범위로 보여줄 중심 percentile 구간 (기본 100=전체 범위, 안 잘림; "
                              "80처럼 100 미만으로 주면 중심 80%%만 표시하고 밖의 점은 잘려서 그려짐)")
    parser.add_argument("--reward-enum", type=int, default=1,
                         help="이 reward_enum(0=region,1=path_length,2=interactable,3=hazard,4=collectable)"
                              " 값을 가진 text만 사용 (기본: 1). 전부 쓰려면 음수를 지정")
    parser.add_argument("--no-trajectory", action="store_true",
                         help="게임별 레벨 경로 표시를 끈다 (기본: 그림)")
    parser.add_argument("--trajectory-style", choices=["cluster", "arrow"], default="arrow",
                         help="'cluster': bin별 centroid를 선+번호로 전부 잇는다. "
                              "'arrow': 낮은/높은 레벨 대표점만 화살표 하나로 잇는다 (기본: cluster)")
    parser.add_argument("--modality", choices=["state", "text"], default="text",
                         help="어느 임베딩을 볼지 (기본: text)")
    parser.add_argument("--no-cache", action="store_true", help="t-SNE 캐시를 쓰지 않고 항상 새로 계산")
    parser.add_argument("--no-depth-fade", action="store_true",
                         help="3D에서 카메라로부터 먼 점을 옅게 보이는 depth shading을 끈다 (기본: 켜짐)")
    parser.add_argument("--no-pdf", action="store_true", help="pdf 버전을 같이 저장하지 않는다 (기본: 같이 저장)")
    parser.add_argument("--flip-depth-axis", action="store_true",
                         help="우하단(depth) 축 값을 negate해서 앞/뒤를 뒤집는다 (기본: 안 뒤집음)")
    parser.add_argument("--level-thumbnails", nargs="+", default=None,
                         help="game=bin 형식으로 지정하면 해당 (game, quantized_bin)의 대표 임베딩 포인트에 "
                              "notebooks/render_level_thumbnails.py로 미리 렌더링해둔 레벨 이미지를 연결해서 보여준다. "
                              "예: --level-thumbnails pokemon=0 zelda=2 sokoban=4 doom=6 dungeon=7")
    parser.add_argument("--levels-dir", default=None,
                         help="레벨 썸네일 PNG가 있는 폴더 (기본: notebooks/tsne_by_level/levels)")
    parser.add_argument("--out", default=None,
                         help="출력 png 경로 (기본: notebooks/tsne_by_level/ 아래 자동 생성)")
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.npz):
        parser.error("--run-dir 또는 --npz 중 하나만 지정하세요.")
    if args.only_game and args.per_game:
        parser.error("--only-game 과 --per-game 은 함께 쓸 수 없습니다.")

    npz_path = Path(args.run_dir) / "embeddings_state_text.npz" if args.run_dir else Path(args.npz)

    out_path = (
        Path(args.out) if args.out
        else ROOT / "notebooks" / "tsne_by_level" / f"{npz_path.parent.name}_{args.dim}_by_level.png"
    )
    unseen_games = parse_unseen_games(args.unseen_games)
    reward_enum = None if args.reward_enum < 0 else args.reward_enum

    coords, games, quantized_bins = compute_tsne_data(
        npz_path,
        dim=args.dim,
        max_per_game=args.max_per_game,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        max_iter=args.max_iter,
        seed=args.seed,
        reward_enum=reward_enum,
        modality=args.modality,
        use_cache=not args.no_cache,
    )

    level_thumbnails = None
    if args.level_thumbnails:
        level_thumbnails = {}
        for item in args.level_thumbnails:
            game, bin_str = item.split("=")
            level_thumbnails[game.strip()] = int(bin_str)
    levels_dir = Path(args.levels_dir) if args.levels_dir else None

    def _render(only_game: str | None, path: Path) -> None:
        render_plot(
            coords, games, quantized_bins, path,
            title=f"CLIP decoder {args.dim.upper()} t-SNE ({args.modality})"
                  + (f" — {only_game}" if only_game else ""),
            unseen_games=unseen_games,
            axis_percentile=args.axis_percentile,
            show_trajectory=not args.no_trajectory,
            trajectory_style=args.trajectory_style,
            only_game=only_game,
            depth_fade=not args.no_depth_fade,
            save_pdf=not args.no_pdf,
            flip_depth_axis=args.flip_depth_axis,
            level_thumbnails=level_thumbnails,
            levels_dir=levels_dir,
        )

    if args.per_game:
        for g in sorted(set(games.tolist())):
            _render(g, out_path.with_name(f"{out_path.stem}_{g}{out_path.suffix}"))


    else:
        _render(args.only_game, out_path)


if __name__ == "__main__":
    main()
