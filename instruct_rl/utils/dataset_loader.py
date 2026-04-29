"""
instruct_rl/utils/dataset_loader.py
====================================
MultiGameDataset 기반 Instruct 빌더.
jax.jit 바깥에서 호출하여 데이터셋을 로드하고 Instruct 객체를 빌드한다.
"""
from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import hashlib

from instruct_rl.dataclass import Instruct
from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)

# reward_enum → 사람이 읽을 수 있는 이름
REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}


def load_dataset_instruct(config):
    """MultiGameDataset에서 Instruct 객체를 빌드한다.

    Parameters
    ----------
    config : TrainConfig
        dataset_game, dataset_reward_enum, dataset_train_ratio, seed, nlp_input_dim 필요.

    Returns
    -------
    (train_inst, test_inst) : tuple[Instruct, Instruct]
    """
    from dataset.multigame import MultiGameDataset

    # eval_games가 지정된 경우 평가 데이터 로딩에 우선 사용 (체크포인트 경로는 game 기준 유지)
    _eval_games_str = getattr(config, 'eval_games', None)
    _load_game = _eval_games_str if _eval_games_str is not None else config.dataset_game

    # eval_dataset_reward_enums 파싱 (로딩 로그·필터링·테이블 표시에서 공통 사용)
    _eval_re_raw = getattr(config, 'eval_dataset_reward_enums', None)
    if _eval_re_raw is not None:
        if isinstance(_eval_re_raw, (str, int)):
            _eval_re_list = [int(c) for c in str(_eval_re_raw)]
        else:
            _eval_re_list = [int(x) for x in _eval_re_raw]
    else:
        _eval_re_list = None

    _effective_re = _eval_re_list if _eval_re_list else (
        [config.dataset_reward_enum] if config.dataset_reward_enum is not None else None
    )

    logger.info(
        f"Loading MultiGameDataset (game={_load_game}, reward_enum={_effective_re})"
        + (f"  [eval_games override: {_eval_games_str}]" if _eval_games_str else "")
    )

    # 'all'이면 전체 게임 로드, 약어면 역매핑으로 full name 리스트 획득
    from conf.game_utils import GAME_ABBR, GAME_ABBR_INV, ALL_GAMES, parse_game_str
    _dg = _load_game
    if _dg == 'all':
        _game_names = ALL_GAMES  # ['dungeon', 'pokemon', 'sokoban', 'doom', 'doom2', 'zelda']
    elif _dg in GAME_ABBR:
        _game_names = GAME_ABBR[_dg]  # 단일 약어 → full name 리스트
    elif len(_dg) % 2 == 0 and all(_dg[i:i+2] in GAME_ABBR for i in range(0, len(_dg), 2)):
        # 복합 약어 (예: "dgpk") → parse_game_str로 파싱
        includes = parse_game_str(_dg)
        _game_names = [name for name in ALL_GAMES if includes.get(f"include_{name}", False)]
    else:
        _game_names = [_dg]  # 이미 full name

    ds = MultiGameDataset(
        include_dungeon=('dungeon' in _game_names),
        include_pokemon=('pokemon' in _game_names),
        include_sokoban=('sokoban' in _game_names),
        include_doom=('doom' in _game_names),
        include_doom2=('doom2' in _game_names),
        include_zelda=('zelda' in _game_names),
        use_tile_mapping=False,
    )

    # 게임별 필터링 ('all'이면 전체 사용)
    if _dg == 'all':
        samples = list(ds)
    else:
        samples = ds.by_games(_game_names)

    # ── 게임별 처리 통계 표 출력 (분할/샘플링 후 최종 정보 포함) ──────────────

    # reward_enum 필터링 — 위에서 파싱한 _eval_re_list / _effective_re 사용
    if _eval_re_list:
        _re_set = set(_eval_re_list)
        samples = [s for s in samples if s.meta.get("reward_enum") in _re_set]
        logger.info(f"eval_dataset_reward_enums={_eval_re_list}: {len(samples)} samples")
    elif config.dataset_reward_enum is not None:
        samples = [s for s in samples if s.meta.get("reward_enum") == config.dataset_reward_enum]

    # reward annotation이 있는 샘플만
    samples = [s for s in samples if "reward_enum" in s.meta and "conditions" in s.meta]

    # invalid instruction 필터 (인코더 학습과 동일 조건)
    def _invalid_instruction(inst) -> bool:
        if inst is None:
            return True
        s = str(inst).strip()
        return s == "" or s.lower() == "none" or s.lower() == "nan"
    samples = [s for s in samples if not _invalid_instruction(s.instruction)]

    # longtail cut (인코더 학습과 동일 조건)
    if getattr(config, "longtail_cut", False):
        from encoder.data.clip_batch import apply_longtail_cut
        n_before = len(samples)
        samples = apply_longtail_cut(samples)
        logger.info(f"Longtail cut: {n_before} → {len(samples)} samples")

    # condition 값 기반 필터링
    cond_filter = getattr(config, "dataset_condition_filter", None)
    if cond_filter:
        filters = _parse_condition_filters(cond_filter)
        before = len(samples)
        samples = _apply_condition_filters(samples, filters)
        logger.info(f"Condition filter '{cond_filter}': {before} → {len(samples)} samples")

    assert len(samples) > 0, (
        f"No samples found for game={_load_game}, "
        f"reward_enum={config.dataset_reward_enum}. "
        f"Check that reward annotations exist."
    )

    # ── eval 모드: (game, re) 그룹별 고정 수 서브샘플링 ──────────────────
    eval_samples_per_group = getattr(config, 'eval_samples_per_group', None)
    sampled_counts: dict = {}  # game → sampled count (테이블용)
    if eval_samples_per_group is not None:
        # eval_seed 가 있으면 우선 사용 (없으면 config.seed fallback)
        _subsample_seed = getattr(config, 'eval_seed', None)
        if _subsample_seed is None:
            _subsample_seed = config.seed
        samples, sampled_counts = _subsample_per_group(
            samples, eval_samples_per_group, seed=_subsample_seed
        )
        logger.info(
            f"[eval_samples_per_group={eval_samples_per_group}, seed={_subsample_seed}]"
            f" subsampled: {len(samples)} samples"
        )

    all_inst = _build_instruct(samples, config)

    _log_dataset_table(ds, samples, config, sampled_counts=sampled_counts,
                       re_filter_list=_effective_re)

    return all_inst, all_inst, samples


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────


def _subsample_per_group(samples, n_per_group: int, seed: int = 0):
    """(game, re) 그룹별로 최대 n_per_group 개를 서브샘플링한다.

    가용 샘플이 n_per_group 보다 적은 그룹은 전부 사용.

    Returns
    -------
    subsampled : list
    sampled_counts : dict  game → sampled count  (re가 1개인 경우 game 기준)
    """
    import random as _random
    from collections import defaultdict

    # (game, re) 그룹핑
    by_group: dict = defaultdict(list)
    for s in samples:
        re = s.meta.get("reward_enum", None)
        by_group[(s.game, re)].append(s)

    # 그룹 내 순서 고정 (결정론적 보장)
    for key in by_group:
        by_group[key].sort(key=lambda s: str(getattr(s, 'source_id', s)))

    result = []
    sampled_counts: dict = {}  # game → count (re 단일 가정, 복수면 합산)

    for (game, re) in sorted(by_group.keys()):
        # 그룹별로 독립 시드 사용 → eval_games가 달라도 같은 게임은 같은 샘플이 뽑힘
        # NOTE: Python built-in hash() is randomized per-process (PYTHONHASHSEED).
        #       hashlib 기반 결정론적 해시 사용.
        _key_bytes = f"{game}_{re}".encode()
        _key_hash = int(hashlib.md5(_key_bytes).hexdigest(), 16) & 0xFFFFFFFF
        group_seed = seed ^ _key_hash
        group_rng = _random.Random(group_seed)
        pool = by_group[(game, re)][:]
        group_rng.shuffle(pool)
        chosen = pool[:n_per_group]
        result.extend(chosen)
        sampled_counts[game] = sampled_counts.get(game, 0) + len(chosen)

    # 최종 결과 셔플도 재현 가능하게 고정 시드 사용
    _random.Random(seed).shuffle(result)
    return result, sampled_counts

def _build_instruct(sample_list, config):
    """샘플 리스트에서 Instruct 객체를 빌드한다."""
    n = len(sample_list)

    reward_i_list = []
    for s in sample_list:
        reward_i_list.append([s.meta["reward_enum"]])
    reward_i = jnp.array(reward_i_list, dtype=jnp.int32)

    condition_list = []
    for s in sample_list:
        conds = s.meta.get("conditions", {})
        row = []
        for i in range(0, 5):
            val = conds.get(i, conds.get(str(i), -1))
            row.append(float(val))
        condition_list.append(row)
    condition = jnp.array(condition_list, dtype=jnp.float32)

    # ── 임베딩 계산 ──────────────────────────────────────────────────────
    if getattr(config, "use_clip", False) and config.nlp_input_dim > 0:
        embedding = _compute_clip_embeddings(sample_list, config)
    elif getattr(config, "use_nlp", False) and config.nlp_input_dim > 0:
        embedding = _compute_bert_embeddings(sample_list, config.nlp_input_dim)
    else:
        embedding = jnp.zeros((n, max(1, config.nlp_input_dim)), dtype=jnp.float32)

    condition_id = jnp.arange(n, dtype=jnp.int32).reshape(-1, 1)

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
    )


def _compute_clip_embeddings(sample_list, config):
    """사전학습된 CLIP encoder를 통해 instruction 텍스트 → latent embedding을 계산한다.

    1) openai/clip-vit-base-patch32 로 토크나이즈
    2) 사전학습된 ContrastiveModule 의 encode_text 를 사용하여
       512-dim raw CLIP → output_dim (e.g. 64) latent space 로 projection

    instruction이 None인 샘플은 zeros 임베딩으로 대체한다.
    """
    import numpy as np
    from glob import glob
    from os.path import basename, join

    nlp_input_dim = config.nlp_input_dim
    encoder_config = config.encoder

    try:
        from transformers import CLIPProcessor
    except ImportError:
        logger.warning("transformers not installed — falling back to zero embeddings")
        return jnp.zeros((len(sample_list), nlp_input_dim), dtype=jnp.float32)

    # ── 1) 토크나이즈 ──────────────────────────────────────────────────────
    model_name = "openai/clip-vit-base-patch32"
    logger.info(f"Computing CLIP latent embeddings (output_dim={encoder_config.output_dim}) "
                f"for {len(sample_list)} samples")

    processor = CLIPProcessor.from_pretrained(model_name)

    texts = []
    has_text = []
    for s in sample_list:
        if s.instruction is not None and len(s.instruction.strip()) > 0:
            texts.append(s.instruction)
            has_text.append(True)
        else:
            texts.append("")  # placeholder
            has_text.append(False)

    inputs = processor(
        text=texts, return_tensors="jax",
        padding="max_length", truncation=True, max_length=encoder_config.token_max_len,
    )
    input_ids = jnp.array(inputs["input_ids"])            # (N, token_max_len)
    attention_mask = jnp.array(inputs["attention_mask"])   # (N, token_max_len)

    # ── 2) 사전학습된 encoder 로드 ────────────────────────────────────────
    from encoder.clip_model import get_cnnclip_encoder, get_cnnclip_decoder_encoder
    from flax.training import checkpoints as flax_ckpts

    # decoder 모드 (MGPCGRL) 인 경우 ContrastiveDecoderModule 사용
    _use_decoder = getattr(config, "use_decoder_reward_shaping", False)
    if _use_decoder:
        from conf.config import DecoderConfig
        decoder_cfg = DecoderConfig(
            num_reward_classes=getattr(config, "decoder_reward_classes", 5)
        )
        module, _ = get_cnnclip_decoder_encoder(
            encoder_config, decoder_config=decoder_cfg, RL_training=True,
        )
    else:
        module, _ = get_cnnclip_encoder(encoder_config, RL_training=True)

    # dummy init
    rng = jax.random.PRNGKey(0)
    dummy_ids = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_pix = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)
    dummy_reward_enum = jnp.zeros((1,), dtype=jnp.int32)
    mode = "text_state" if encoder_config.state else "text"
    # ContrastiveDecoderModule은 reward_enum kwarg를 지원; ContrastiveModule은 무시
    _init_kwargs = dict(mode=mode, training=False)
    if _use_decoder:
        _init_kwargs["reward_enum"] = dummy_reward_enum
    variables = module.init(rng, dummy_ids, dummy_mask, dummy_pix, **_init_kwargs)

    # 체크포인트 복원
    ckpt_path = encoder_config.ckpt_path
    if ckpt_path is not None:
        ckpt_subdirs = glob(join(ckpt_path, '*'))
        ckpt_steps = sorted(
            [int(basename(d)) for d in ckpt_subdirs if basename(d).isdigit()],
            reverse=True,
        )
        if ckpt_steps:
            ckpt_dir = join(ckpt_path, str(ckpt_steps[0]))
            enc_state = flax_ckpts.restore_checkpoint(ckpt_dir, target=None, prefix="")
            if enc_state is not None:
                # train_clip 체크포인트는 TrainState 형태로 저장되므로
                # {"step": ..., "params": {"params": {...}}, "opt_state": ...}
                # module.apply 에는 {"params": {...}} 를 넘겨야 한다.
                if "params" in enc_state and "params" in enc_state["params"]:
                    variables = enc_state["params"]  # {"params": {encoders, temperature, ...}}
                    logger.info(f"Encoder checkpoint loaded (TrainState format) from {ckpt_dir} (step {ckpt_steps[0]})")
                else:
                    variables = enc_state
                    logger.info(f"Encoder checkpoint loaded from {ckpt_dir} (step {ckpt_steps[0]})")
            else:
                logger.warning(f"Checkpoint restore returned None from {ckpt_dir}")
        else:
            logger.warning(f"No checkpoint steps found in {ckpt_path}")
    else:
        logger.warning("encoder.ckpt_path is None — using randomly initialized encoder for text projection")

    # ── 3) encode_text 로 latent embedding 계산 ───────────────────────────
    @jax.jit
    def _encode_text_batch(variables, input_ids, attention_mask):
        return module.apply(
            variables, input_ids, attention_mask, None,
            mode="text", training=False,
        )

    # 배치 단위로 처리 (OOM 방지)
    batch_size = 256
    n = len(sample_list)
    all_embeddings = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = _encode_text_batch(
            variables,
            input_ids[start:end],
            attention_mask[start:end],
        )
        text_embed = np.array(out["text_embed"])  # (batch, output_dim)
        all_embeddings.append(text_embed)

    clip_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, output_dim)

    # instruction 없는 샘플은 zeros
    for i, ht in enumerate(has_text):
        if not ht:
            clip_embeddings[i] = 0.0

    # nlp_input_dim에 맞게 패딩 또는 절삭
    embed_dim = clip_embeddings.shape[1]
    if nlp_input_dim > embed_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - embed_dim))
        clip_embeddings = np.pad(clip_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < embed_dim:
        clip_embeddings = clip_embeddings[:, :nlp_input_dim]

    logger.info(f"CLIP latent embeddings shape: {clip_embeddings.shape}")
    return jnp.array(clip_embeddings, dtype=jnp.float32)


def _compute_bert_embeddings(sample_list, nlp_input_dim):
    """BERT를 사용하여 instruction 텍스트에서 임베딩을 계산한다.

    instruction이 None인 샘플은 zeros 임베딩으로 대체된다.
    """
    import numpy as np

    try:
        from transformers import AutoTokenizer, FlaxAutoModel
    except ImportError:
        logger.warning("transformers not installed — falling back to zero embeddings")
        return jnp.zeros((len(sample_list), nlp_input_dim), dtype=jnp.float32)

    model_name = "bert-base-uncased"  # 768-dim
    logger.info(f"Computing BERT embeddings with {model_name} for {len(sample_list)} samples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModel.from_pretrained(model_name)

    texts = []
    has_text = []
    for s in sample_list:
        if s.instruction is not None and len(s.instruction.strip()) > 0:
            texts.append(s.instruction)
            has_text.append(True)
        else:
            texts.append("")  # placeholder
            has_text.append(False)

    # 배치 단위로 BERT forward → OOM 방지
    from tqdm import tqdm
    bert_batch_size = 64
    all_cls = []
    batches = range(0, len(texts), bert_batch_size)
    for i in tqdm(batches, desc="BERT embeddings", unit="batch"):
        batch_texts = texts[i: i + bert_batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="jax",
            padding=True, truncation=True, max_length=128,
        )
        outputs = model(**tokens)
        all_cls.append(np.array(outputs.last_hidden_state[:, 0, :]))

    cls_embeddings = np.concatenate(all_cls, axis=0)  # (N, 768)

    # instruction 없는 샘플은 zeros
    for i, ht in enumerate(has_text):
        if not ht:
            cls_embeddings[i] = 0.0

    # nlp_input_dim에 맞게 패딩 또는 절삭
    bert_dim = cls_embeddings.shape[1]
    if nlp_input_dim > bert_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - bert_dim))
        cls_embeddings = np.pad(cls_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < bert_dim:
        cls_embeddings = cls_embeddings[:, :nlp_input_dim]

    logger.info(f"BERT embeddings shape: {cls_embeddings.shape}")
    return jnp.array(cls_embeddings, dtype=jnp.float32)


def _log_dataset_table(ds, all_samples, config, *, sampled_counts: dict = None,
                       re_filter_list=None):
    """(game, re) 조합별 처리 통계를 표로 출력한다 (줄마다 별도 logger.info)."""
    import numpy as _np
    from collections import defaultdict

    has_sampled = bool(sampled_counts)
    col_sampled = "Sampled"

    # (game, re) → {n, instr, cond_vals}
    cell: dict = defaultdict(lambda: {"n": 0, "instr": 0, "cond_vals": []})

    for s in ds:
        re = s.meta.get("reward_enum")
        if re is None or "conditions" not in s.meta:
            continue
        key = (s.game, re)
        cell[key]["n"] += 1
        if s.instruction:
            cell[key]["instr"] += 1
        conds = s.meta.get("conditions", {})
        val = conds.get(re, conds.get(str(re), None))
        if val is not None:
            cell[key]["cond_vals"].append(float(val))

    # 캐시 해시 (앞 12자)
    cache_keys: dict = getattr(ds, "_game_cache_keys", {})

    games   = sorted({k[0] for k in cell})
    re_vals = sorted({k[1] for k in cell})
    # re_filter_list 가 있으면 그 목록 기준, 없으면 단일 dataset_reward_enum 기준
    if re_filter_list:
        _re_filter_set = set(re_filter_list)
        re_filter = None  # highlight 판단에 set 사용
    else:
        _re_filter_set = None
        re_filter = config.dataset_reward_enum

    col_game  = "Game"
    col_re    = "re(name)"
    col_hash  = "Hash"
    col_n     = "Samples"
    col_instr = "w/ Instr"
    col_cmin  = "cond_min"
    col_cmax  = "cond_max"

    rows_data = []
    for g in games:
        for re in re_vals:
            if (g, re) not in cell:
                continue
            c = cell[(g, re)]
            re_label = f"{re}({REWARD_ENUM_NAMES.get(re, '?')})"
            if _re_filter_set is not None:
                highlight = re in _re_filter_set
            else:
                highlight = (re_filter is None) or (re == re_filter)
            arr = _np.array(c["cond_vals"]) if c["cond_vals"] else None
            rows_data.append({
                "game":     g,
                "re":       re_label,
                "hash":     cache_keys.get(g, "")[:12],
                "n":        c["n"],
                "instr":    c["instr"],
                "cmin":     f"{arr.min():.1f}" if arr is not None else "-",
                "cmax":     f"{arr.max():.1f}" if arr is not None else "-",
                "selected": highlight,
            })

    selected = [r for r in rows_data if r["selected"]]
    if not selected:
        logger.info("No samples found for re=%s", re_filter)
        return

    w0 = max(len(col_game),  max(len(r["game"]) for r in selected))
    w1 = max(len(col_re),    max(len(r["re"])   for r in selected))
    w2 = max(len(col_hash),  max(len(r["hash"]) for r in selected))
    w3 = max(len(col_n),     max(len(str(r["n"]))     for r in selected))
    w4 = max(len(col_instr), max(len(str(r["instr"])) for r in selected))
    w5 = max(len(col_cmin),  max(len(r["cmin"]) for r in selected))
    w6 = max(len(col_cmax),  max(len(r["cmax"]) for r in selected))
    if has_sampled:
        w7 = max(len(col_sampled), max(len(str(sampled_counts.get(r["game"], "-"))) for r in selected))

    if has_sampled:
        sep = (f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}"
               f"+{'-'*(w3+2)}+{'-'*(w4+2)}+{'-'*(w7+2)}+{'-'*(w5+2)}+{'-'*(w6+2)}+")
        header = (f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
                  f"| {col_n:>{w3}} | {col_instr:>{w4}} "
                  f"| {col_sampled:>{w7}} | {col_cmin:>{w5}} | {col_cmax:>{w6}} |")
    else:
        sep = (f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}"
               f"+{'-'*(w3+2)}+{'-'*(w4+2)}+{'-'*(w5+2)}+{'-'*(w6+2)}+")
        header = (f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
                  f"| {col_n:>{w3}} | {col_instr:>{w4}} "
                  f"| {col_cmin:>{w5}} | {col_cmax:>{w6}} |")

    tot_n     = sum(r["n"]     for r in selected)
    tot_instr = sum(r["instr"] for r in selected)
    tot_sampled = sum(sampled_counts.values()) if has_sampled else None
    if _re_filter_set is not None:
        re_label_str = ",".join(str(r) for r in sorted(_re_filter_set))
        re_name_str  = ",".join(REWARD_ENUM_NAMES.get(r, "?") for r in sorted(_re_filter_set))
    elif re_filter is None:
        re_label_str, re_name_str = "all", "all"
    else:
        re_label_str = str(re_filter)
        re_name_str  = REWARD_ENUM_NAMES.get(re_filter, "?")
    if has_sampled:
        total_row = (f"| {'TOTAL (re='+re_label_str+')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
                     f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {tot_sampled:>{w7}} | {'':{w5}} | {'':{w6}} |")
    else:
        total_row = (f"| {'TOTAL (re='+re_label_str+')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
                     f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {'':{w5}} | {'':{w6}} |")

    logger.info("Dataset Summary  "
                f"(game={config.dataset_game}, re={re_label_str}/{re_name_str}, "
                f"train_ratio={config.dataset_train_ratio})")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    prev_game = None
    for r in rows_data:
        if not r["selected"]:
            continue
        if prev_game and prev_game != r["game"]:
            logger.info(sep)
        if has_sampled:
            sc = sampled_counts.get(r["game"], "-")
            row = (f"| {r['game']:<{w0}} | {r['re']:<{w1}} | {r['hash']:<{w2}} "
                   f"| {r['n']:>{w3}} | {r['instr']:>{w4}} "
                   f"| {sc:>{w7}} | {r['cmin']:>{w5}} | {r['cmax']:>{w6}} |")
        else:
            row = (f"| {r['game']:<{w0}} | {r['re']:<{w1}} | {r['hash']:<{w2}} "
                   f"| {r['n']:>{w3}} | {r['instr']:>{w4}} "
                   f"| {r['cmin']:>{w5}} | {r['cmax']:>{w6}} |")
        logger.info(row)
        prev_game = r["game"]
    logger.info(sep)
    logger.info(total_row)
    logger.info(sep)


def _log_dataset_summary(config, samples):
    """reward_enum 필터 후 최종 샘플 요약 (condition 통계 포함)."""
    import numpy as _np

    re_counter = Counter(s.meta["reward_enum"] for s in samples)
    logger.info("Filtered samples: %d  (reward_enum breakdown below)", len(samples))
    for re_val in sorted(re_counter.keys()):
        re_samples = [s for s in samples if s.meta["reward_enum"] == re_val]
        cond_vals = []
        for s in re_samples:
            conds = s.meta.get("conditions", {})
            val = conds.get(re_val, conds.get(str(re_val), None))
            if val is not None:
                cond_vals.append(float(val))
        if cond_vals:
            arr = _np.array(cond_vals)
            logger.info(
                "  re=%d (%s): %d samples  cond[min=%.1f, max=%.1f, mean=%.2f, std=%.2f]",
                re_val, REWARD_ENUM_NAMES.get(re_val, "?"), len(re_samples),
                arr.min(), arr.max(), arr.mean(), arr.std(),
            )
        else:
            logger.info("  re=%d (%s): %d samples",
                        re_val, REWARD_ENUM_NAMES.get(re_val, "?"), len(re_samples))


def _log_split_summary(train_samples, test_samples, train_inst, *, sampled_counts: dict = None):
    """Train/Test 분할 결과 요약. sampled_counts가 있으면 Sampled 컬럼을 추가한다."""
    train_game = Counter(s.game for s in train_samples)
    test_game  = Counter(s.game for s in test_samples)
    games = sorted(set(train_game) | set(test_game))

    has_sampled = bool(sampled_counts)

    col_game    = "Game"
    col_train   = "Train"
    col_test    = "Test"
    col_sampled = "Sampled"

    w0 = max(len(col_game),  max(len(g) for g in games))
    w1 = max(len(col_train), max(len(str(train_game[g])) for g in games))
    w2 = max(len(col_test),  max(len(str(test_game[g]))  for g in games))
    if has_sampled:
        w3 = max(len(col_sampled), max(len(str(sampled_counts.get(g, 0))) for g in games))

    if has_sampled:
        sep    = f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}+{'-'*(w3+2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} | {col_sampled:>{w3}} |"
        total_sampled = sum(sampled_counts.values())
        total_row = (f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | "
                     f"{len(test_samples):>{w2}} | {total_sampled:>{w3}} |")
    else:
        sep    = f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} |"
        total_row = (f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | "
                     f"{len(test_samples):>{w2}} |")

    logger.debug("Train/Test Split  "
                f"(total=%d, train=%d, test=%d)",
                len(train_samples) + len(test_samples),
                len(train_samples), len(test_samples))
    logger.debug(sep)
    logger.debug(header)
    logger.debug(sep)
    for g in games:
        if has_sampled:
            row = f"| {g:<{w0}} | {train_game[g]:>{w1}} | {test_game[g]:>{w2}} | {sampled_counts.get(g, 0):>{w3}} |"
        else:
            row = f"| {g:<{w0}} | {train_game[g]:>{w1}} | {test_game[g]:>{w2}} |"
        logger.debug(row)
    logger.debug(sep)
    logger.debug(total_row)
    logger.debug(sep)


# ── Condition 필터 유틸 ────────────────────────────────────────────────────────

import re as _re
from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional, List as _List


@_dataclass
class _ConditionFilter:
    """단일 condition 필터 조건."""
    enum_idx: int                   # condition index (0~4)
    min_val: _Optional[float] = None  # inclusive lower bound
    max_val: _Optional[float] = None  # inclusive upper bound


def _parse_condition_filters(filter_str: str) -> _List[_ConditionFilter]:
    """필터 문자열을 파싱한다.

    포맷 (쉼표로 여러 개 구분):
        enum_{i}_min_{lo}_max_{hi}   — lo ≤ condition[i] ≤ hi
        enum_{i}_min_{lo}            — lo ≤ condition[i]
        enum_{i}_max_{hi}            — condition[i] ≤ hi

    Examples
    --------
        "enum_0_min_3_max_10"        → condition[0] in [3, 10]
        "enum_0_min_3_max_10,enum_2_max_50"  → 두 필터 AND
    """
    pattern = _re.compile(
        r"enum_(\d+)"
        r"(?:_min_([\d.]+))?"
        r"(?:_max_([\d.]+))?"
    )
    filters = []
    for token in filter_str.split(","):
        token = token.strip()
        if not token:
            continue
        m = pattern.fullmatch(token)
        if m is None:
            raise ValueError(
                f"Invalid condition filter: '{token}'. "
                f"Expected format: enum_{{i}}_min_{{lo}}_max_{{hi}} "
                f"(min/max are each optional but at least one required)"
            )
        idx = int(m.group(1))
        min_val = float(m.group(2)) if m.group(2) is not None else None
        max_val = float(m.group(3)) if m.group(3) is not None else None
        if min_val is None and max_val is None:
            raise ValueError(
                f"Condition filter 'enum_{idx}' has neither min nor max — "
                f"at least one bound is required."
            )
        filters.append(_ConditionFilter(enum_idx=idx, min_val=min_val, max_val=max_val))
    return filters


def _apply_condition_filters(samples, filters: _List[_ConditionFilter]):
    """필터 리스트를 AND 조합으로 적용하여 샘플을 필터링한다."""
    for f in filters:
        def _keep(s, _f=f):
            conds = s.meta.get("conditions", {})
            val = conds.get(_f.enum_idx, conds.get(str(_f.enum_idx), None))
            if val is None:
                return False
            val = float(val)
            if _f.min_val is not None and val < _f.min_val:
                return False
            if _f.max_val is not None and val > _f.max_val:
                return False
            return True
        samples = [s for s in samples if _keep(s)]
    return samples

