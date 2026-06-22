"""
train_adapter.py
================
인코더 체크포인트 기반 few-shot adapter 사전 학습.

adapter를 학습하고 {saves_dir}/{encoder_ckpt_name}_adp-{type}/ 에 새 encoder 형식으로 저장한다.
  - ckpts/           → base encoder ckpts 절대경로 symlink
  - norm_stats.json  → base encoder에서 복사
  - dataset_setting.json → adapter 정보 + unseen_support_source_ids 포함
  - adapter_state.pkl    → adapter 파라미터 (few_shot_adapter.py 형식)

이후 RL 학습에서는 encoder.ckpt_name={adapter_ckpt_name} 으로 바로 사용 가능하다.

실행:
    python train_adapter.py \\
        encoder.ckpt_dir=/mnt/nas/mgpcgrl/mgpcgrl_adapter \\
        encoder.ckpt_name=clipdec-game-all_unseen-zd_ur-0.0_exp-def_0 \\
        decoder.adapter_type=film \\
        reward_unseen_ratio=0.03 \\
        saves_dir=/mnt/nas/mgpcgrl/mgpcgrl_adapter
"""
import hashlib
import json
import logging
import os
import random
import re
import shutil
from collections import defaultdict

import hydra
import numpy as np
import wandb

from conf.config import MGPCGRLConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
from instruct_rl.utils.log_utils import suppress_jax_debug_logs

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()


@hydra.main(version_base=None, config_path="./conf", config_name="train_mgpcgrl")
def main(config: MGPCGRLConfig):
    # ── ur + adp_ur + unseen_game → encoder.ckpt_name / reward_unseen_ratio 자동 구성 ──
    _ur          = getattr(config, "ur", None)
    _adp_ur      = getattr(config, "adp_ur", None)
    _unseen_game = getattr(config, "unseen_game", None)
    if _ur is not None and _adp_ur is not None and _unseen_game:
        _ur_f   = float(_ur)
        _adp_f  = float(_adp_ur)
        # 유효 조합: ur=0.0 은 adp_ur 자유, ur>0 이면 반드시 ur==adp_ur
        if _ur_f > 1e-6 and abs(_ur_f - _adp_f) > 1e-6:
            logger.info(
                "Skipping invalid combo ur=%.4f adp_ur=%.4f (enc_ur>0 requires enc_ur==adp_ur)",
                _ur_f, _adp_f,
            )
            return
        _ur_str = str(_ur_f)  # "0.0", "0.03", "0.1" — 파일명과 정확히 일치
        config.encoder.ckpt_name = (
            f"clipdec-game-all_unseen-{_unseen_game}_ur-{_ur_str}_exp-def_0"
        )
        config.reward_unseen_ratio = _adp_f
        logger.info(
            "ur=%s adp_ur=%s unseen_game=%s → ckpt_name=%s  adp_ur=%.4f",
            _ur, _adp_ur, _unseen_game, config.encoder.ckpt_name, config.reward_unseen_ratio,
        )

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("encoder.ckpt_dir and encoder.ckpt_name must be set.")
    _at = getattr(config.decoder, "adapter_type", None)
    if not _at or _at.lower() == "none":
        raise ValueError("decoder.adapter_type must be film/lora/moe for adapter training.")

    adp_ur = config.reward_unseen_ratio
    if adp_ur <= 0.0:
        raise ValueError("reward_unseen_ratio must be > 0 for adapter training.")

    _m = re.search(r"_ur-([\d.]+)_", config.encoder.ckpt_name)
    enc_ur = float(_m.group(1)) if _m else 0.0

    saves_dir = getattr(config, "saves_dir", None) or config.encoder.ckpt_dir
    # adp_ur을 이름에 포함하여 같은 enc_ur에 대한 다른 adp_ur 구분
    _aur_str = f"{adp_ur:.2f}".rstrip("0").rstrip(".")
    adapter_ckpt_name = f"{config.encoder.ckpt_name}_aur-{_aur_str}_adp-{_at}"
    adapter_dir = os.path.join(saves_dir, adapter_ckpt_name)
    adapter_state_path = os.path.join(adapter_dir, "adapter_state.pkl")

    if os.path.exists(adapter_state_path) and not getattr(config, "overwrite", False):
        logger.info("Adapter already exists, skipping: %s", adapter_state_path)
        return

    # ── encoder dataset_setting.json → seen/unseen 정보 읽기 ─────────────────
    dataset_setting_path = os.path.join(
        config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json"
    )
    dataset_setting: dict = {}
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path) as f:
            dataset_setting = json.load(f)
        seen_games = dataset_setting.get("seen_games", [])
        if seen_games:
            config.reward_seen_games = list(seen_games)
        seen_abbrs = dict.fromkeys(
            GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV
        )
        config.game = "all" if seen_abbrs.keys() == GAME_ABBR.keys() else "".join(seen_abbrs)
    else:
        logger.warning("dataset_setting.json not found: %s", dataset_setting_path)

    # adapter는 모든 enum 포함, 전체 샘플 로드
    config.dataset_reward_enum = "all"
    config.dataset_game = "all"
    config.dataset_seen_ratio = 1.0
    config.dataset_unseen_ratio = 1.0

    # ── init_config: encoder.ckpt_path 설정 ──────────────────────────────────
    from instruct_rl.utils.path_utils import init_config
    config = init_config(config)
    # init_config가 dataset_game="all"을 config.game(seen games만)으로 덮어쓰므로
    # unseen 게임 포함 전체 샘플 로드를 위해 복원한다.
    config.dataset_game = "all"

    # ── W&B 초기화 ────────────────────────────────────────────────────────────
    from instruct_rl.utils.env_loader import get_wandb_key
    wandb_key = get_wandb_key()
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project=getattr(config, "wandb_project", None) or "mgpcgrl_adapt",
            name=f"{_at}-enc{enc_ur:.2f}-adp{adp_ur:.2f}-{config.encoder.ckpt_name[-20:]}",
            group=f"train_adapter_{_at}",
            config={
                "adapter_type":        _at,
                "adapter_steps":       config.decoder.adapter_steps,
                "adapter_rank":        config.decoder.adapter_rank,
                "adapter_num_experts": config.decoder.adapter_num_experts,
                "adapter_lr":          config.decoder.adapter_lr,
                "enc_unseen_ratio":    enc_ur,
                "adp_unseen_ratio":    adp_ur,
                "encoder_ckpt":        config.encoder.ckpt_name,
            },
        )

    logger.info(
        "=== train_adapter: type=%s  enc_ur=%.4f  adp_ur=%.4f  encoder=%s ===",
        _at, enc_ur, adp_ur, config.encoder.ckpt_name,
    )

    # ── 1. 샘플 + 임베딩 로드 (adapter 비활성, decoder 스킵) ─────────────────
    orig_adapter = config.decoder.adapter_type
    orig_rdm = getattr(config, "reward_decoder_mode", "unseen")
    config.decoder.adapter_type = None
    config.reward_decoder_mode = "noop"

    from instruct_rl.utils.dataset_loader import load_dataset_instruct
    all_inst, _, samples = load_dataset_instruct(config)

    config.decoder.adapter_type = orig_adapter
    config.reward_decoder_mode = orig_rdm

    import jax.numpy as jnp
    embeddings = np.asarray(all_inst.embedding, dtype=np.float32)  # (N, D)
    logger.info("Loaded %d samples, embedding shape=%s", len(samples), embeddings.shape)

    # ── 2. support / query 분할 (adp_ur 기반 결정론적 셔플) ──────────────────
    seen_games = set(config.reward_seen_games)
    if "doom" in seen_games or "doom2" in seen_games:
        seen_games.update({"doom", "doom2"})

    unseen_game_indices: dict = defaultdict(list)
    for i, s in enumerate(samples):
        if s.game not in seen_games:
            unseen_game_indices[s.game].append(i)

    support_indices, query_indices = [], []
    unseen_support_source_ids: dict = {}

    for game in sorted(unseen_game_indices.keys()):
        indices = unseen_game_indices[game]
        _seed = int(hashlib.md5(f"{game}_{adp_ur:.6f}".encode()).hexdigest(), 16) % (2 ** 31)
        shuffled = list(indices)
        random.Random(_seed).shuffle(shuffled)
        n_support = max(1, int(len(indices) * adp_ur)) if adp_ur < 1.0 else len(indices)
        game_support = shuffled[:n_support]
        game_query = shuffled[n_support:]
        support_indices.extend(game_support)
        query_indices.extend(game_query)
        unseen_support_source_ids[game] = [samples[i].source_id for i in game_support]
        logger.info(
            "game=%s  total=%d  support=%d  query=%d",
            game, len(indices), len(game_support), len(game_query),
        )

    if not support_indices:
        logger.error("No support samples found. Check reward_seen_games and adp_ur.")
        return

    # ── 3. support GT 레이블 ──────────────────────────────────────────────────
    num_classes = config.decoder.num_reward_classes
    sup_reward_i = np.array([samples[i].meta["reward_enum"] for i in support_indices], dtype=np.int32)
    sup_condition = np.full((len(support_indices), num_classes), -1.0, dtype=np.float32)
    for local_i, orig_i in enumerate(support_indices):
        conds = samples[orig_i].meta.get("conditions", {})
        for j in range(num_classes):
            sup_condition[local_i, j] = float(conds.get(j, conds.get(str(j), -1)))

    query_arr = (
        jnp.array(embeddings[query_indices]) if query_indices
        else jnp.array(embeddings[support_indices[:1]])  # fallback (발생 시 경고)
    )
    if not query_indices:
        logger.warning("No query samples (adp_ur=1.0?); adapter applied to support set itself.")

    # ── 4. encoder module/variables 로드 ─────────────────────────────────────
    from instruct_rl.utils.dataset_loader_helpers.embeddings import _load_shared_clip_module_and_ckpt
    shared_module, shared_variables = _load_shared_clip_module_and_ckpt(config)

    # ── 5. adapter 학습 ───────────────────────────────────────────────────────
    os.makedirs(adapter_dir, exist_ok=True)
    from encoder.few_shot_adapter import adapt_embeddings
    adapt_embeddings(
        query=query_arr,
        support=jnp.array(embeddings[support_indices]),
        support_reward_i=jnp.array(sup_reward_i),
        support_condition=jnp.array(sup_condition),
        apply_fn=shared_module.apply,
        variables=shared_variables,
        adapter_type=_at,
        rank=config.decoder.adapter_rank,
        num_experts=config.decoder.adapter_num_experts,
        lr=config.decoder.adapter_lr,
        steps=config.decoder.adapter_steps,
        save_path=adapter_state_path,
        load_path=None,  # 항상 새로 학습
    )

    # ── 6. 새 encoder 형식 생성 ───────────────────────────────────────────────
    base_enc_dir = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name)

    # ckpts symlink (base encoder 가중치 재사용)
    ckpts_link = os.path.join(adapter_dir, "ckpts")
    if not os.path.lexists(ckpts_link):
        base_ckpts = os.path.abspath(os.path.join(base_enc_dir, "ckpts"))
        os.symlink(base_ckpts, ckpts_link)

    # norm_stats.json 복사
    norm_src = os.path.join(base_enc_dir, "norm_stats.json")
    norm_dst = os.path.join(adapter_dir, "norm_stats.json")
    if os.path.exists(norm_src) and not os.path.exists(norm_dst):
        shutil.copy(norm_src, norm_dst)

    # dataset_setting.json (adapter 정보 반영)
    adapter_setting = dict(dataset_setting)
    adapter_setting["unseen_ratio"] = adp_ur  # adapter의 unseen ratio (RL 학습에서 reward_unseen_ratio로 주입)
    adapter_setting["unseen_support_source_ids"] = unseen_support_source_ids
    adapter_setting["adapter_type"] = _at
    adapter_setting["base_encoder_ckpt_name"] = config.encoder.ckpt_name
    with open(os.path.join(adapter_dir, "dataset_setting.json"), "w") as f:
        json.dump(adapter_setting, f, indent=2, ensure_ascii=False)

    logger.info("=== Adapter encoder saved to %s ===", adapter_dir)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
