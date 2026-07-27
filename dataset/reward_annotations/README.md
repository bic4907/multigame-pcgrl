# Reward Annotation Pipeline

A two-stage pipeline: compute reward measures over game level samples, then generate natural-language instructions through the OpenAI Batch API.

```
cache (MultiGameDataset)
    {key}.ann.json  (5 measures computed, instructions still empty)
{key}.ann.json  (5 measures computed, instructions still empty)
    ↓  generate_instructions.py --run
{key}.ann.json  (instruction_raw / instruction_uni filled in)
```

---

## all Usage (--run)

```bash
# Step 1: measure annotation
python dataset/reward_annotations/annotate.py

# Step 2: generate instructions (submit → poll → save automatically)
python dataset/reward_annotations/generate_instructions.py --run
```

`--run` submits one batch per game, polls until completion, and writes the results back into ann.json.

### Restricting the run (specific games / enums)

```bash
python dataset/reward_annotations/annotate.py --games doom zelda

python dataset/reward_annotations/generate_instructions.py --run \
    --games doom zelda --enums 0 1
```

---

## Step 1 — annotate.py

Reads the map arrays from the cache (`dataset/multigame/cache/artifacts/`), computes the 5 measures, and stores them in `{key}.ann.json`.

### Reward enum definitions

| enum | feature_name | content |
|------|------|------|
| 0 | `region` | number of connected passable regions |
| 1 | `path_length` | length of the longest path |
| 2 | `interactable_count` | number of Interactive tiles |
| 3 | `hazard_count` | number of Hazard tiles |
| 4 | `collectable_count` | number of Collectable tiles |

**Passable definition**: unified EMPTY(1) + HAZARD(4) + COLLECTABLE(5) — shared by every game

### Per-game sub_condition (tiles counted for each measure)

| game | interactable | hazard | collectable |
|------|------|------|------|
| doom | spawn + door + danger | enemy | item |
| zelda | door + block + start | mob | object |
| sokoban | box | — | — |
| pokemon | spawn + water | enemy | object |
| dungeon | — | enemy | treasure |

### Options

| option | default | description |
|------|------|------|
| `--games` | all 5 | games to process |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | cache root directory |
| `--force` | False | overwrite an existing ann.json |

### Output

`dataset/multigame/cache/artifacts/{hash}/{game}/{key}.ann.json`

```json
{
  "game": "doom",
  "n_samples": 1000,
  "has_instructions": false,
  "annotations": [
    {
      "key": "dm000000",
      "source_id": "Doom1_map01_000",
      "reward_enum": 0,
      "feature_name": "region",
      "condition_0": 3.0,
      "instruction_raw": null,
      "instruction_uni": null
    }
  ]
}
```

---

## Step 2 — generate_instructions.py

For every sample in ann.json, asks GPT to write `instruction_raw` / `instruction_uni` and writes the results back into ann.json.

### What `--run` does

```
1. (game, feature) pairs whose threshold is None get the literal instruction "None"
2. The remaining rows are written to a per-game JSONL file (batches/{timestamp}.jsonl)
3. One OpenAI batch is submitted per game
4. Polling loop (every 10 seconds by default):
     - completed batch → result parsing → ann.json update
     - failed or cancelled → logged and skipped
5. Ends once every batch has finished
```

### instruction_raw vs instruction_uni

| field | derived from |
|------|------|
| `instruction_raw` | game-specific tile names (ENEMY, DOOR, SPAWN, ...) |
| `instruction_uni` | unified category names (empty / wall / interactive / hazard / collectable) |

### Options

| option | default | description |
|------|------|------|
| `--games` | all 5 | games to process |
| `--enums` | `0 1 2 3 4` | reward_enums to process |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | cache root |
| `--force` | False | regenerate instructions that already exist |
| `--poll-interval` | 10 | polling interval in seconds |
| `--limit` | None | maximum rows to process (for testing) |

### Common invocations

```bash
# Build the JSONL and submit the batch (do not wait)
python dataset/reward_annotations/generate_instructions.py --submit

# Retrieve a finished batch and update ann.json
python dataset/reward_annotations/generate_instructions.py --retrieve BATCH_ID

# Check batch status
python dataset/reward_annotations/generate_instructions.py --status BATCH_ID

# Cancel a batch
python dataset/reward_annotations/generate_instructions.py --log
```

Batch submissions and completions are logged to `batches/batch_log.csv`.

---

## Prompt configuration

### system_prompt.txt

The system prompt handed to GPT. It fixes the output format, the length limit and how the condition is described.

```
You are a game level description writer for PCGRL.
Write one sentence (≤10 words) describing the level's intensity.
Output JSON: {"instruction_raw": "...", "instruction_uni": "..."}
```

**Key constraints:**
- Hard length limit (`STRICT LIMIT: 10 words or fewer`)
- Descriptive, not imperative (`brief, factual description` / `NOT a design command`)
- Fixed output schema (the two instruction fields)

### instruction_config.py

Configuration for instruction generation. Editing the entries below changes the generated text.

#### CUSTOM_THRESHOLDS

Thresholds splitting a feature into 4 intensity levels (3 cut points → 4 bins).
`None` skips the GPT call and stores the literal string `"None"`.

```python
CUSTOM_THRESHOLDS = {
    "dungeon_region":       [1.5, 4.5, 14.5],  # very few / somewhat few / somewhat many / very many
    "sokoban_hazard_count": None,               # sokoban has no hazards → skipped
    ...
}
```

> After changing a threshold, pass `--force` to regenerate the affected instructions.

#### FEATURE_ZONE_LABELS

Label strings for intensity levels 0-3. They are inserted into the prompt's `Intensity level` field.

```python
FEATURE_ZONE_LABELS = {
    "region": ["very few regions", "somewhat few regions",
               "somewhat many regions", "very many regions"],
    ...
}
```

#### VOCAB_SETS

Candidate words offered to GPT per intensity level. They appear in the prompt's `Suggested vocabulary` field, and one is drawn at random per request.

```python
VOCAB_SETS = {
    "region": [
        ["few", "sparse", "marginal"],           # level 0 — very few
        ["some", "moderate", "slight"],           # level 1 — somewhat few
        ["several", "multiple", "partitioned"],   # level 2 — somewhat many
        ["fragmented", "numerous", "many"],       # level 3 — very many
    ],
    ...
}
```

#### GAME_DESCRIPTIONS / FEATURE_DESCRIPTIONS

The game and feature names shown at the top of the prompt, giving GPT the context it needs.

```python
GAME_DESCRIPTIONS = {
    "doom": "Doom (top-down view of a first-person shooter dungeon map)",
    ...
}

FEATURE_DESCRIPTIONS = {
    "region": "number of disconnected passable-area clusters — count of separate walkable zones",
    ...
}
```

---

## Model settings

Defined directly at the top of `generate_instructions.py`:

```python
MODEL       = "gpt-5.4-mini"   # OpenAI model used
MAX_TOKENS  = 300              # maximum output tokens
TEMPERATURE = 2.0              # sampling temperature (higher = more varied)
```

---

## Directory layout

`.env` file in  config:

```
OPENAI_API_KEY=sk-...
```

---

## file structure

```
dataset/reward_annotations/
├── annotate.py              # Step 1: measure compute → ann.json create
├── generate_instructions.py # Step 2: OpenAI Batch API → instruction generation
├── instruction_config.py    # prompt configuration (thresholds, vocab, labels)
├── system_prompt.txt        # GPT system prompt
└── batches/
    ├── batch_log.csv        # batch submission / completion log
    └── batch_{timestamp}.jsonl  # per-game batch request file
```
