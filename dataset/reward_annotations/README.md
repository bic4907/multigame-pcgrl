# Reward Annotation Pipeline

game level sample in  text reward measure  computetext, OpenAI Batch API to  text instruction  createtext  2text pipeline.

```
cache (MultiGameDataset)
    ↓  annotate.py
{key}.ann.json  (5text measure text, instruction text)
    ↓  generate_instructions.py --run
{key}.ann.json  (instruction_raw / instruction_uni text text)
```

---

## all Usage (--run)

```bash
# Step 1: measure annotation
python dataset/reward_annotations/annotate.py

# Step 2: instruction create (text → finish text → automatic save)
python dataset/reward_annotations/generate_instructions.py --run
```

`--run`  gameby batch  text, finishtext text text, finish text ann.json in  result  applytext.

### text Usage (text game / enum)

```bash
python dataset/reward_annotations/annotate.py --games doom zelda

python dataset/reward_annotations/generate_instructions.py --run \
    --games doom zelda --enums 0 1
```

---

## Step 1 — annotate.py

cache(`dataset/multigame/cache/artifacts/`) in  map array  text 5 text measure  computetext `{key}.ann.json` in  savetext.

### Reward Enum text of

| enum | feature_name | content |
|------|------|------|
| 0 | `region` | text passable text text |
| 1 | `path_length` |  text text path text  |
| 2 | `interactable_count` | Interactive tile text |
| 3 | `hazard_count` | Hazard tile text |
| 4 | `collectable_count` | Collectable tile text |

**passable basis**: unified EMPTY(1) + HAZARD(4) + COLLECTABLE(5) — text game common

### gametext sub_condition (count compute basis tile)

| game | interactable | hazard | collectable |
|------|------|------|------|
| doom | spawn + door + danger | enemy | item |
| zelda | door + block + start | mob | object |
| sokoban | box | — | — |
| pokemon | spawn + water | enemy | object |
| dungeon | — | enemy | treasure |

### text text

| text | default value | text |
|------|------|------|
| `--games` | all 5text | processtext game list |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | cache text directory |
| `--force` | False | existing ann.json overwrite |

### text

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

ann.json  text each sample in  text GPT in text `instruction_raw` / `instruction_uni` create  requesttext result  ann.json in  savetext.

### --run text text

```
1. threshold=None text (game, feature) text → instruction  "None" as  direct text
2. textprocess row in  text gametext JSONL file create (batches/{timestamp}.jsonl)
3. OpenAI Batch API in  gameby batch text
4. text text (default 10seconds text):
     - finishtext batch → result parsing → ann.json update
     - failure/text →  to text text  after  skip
5. text batch finish text text
```

### instruction_raw vs instruction_uni

| text | basis |
|------|------|
| `instruction_raw` | game text tile name text for  (ENEMY, DOOR, SPAWN text) |
| `instruction_uni` | unified text text for  (empty / wall / interactive / hazard / collectable) |

### text text

| text | default value | text |
|------|------|------|
| `--games` | all 5text | processtext game list |
| `--enums` | `0 1 2 3 4` | processtext reward_enum |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | cache text |
| `--force` | False |  text text instruction also  textcreate |
| `--poll-interval` | 10 | text text (seconds) |
| `--limit` | None | processtext maximum row text (text for ) |

### text text Usage

```bash
# JSONL create + batch text (finish text none)
python dataset/reward_annotations/generate_instructions.py --submit

# finishtext batch result text + ann.json update
python dataset/reward_annotations/generate_instructions.py --retrieve BATCH_ID

# batch text check
python dataset/reward_annotations/generate_instructions.py --status BATCH_ID

# batch  text text
python dataset/reward_annotations/generate_instructions.py --log
```

batch text  text  `batches/batch_log.csv` in  writetext.

---

## text text

### system_prompt.txt

GPT in text  before text  text text. text text, text, text condition  text of text.

```
You are a game level description writer for PCGRL.
Write one sentence (≤10 words) describing the level's intensity.
Output JSON: {"instruction_raw": "...", "instruction_uni": "..."}
```

**text text text:**
- text text  text (`STRICT LIMIT: 10 words or fewer`)
- text text (`brief, factual description` / `NOT a design command`)
- text tabletext (text·text text text)

### instruction_config.py

text text create in  text for text  text config text. below text  text text content  text.

#### CUSTOM_THRESHOLDS

feature text  4text intensity level to  splittext  text (3text text → 4 bin).
`None` text GPT call text  `"None"` string to  text.

```python
CUSTOM_THRESHOLDS = {
    "dungeon_region":       [1.5, 4.5, 14.5],  # very few / somewhat few / somewhat many / very many
    "sokoban_hazard_count": None,               # text in  hazard none → text
    ...
}
```

> text text  after  instruction  textcreatetext `--force` text text for .

#### FEATURE_ZONE_LABELS

intensity level 0~3 in  text  text text string. text text of  `Intensity level` text in  tabletext.

```python
FEATURE_ZONE_LABELS = {
    "region": ["very few regions", "somewhat few regions",
               "somewhat many regions", "very many regions"],
    ...
}
```

#### VOCAB_SETS

each intensity levelby GPT in text text  text list. text in  `Suggested vocabulary` text to  text, text requesttext list in  randomtext 1text selecttext.

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

text text top in  text  game text feature text. GPT  text   text  text text for text.

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

## text config

`generate_instructions.py` top in  direct text:

```python
MODEL       = "gpt-5.4-mini"   # text for text OpenAI text
MAX_TOKENS  = 300              # maximum text text
TEMPERATURE = 2.0              # text (text text text tabletext)
```

---

## text text

`.env` file in  config:

```
OPENAI_API_KEY=sk-...
```

---

## file structure

```
dataset/reward_annotations/
├── annotate.py              # Step 1: measure compute → ann.json create
├── generate_instructions.py # Step 2: OpenAI Batch API → instruction text
├── instruction_config.py    # text config text (threshold, vocab, text)
├── system_prompt.txt        # GPT text text
└── batches/
    ├── batch_log.csv        # batch text/finish  text
    └── batch_{timestamp}.jsonl  # gametext batch request file
```
