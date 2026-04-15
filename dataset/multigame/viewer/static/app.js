/* ── DOM 참조 ───────────────────────────────────────────────────────────── */
const gameSelect  = document.getElementById('gameSelect');
const indexInput  = document.getElementById('indexInput');
const prevBtn     = document.getElementById('prevBtn');
const nextBtn     = document.getElementById('nextBtn');
const goBtn       = document.getElementById('goBtn');
const meta        = document.getElementById('meta');
const errorBox    = document.getElementById('error');
const canvas      = document.getElementById('levelCanvas');
const ctx         = canvas.getContext('2d');
const legendEl    = document.getElementById('legend');
const modeTabs    = document.querySelectorAll('.mode-tab');
const mappingPanel = document.getElementById('mapping-panel');
const mappingContent = document.getElementById('mapping-content');
const viewModeSelect = document.getElementById('viewModeSelect');
const albumSizeSelect = document.getElementById('albumSizeSelect');
const reloadBtn = document.getElementById('reloadBtn');
const reloadStatus = document.getElementById('reloadStatus');
const viewRow = document.getElementById('view-row');
const albumWrap = document.getElementById('album-wrap');
const albumGrid = document.getElementById('album-grid');

/* ── 상태 ───────────────────────────────────────────────────────────────── */
let gameCounts  = {};
let currentGame = null;
let currentIndex= 0;
let renderMode  = 'raw-color';   // 'raw-color' | 'raw-image' | 'unified-color' | 'unified-image'
let lastSample  = null;    // 마지막으로 받은 payload 캐시
let currentMapping = null; // 현재 게임 mapping.json 메타
const mappingCache = {};   // game -> mapping payload

let viewMode = 'album';   // 'single' | 'album'
let albumPageSize = 30;
let albumAutoSize = true;
let albumStart = 0;
let isInitialized = false; // init() 완료 여부
const sampleCache = {};    // `${game}:${idx}` -> sample payload
const tileImageCache = new Map(); // url -> HTMLImageElement | null(failed)

function getSelectedGames() {
  if (!gameSelect || !gameSelect.value) return [];
  return [gameSelect.value];
}

function getPrimaryGame() {
  return gameSelect ? gameSelect.value : null;
}

/* ── 유틸 ───────────────────────────────────────────────────────────────── */
function clearError()    { errorBox.textContent = ''; }
function setError(msg)   { errorBox.textContent = msg || ''; }

function normPalette(obj) {
  const m = new Map();
  Object.entries(obj || {}).forEach(([k, v]) => m.set(Number(k), v));
  return m;
}

function clampIndex(game, idx) {
  const n = gameCounts[game] || 0;
  if (n <= 0) return 0;
  if (idx < 0) return n - 1;
  if (idx >= n) return 0;
  return idx;
}

function parseRenderMode(mode) {
  const unified = mode.startsWith('unified-');
  const image = mode.endsWith('-image');
  return { unified, image };
}

function buildTileImageUrl(name) {
  if (!name) return null;
  return `/tile_ims/${encodeURIComponent(name)}`;
}

/**
 * 이미지를 캐시에서 가져오거나 로드를 시작합니다.
 * onReady 콜백을 등록하면 로드 완료 시 호출됩니다.
 * @param {string} url
 * @param {function} [onReady] - 로드 완료 시 콜백
 * @returns {HTMLImageElement|null} 이미 로드된 경우 img 반환, 로딩 중이면 null
 */
function getTileImage(url, onReady) {
  if (!url) return null;

  const cached = tileImageCache.get(url);
  if (cached === null) return null;  // 404 등 실패
  if (cached && cached.complete && cached.naturalWidth > 0) return cached;

  // 로딩 중이거나 처음 요청
  if (!tileImageCache.has(url)) {
    const img = new Image();
    img.onload = () => {
      tileImageCache.set(url, img);
      if (onReady) onReady(img);
    };
    img.onerror = () => {
      tileImageCache.set(url, null);
    };
    tileImageCache.set(url, img);  // 로딩 중 표시
    img.src = url;
  } else {
    // 로딩 중인 img에 onReady 연결
    const img = tileImageCache.get(url);
    if (img && onReady) {
      const prevOnload = img.onload;
      img.onload = (e) => {
        if (prevOnload) prevOnload(e);
        onReady(img);
      };
    }
  }
  return null;
}

/**
 * 특정 게임의 모든 타일 이미지를 미리 로드합니다.
 * @param {object} mappingInfo
 * @param {string} mode
 * @param {function} onAllReady - 모두 로드 완료 시 콜백
 */
function preloadTileImages(mappingInfo, mode, onAllReady) {
  if (!mappingInfo) { if (onAllReady) onAllReady(); return; }
  const cfg = parseRenderMode(mode);
  const imageMap = cfg.unified
    ? (mappingInfo.unified_tile_images || {})
    : (mappingInfo.raw_tile_images || {});

  const urls = Object.values(imageMap).filter(Boolean).map(buildTileImageUrl).filter(Boolean);
  const uniqueUrls = [...new Set(urls)];
  if (uniqueUrls.length === 0) { if (onAllReady) onAllReady(); return; }

  let remaining = 0;
  uniqueUrls.forEach((url) => {
    const cached = tileImageCache.get(url);
    if (cached === null) return; // 실패한 것은 스킵
    if (cached && cached.complete && cached.naturalWidth > 0) return; // 이미 로드됨
    remaining++;
  });

  if (remaining === 0) { if (onAllReady) onAllReady(); return; }

  let done = 0;
  const check = () => {
    done++;
    if (done >= remaining && onAllReady) onAllReady();
  };

  uniqueUrls.forEach((url) => {
    const cached = tileImageCache.get(url);
    if (cached === null) return;
    if (cached && cached.complete && cached.naturalWidth > 0) return;
    getTileImage(url, check);
  });
}

function tileImageNameForId(id, mode, mappingInfo) {
  if (!mappingInfo) return null;
  const key = String(id);
  const { unified } = parseRenderMode(mode);
  if (unified) {
    return (mappingInfo.unified_tile_images || {})[key] || null;
  }
  return (mappingInfo.raw_tile_images || {})[key] || null;
}

/* ── 렌더링 ─────────────────────────────────────────────────────────────── */
const TILE_SIZE_MIN = 4;
const GRID_LINE_COLOR = 'rgba(255,255,255,0.08)';

function calcTile(h, w) {
  return Math.max(TILE_SIZE_MIN, Math.floor(Math.min(720 / h, 720 / w)));
}

/**
 * mode: 'raw' – array + palette
 *       'unified' – unified_array + unified_palette
 *       'symbol'  – unified_array + unified_palette + name text overlay
 */
function drawLevel(sample, mode) {
  const cfg = parseRenderMode(mode);
  const array2d = cfg.unified ? sample.unified_array : sample.array;
  const palette = normPalette(cfg.unified ? sample.unified_palette : sample.palette);

  if (!array2d || !Array.isArray(array2d)) {
    setError(`렌더링 실패: ${mode} 모드 데이터 누락`);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const h = array2d.length;
  const w = h > 0 ? array2d[0].length : 0;
  if (!h || !w) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }

  const tile = calcTile(h, w);
  canvas.width = w * tile;
  canvas.height = h * tile;

  function doDraw() {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const tid = Number(array2d[y][x]);
        const color = palette.get(tid) || [255, 0, 255];

        if (cfg.image) {
          const imageName = tileImageNameForId(tid, mode, currentMapping);
          const url = buildTileImageUrl(imageName);
          const img = getTileImage(url);
          if (img && img.complete && img.naturalWidth > 0) {
            ctx.drawImage(img, x * tile, y * tile, tile, tile);
          } else {
            ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
            ctx.fillRect(x * tile, y * tile, tile, tile);
          }
        } else {
          ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
          ctx.fillRect(x * tile, y * tile, tile, tile);
        }
      }
    }

    ctx.strokeStyle = GRID_LINE_COLOR;
    ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += 1) {
      const lx = x * tile + 0.5;
      ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, h * tile); ctx.stroke();
    }
    for (let y = 0; y <= h; y += 1) {
      const ly = y * tile + 0.5;
      ctx.beginPath(); ctx.moveTo(0, ly); ctx.lineTo(w * tile, ly); ctx.stroke();
    }
  }

  if (cfg.image && currentMapping) {
    preloadTileImages(currentMapping, mode, doDraw);
  } else {
    doDraw();
  }
}

function buildLegend(sample, mode) {
  const cfg = parseRenderMode(mode);
  const palette = normPalette(cfg.unified ? sample.unified_palette : sample.palette);
  const nameMap = cfg.unified ? sample.unified_names : sample.tile_names;
  const array2d = cfg.unified ? sample.unified_array : sample.array;

  if (!array2d || !Array.isArray(array2d)) {
    legendEl.innerHTML = '<h4 style="color:#fda4af">데이터 로드 오류</h4>';
    return;
  }

  const present = new Set(array2d.flat().map(Number));
  legendEl.innerHTML = '';

  const h4 = document.createElement('h4');
  h4.textContent = cfg.unified
    ? (cfg.image ? 'Unified 이미지' : 'Unified 컬러')
    : (cfg.image ? 'Raw 이미지' : 'Raw 컬러');
  legendEl.appendChild(h4);

  [...present].sort((a, b) => a - b).forEach((tid) => {
    const color = palette.get(tid) || [255, 0, 255];
    const label = (nameMap && nameMap[String(tid)]) || String(tid);
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <div class="legend-swatch" style="background:rgb(${color[0]},${color[1]},${color[2]})"></div>
      <span class="legend-label">${label}</span>
      <span class="legend-id">#${tid}</span>`;
    legendEl.appendChild(item);
  });
}

function drawLevelToCanvasElement(sample, mode, targetCanvas, mappingInfo) {
  if (!targetCanvas) return;
  const tctx = targetCanvas.getContext('2d');
  const cfg = parseRenderMode(mode);
  const array2d = cfg.unified ? sample.unified_array : sample.array;
  const palette = normPalette(cfg.unified ? sample.unified_palette : sample.palette);
  const mInfo = mappingInfo || currentMapping;

  if (!Array.isArray(array2d) || !array2d.length || !array2d[0].length) return;

  const h = array2d.length;
  const w = array2d[0].length;
  const tile = Math.max(1, Math.floor(Math.min(48 / h, 48 / w)));
  targetCanvas.width = w * tile;
  targetCanvas.height = h * tile;

  function doDraw() {
    tctx.fillStyle = '#0f172a';
    tctx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);

    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const tid = Number(array2d[y][x]);
        const color = palette.get(tid) || [255, 0, 255];

        if (cfg.image) {
          const imageName = tileImageNameForId(tid, mode, mInfo);
          const url = buildTileImageUrl(imageName);
          const img = getTileImage(url);
          if (img && img.complete && img.naturalWidth > 0) {
            tctx.drawImage(img, x * tile, y * tile, tile, tile);
          } else {
            tctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
            tctx.fillRect(x * tile, y * tile, tile, tile);
          }
        } else {
          tctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
          tctx.fillRect(x * tile, y * tile, tile, tile);
        }
      }
    }
  }

  if (cfg.image && mInfo) {
    preloadTileImages(mInfo, mode, doDraw);
  } else {
    doDraw();
  }
}

function setViewModeUI() {
  const isAlbum = viewMode === 'album';
  if (viewRow) viewRow.style.display = isAlbum ? 'none' : 'flex';
  if (albumWrap) albumWrap.style.display = isAlbum ? 'block' : 'none';
}

function clampAlbumStart(start, selectedGames) {
  const total = selectedGames.reduce((acc, g) => acc + (gameCounts[g] || 0), 0);
  if (total <= 0) return 0;
  const maxStart = Math.max(0, total - albumPageSize);
  if (start < 0) return maxStart;
  if (start > maxStart) return 0;
  return start;
}

function resolveAlbumOffset(selectedGames, offset) {
  let rem = offset;
  for (const g of selectedGames) {
    const n = gameCounts[g] || 0;
    if (rem < n) return { game: g, index: rem };
    rem -= n;
  }
  return null;
}

async function getSampleAndMapping(game, idx) {
  const sample = await fetchSampleCached(game, idx);
  let mappingInfo;
  try {
    mappingInfo = await fetchMapping(game);
  } catch (mappingErr) {
    console.warn('[viewer] mapping fetch failed, using sample fallback:', mappingErr);
    mappingInfo = {
      game,
      mapping: inferMappingFromSample(sample),
      tile_names: sample.tile_names || {},
      unified_names: sample.unified_names || {},
      unified_palette: sample.unified_palette || {},
    };
  }
  return [mergeUnifiedFromMapping(sample, mappingInfo), mappingInfo];
}

function computeAutoAlbumPageSize() {
  if (!albumGrid) return albumPageSize;
  const rect = albumGrid.getBoundingClientRect();
  const gridWidth = Math.max(1, rect.width || window.innerWidth);
  const top = rect.top || 0;
  const gridHeight = Math.max(120, window.innerHeight - top - 24);

  const minItem = 90;
  const gap = 8;
  const cols = Math.max(1, Math.floor((gridWidth + gap) / (minItem + gap)));

  // 카드 높이는 거의 정사각형이므로 폭 기준으로 계산.
  const itemSize = Math.floor((gridWidth - gap * (cols - 1)) / cols);
  const rows = Math.max(1, Math.floor((gridHeight + gap) / (itemSize + gap)));

  return Math.max(1, cols * rows);
}

function refreshAlbumPageSize() {
  if (!albumAutoSize) return;
  albumPageSize = computeAutoAlbumPageSize();
}

async function renderAlbum(startIdx) {
  const selectedGames = getSelectedGames();
  if (!selectedGames.length) return;

  if (albumAutoSize) {
    refreshAlbumPageSize();
  }

  const total = selectedGames.reduce((acc, g) => acc + (gameCounts[g] || 0), 0);
  if (total <= 0) return;

  albumStart = clampAlbumStart(startIdx, selectedGames);
  indexInput.value = String(albumStart);

  const offsets = [];
  for (let i = 0; i < albumPageSize; i += 1) {
    const off = albumStart + i;
    if (off >= total) break;
    offsets.push(off);
  }

  const entries = await Promise.all(offsets.map(async (off) => {
    const resolved = resolveAlbumOffset(selectedGames, off);
    if (!resolved) return null;
    const [sample, mappingInfo] = await getSampleAndMapping(resolved.game, resolved.index);
    return { off, ...resolved, sample, mappingInfo };
  }));

  albumGrid.innerHTML = '';
  entries.filter(Boolean).forEach(({ game, index, sample, mappingInfo }) => {
    const card = document.createElement('div');
    card.className = 'album-item';

    const wrap = document.createElement('div');
    wrap.className = 'album-canvas-wrap';

    const c = document.createElement('canvas');
    c.className = 'album-canvas';
    drawLevelToCanvasElement(sample, renderMode, c, mappingInfo);

    const p = document.createElement('p');
    p.className = 'album-label';
    p.textContent = `${game} #${index}`;

    wrap.appendChild(c);
    card.appendChild(wrap);
    card.appendChild(p);
    card.title = `${game} | ${sample.source_id}`;

    card.addEventListener('click', async () => {
      viewMode = 'single';
      if (viewModeSelect) viewModeSelect.value = 'single';

      // 클릭한 게임을 싱글 보기 기본 선택으로 만든다.
      Array.from(gameSelect.options).forEach((o) => {
        o.selected = (o.value === game);
      });

      setViewModeUI();
      await loadAndRender(game, index);
    });
    albumGrid.appendChild(card);

    if (!currentMapping && mappingInfo) currentMapping = mappingInfo;
  });

  const sizeLabel = albumAutoSize ? `auto(${entries.filter(Boolean).length})` : String(entries.filter(Boolean).length);
  meta.textContent = `games: ${selectedGames.join(', ')}\nalbum: ${albumStart} ~ ${albumStart + (entries.filter(Boolean).length - 1)} / ${total - 1}\nsize: ${sizeLabel}`;
}

/* ── 렌더 전체 파이프라인 ───────────────────────────────────────────────── */
function renderCurrent() {
  if (viewMode === 'album') {
    renderAlbum(albumStart).catch((err) => setError(String(err.message || err)));
    return;
  }

  if (!lastSample) return;
  drawLevel(lastSample, renderMode);
  buildLegend(lastSample, renderMode);

  if (mappingPanel) {
    if (renderMode.startsWith('raw-') || !currentMapping) {
      mappingPanel.style.display = 'none';
    } else {
      renderMappingPanel(lastSample, currentMapping);
    }
  }
}

function renderMeta(sample) {
  const lines = [
    `game: ${sample.game}`,
    `index: ${sample.index} / ${sample.count - 1}`,
    `source_id: ${sample.source_id}`,
    `shape: ${sample.shape[0]} x ${sample.shape[1]}`,
  ];
  meta.textContent = lines.join('\n');
  renderAnnotations(sample.annotations || [], renderMode);
}

const REWARD_SYMBOLS = ['RG', 'PL', 'IC', 'HC', 'CC'];

function renderAnnotations(annotations, mode) {
  const el = document.getElementById('annotations');
  if (!annotations || annotations.length === 0) {
    el.innerHTML = '';
    return;
  }

  const useRaw = !mode || mode.startsWith('raw-');

  const rows = annotations.map(ann => {
    const instr = useRaw ? ann.instruction_raw : ann.instruction_uni;
    if (!instr) return '';
    const sym = REWARD_SYMBOLS[ann.reward_enum] || `#${ann.reward_enum}`;
    const condStr = ann.condition != null ? Number(ann.condition).toFixed(1) : '';
    return `<div class="ann-row">
      <span class="ann-enum">${sym}</span>
      ${condStr ? `<span class="ann-cond">${escHtml(condStr)}</span>` : ''}
      <span class="ann-instr">${escHtml(instr)}</span>
    </div>`;
  }).filter(Boolean);

  el.innerHTML = rows.join('');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function loadAndRender(game, idx) {
  clearError();

  if (viewMode === 'album') {
    currentGame = getPrimaryGame();
    await renderAlbum(idx);
    return;
  }

  const safeIdx = clampIndex(game, idx);
  try {
    const [sample, mappingInfo] = await getSampleAndMapping(game, safeIdx);

    currentGame = game;
    currentIndex = sample.index;
    indexInput.value = String(sample.index);
    currentMapping = mappingInfo;
    lastSample = sample;

    renderMeta(lastSample);
    renderCurrent();
  } catch (err) {
    setError(String(err.message || err));
  }
}

/* ── 이벤트 바인딩 ─────────────────────────────────────────────────────── */
function bindControls() {
  gameSelect.addEventListener('change', () => {
    if (viewMode === 'album') {
      albumStart = 0;
      loadAndRender(getPrimaryGame(), albumStart);
    } else {
      const g = getPrimaryGame();
      if (g) loadAndRender(g, 0);
    }
  });

  prevBtn.addEventListener('click', () => {
    if (viewMode === 'album') {
      loadAndRender(getPrimaryGame(), albumStart - albumPageSize);
    } else if (currentGame) {
      loadAndRender(currentGame, currentIndex - 1);
    }
  });

  nextBtn.addEventListener('click', () => {
    if (viewMode === 'album') {
      loadAndRender(getPrimaryGame(), albumStart + albumPageSize);
    } else if (currentGame) {
      loadAndRender(currentGame, currentIndex + 1);
    }
  });

  goBtn.addEventListener('click', () => {
    const idx = Number(indexInput.value || '0');
    if (viewMode === 'album') {
      loadAndRender(getPrimaryGame(), Number.isFinite(idx) ? idx : 0);
    } else if (currentGame) {
      loadAndRender(currentGame, Number.isFinite(idx) ? idx : 0);
    }
  });

  if (viewModeSelect) {
    viewModeSelect.addEventListener('change', () => {
      viewMode = viewModeSelect.value === 'album' ? 'album' : 'single';
      setViewModeUI();
      const g = getPrimaryGame();
      if (!g) return;
      if (viewMode === 'album') {
        albumStart = 0;
        loadAndRender(g, albumStart);
      } else {
        loadAndRender(g, currentIndex || 0);
      }
    });
  }

  if (albumSizeSelect) {
    albumSizeSelect.addEventListener('change', () => {
      const raw = (albumSizeSelect.value || 'auto').toLowerCase();
      albumAutoSize = (raw === 'auto');
      if (albumAutoSize) {
        refreshAlbumPageSize();
      } else {
        albumPageSize = Math.max(1, Number(raw || '30'));
      }
      if (viewMode === 'album') {
        loadAndRender(getPrimaryGame(), albumStart);
      }
    });
  }

  window.addEventListener('resize', () => {
    if (viewMode === 'album' && albumAutoSize) {
      loadAndRender(getPrimaryGame(), albumStart);
    }
  });

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      if (viewMode === 'album') {
        loadAndRender(getPrimaryGame(), albumStart - albumPageSize);
      } else if (currentGame) {
        loadAndRender(currentGame, currentIndex - 1);
      }
    }
    if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      if (viewMode === 'album') {
        loadAndRender(getPrimaryGame(), albumStart + albumPageSize);
      } else if (currentGame) {
        loadAndRender(currentGame, currentIndex + 1);
      }
    }
  });

  modeTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      modeTabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      renderMode = tab.dataset.mode;
      renderCurrent();
      if (lastSample) {
        renderAnnotations(lastSample.annotations || [], renderMode);
      }
    });
  });

  if (reloadBtn) {
    reloadBtn.addEventListener('click', async () => {
      reloadBtn.disabled = true;
      if (reloadStatus) reloadStatus.textContent = '⏳ 로딩 중...';
      try {
        const res = await fetch('/api/reload', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
          // 캐시 초기화
          Object.keys(sampleCache).forEach((k) => delete sampleCache[k]);
          Object.keys(mappingCache).forEach((k) => delete mappingCache[k]);
          tileImageCache.clear();
          currentMapping = null;
          lastSample = null;

          // 게임 목록 갱신
          gameSelect.innerHTML = '';
          gameCounts = {};
          (data.games || []).forEach((row, i) => {
            gameCounts[row.game] = row.count;
            const opt = document.createElement('option');
            opt.value = row.game;
            opt.textContent = `${row.game} (${row.count.toLocaleString()})`;
            opt.selected = i === 0;
            gameSelect.appendChild(opt);
          });

          if (reloadStatus) {
            reloadStatus.textContent = `✅ ${data.elapsed_sec}s — ${(data.games || []).map((r) => `${r.game}(${r.count})`).join(', ')}`;
          }

          // 첫 번째 게임 첫 샘플로 이동
          const g = getPrimaryGame();
          if (g) {
            albumStart = 0;
            await loadAndRender(g, 0);
          }
        } else {
          if (reloadStatus) reloadStatus.textContent = `❌ ${data.error || '알 수 없는 오류'}`;
        }
      } catch (err) {
        if (reloadStatus) reloadStatus.textContent = `❌ ${err.message}`;
      } finally {
        reloadBtn.disabled = false;
      }
    });
  }
}

/* ── 초기화 ─────────────────────────────────────────────────────────────── */
async function init() {
  bindControls();
  setViewModeUI();

  try {
    const games = await fetchGames();
    if (!games.length) {
      setError('로드 가능한 데이터셋이 없습니다. 경로를 확인해주세요.');
      return;
    }

    gameSelect.innerHTML = '';
    gameCounts = {};
    games.forEach((row, i) => {
      gameCounts[row.game] = row.count;
      const opt = document.createElement('option');
      opt.value = row.game;
      opt.textContent = `${row.game} (${row.count.toLocaleString()})`;
      opt.selected = i === 0;
      gameSelect.appendChild(opt);
    });

    // albumAutoSize일 때는 일단 기본값으로, renderAlbum에서 다시 계산됨
    if (!albumAutoSize && albumSizeSelect) {
      albumPageSize = Math.max(1, Number(albumSizeSelect.value || '30'));
    }

    const g = getPrimaryGame();
    if (g) {
      currentGame = g;
      // 초기 로드 전: albumAutoSize일 때 올바른 크기 계산
      if (albumAutoSize) {
        albumPageSize = computeAutoAlbumPageSize();
      }
      // 초기 로드: album 모드로 빠르게 시작
      viewMode = 'album';
      if (viewModeSelect) viewModeSelect.value = 'album';
      await loadAndRender(g, 0);
    }
    isInitialized = true;
  } catch (err) {
    setError(String(err.message || err));
  }
}


/* ── API 호출 ───────────────────────────────────────────────────────────── */
async function fetchJsonOrThrow(url, fallbackLabel) {
  const res = await fetch(url);
  const contentType = (res.headers.get('content-type') || '').toLowerCase();
  const bodyText = await res.text();

  let data = null;
  if (contentType.includes('application/json')) {
    try {
      data = JSON.parse(bodyText);
    } catch {
      throw new Error(`${fallbackLabel}: JSON 파싱 실패 (${url})`);
    }
  } else {
    const preview = bodyText.slice(0, 80).replace(/\s+/g, ' ');
    throw new Error(`${fallbackLabel}: JSON 응답이 아님 (${res.status}) ${preview}`);
  }

  if (!res.ok) {
    throw new Error((data && data.error) || `${fallbackLabel}: ${res.status}`);
  }
  return data;
}

async function fetchGames() {
  const data = await fetchJsonOrThrow('/api/games', '게임 목록 요청 실패');
  return data.games || [];
}

async function fetchSample(game, idx) {
  return fetchJsonOrThrow(
    `/api/sample?game=${encodeURIComponent(game)}&index=${idx}`,
    '샘플 요청 실패',
  );
}

async function fetchSampleCached(game, idx) {
  const key = `${game}:${idx}`;
  if (sampleCache[key]) return sampleCache[key];
  const sample = await fetchSample(game, idx);
  sampleCache[key] = sample;
  return sample;
}

async function fetchMapping(game) {
  if (mappingCache[game]) return mappingCache[game];
  const data = await fetchJsonOrThrow(
    `/api/mapping?game=${encodeURIComponent(game)}`,
    '매핑 요청 실패',
  );
  mappingCache[game] = data;
  return data;
}

function buildUnifiedFromRaw(array2d, mapping) {
  if (!Array.isArray(array2d)) return [];
  return array2d.map((row) => row.map((v) => {
    const key = String(Number(v));
    if (Object.prototype.hasOwnProperty.call(mapping || {}, key)) {
      return Number(mapping[key]);
    }
    return 0;
  }));
}

function inferMappingFromSample(sample) {
  const out = {};
  const raw = sample && sample.array;
  const uni = sample && sample.unified_array;
  if (!Array.isArray(raw) || !Array.isArray(uni) || raw.length !== uni.length) {
    return out;
  }
  for (let r = 0; r < raw.length; r += 1) {
    const rawRow = raw[r] || [];
    const uniRow = uni[r] || [];
    const w = Math.min(rawRow.length, uniRow.length);
    for (let c = 0; c < w; c += 1) {
      const rk = String(Number(rawRow[c]));
      const uv = Number(uniRow[c]);
      if (!Object.prototype.hasOwnProperty.call(out, rk)) {
        out[rk] = uv;
      }
    }
  }
  return out;
}

function mergeUnifiedFromMapping(sample, mappingInfo) {
  if (!sample || !mappingInfo) return sample;

  if (!Array.isArray(sample.unified_array)) {
    sample.unified_array = buildUnifiedFromRaw(sample.array, mappingInfo.mapping || {});
  }
  if (!sample.unified_palette) {
    sample.unified_palette = mappingInfo.unified_palette || {};
  }
  if (!sample.unified_names) {
    sample.unified_names = mappingInfo.unified_names || {};
  }
  if (!sample.tile_names) {
    sample.tile_names = mappingInfo.tile_names || {};
  }
  return sample;
}

function renderMappingPanel(sample, mappingInfo) {
  if (!mappingPanel || !mappingContent || !mappingInfo) return;

  const mapping = mappingInfo.mapping || {};
  const tileNames = mappingInfo.tile_names || {};
  const unifiedNames = mappingInfo.unified_names || {};
  const unifiedPalette = normPalette(mappingInfo.unified_palette || {});

  const presentRaw = new Set((sample.array || []).flat().map((v) => String(Number(v))));
  const keys = Object.keys(mapping).filter((k) => presentRaw.has(k)).sort((a, b) => Number(a) - Number(b));

  mappingContent.innerHTML = '';
  if (keys.length === 0) {
    mappingPanel.style.display = 'none';
    return;
  }
  mappingPanel.style.display = 'block';

  keys.forEach((rawKey) => {
    const uKey = String(mapping[rawKey]);
    const uColor = unifiedPalette.get(Number(uKey)) || [255, 0, 255];
    const rawName = tileNames[rawKey] || rawKey;
    const uniName = unifiedNames[uKey] || uKey;

    const row = document.createElement('div');
    row.className = 'mapping-row';
    row.innerHTML = `
      <div class="mapping-src">
        <div class="legend-swatch" style="background:#334155"></div>
        <span class="mapping-label">${rawName}</span>
        <span class="mapping-id">#${rawKey}</span>
      </div>
      <span class="mapping-arrow">→</span>
      <div class="mapping-dst">
        <div class="legend-swatch" style="background:rgb(${uColor[0]},${uColor[1]},${uColor[2]})"></div>
        <span class="mapping-label">${uniName}</span>
        <span class="mapping-id">#${uKey}</span>
      </div>`;
    mappingContent.appendChild(row);
  });
}

init();

