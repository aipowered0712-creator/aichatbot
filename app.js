const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const ui = {
  sidebar: $('#sidebar'), overlay: $('#overlay'), welcome: $('#welcome'),
  thread: $('#chatThread'), input: $('#promptInput'), composer: $('#composer'),
  knowledge: $('#knowledgeToggle'), conversation: $('#conversation'),
  send: $('#sendBtn'), toast: $('#toast'), sourceDialog: $('#sourceDialog'),
  modelSelect: $('#modelSelect'), modelMenu: $('#modelMenu'), documentsDialog: $('#documentsDialog'),
  devDialog: $('#devDialog')
};

const models = {
  local: { label: '本地 · Kimi-VL-A3B', name: 'Kimi-VL-A3B-Instruct', id: 'Kimi-VL-A3B-Instruct', provider: 'local', icon: '#i-cpu' },
  qwen: { label: 'HKPC GPU · Qwen3.8-27B', name: 'Qwen3.8-27B', id: 'public/qwen3.8-27b', provider: 'online', icon: '#i-cloud' },
  deepseek: { label: 'HKPC GPU · DeepSeek-V4-Flash', name: 'DeepSeek-V4-Flash-W8A8-MTP', id: 'public/deepseek-v4-flash-w8a8-mtp', provider: 'online', icon: '#i-cloud' },
  minimax: { label: 'HKPC GPU · MiniMax-M3', name: 'MiniMax-M3', id: 'public/minimax-m3', provider: 'online', icon: '#i-cloud' },
  qwenmax: { label: '商用 · Qwen3.8-Max-0902', name: 'Qwen3.8-Max-0902', id: 'public/qwen3.8-max-0902', provider: 'online', icon: '#i-cloud' }
};

const sidebarExamples = [
  {
    title: '查一項參數',
    hint: '按原文回答',
    prompt: '請根據目前知識庫，找出最相關的章節，說明一項關鍵參數應如何設定。保留原文的數值、單位和適用條件，並標明出處。文檔沒有寫明的，請直接說明無法確認。'
  },
  {
    title: '比較兩項差異',
    hint: '只比較文檔寫到的內容',
    prompt: '請比較目前知識庫中兩個相關對象，在關鍵參數、適用條件和注意事項上有何差異。每項都標明出處。文檔沒有寫到的，不要說成相同或不同。'
  },
  {
    title: '整理附錄表格',
    hint: '抽出對照數據',
    prompt: '請把目前知識庫裡的表格或附錄整理成對照，保留原文的數值、單位和限制條件，並標明出處。'
  }
];

const handbookExamples = [
  {
    icon: '問',
    iconClass: 's1',
    cardTitle: '文檔問答',
    cardText: '這項參數怎樣設定？',
    prompt: '請根據目前知識庫，找出最相關的章節，說明一項關鍵參數應如何設定。保留原文的數值、單位和適用條件，並標明出處。文檔沒有寫明的，請直接說明無法確認。'
  },
  {
    icon: '比',
    iconClass: 's2',
    cardTitle: '差異對比',
    cardText: '兩個對象有何不同？',
    prompt: '請比較目前知識庫中兩個相關對象，在關鍵參數、適用條件和注意事項上有何差異。每項都標明出處。文檔沒有寫到的，不要說成相同或不同。'
  },
  {
    icon: '提',
    iconClass: 's3',
    cardTitle: '參數提取',
    cardText: '把表格整理成對照',
    prompt: '請把目前知識庫裡的表格或附錄整理成對照，保留原文的數值、單位和限制條件，並標明出處。'
  },
  {
    icon: '摘',
    iconClass: 's4',
    cardTitle: '章節摘要',
    cardText: '總結這一章的要點',
    prompt: '請總結目前知識庫中最相關章節的要點，包括適用範圍、關鍵步驟和注意事項，並標明出處。不要補充文檔沒有寫的內容。'
  }
];

const handbookChips = [
  { label: '查關鍵條件', prompt: '請根據目前知識庫，列出一項關鍵條件的數值、單位和適用範圍，並標明出處。' },
  { label: '事前要先處理嗎', prompt: '請根據目前知識庫，說明進行下一步之前是否需要先處理。若需要，寫出原文中的條件、限度和做法，並標明出處。' },
  { label: '整理附錄對照', prompt: '請把目前知識庫附錄或對照表中的數據整理出來，保留原文數值和單位，並標明出處。' }
];
function readSavedModel() {
  try {
    const key = localStorage.getItem('knowledge-rag-model');
    if (key === 'online') return 'qwen';
    return models[key] ? key : null;
  } catch { return null; }
}
const savedModel = readSavedModel();
const HISTORY_STORAGE_KEY = 'knowledge-rag-conversations-v1';
const MAX_SAVED_CONVERSATIONS = 20;
const state = {
  generating: false, runId: 0, lastPrompt: '', toastTimer: null, indexTimer: null,
  defaultedModel: false, scope: '全部知識空間', useKnowledge: true, libraries: [], uploadLibrary: 'moldpdf', activeLibrary: '',
  selectedModel: savedModel || 'qwen', controller: null,
  conversations: readSavedConversations(), activeConversationId: null,
  indexStatus: 'checking', localError: '',
  debugRetrieval: localStorage.getItem('knowledge-rag-debug') === '1'
};
const ragState = { synced: false, dirty: false };
const isApple = /Mac|iPhone|iPad/.test(navigator.platform || '');
const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
const escapeHtml = (value) => value.replace(/[&<>'"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));

function setNav(open) {
  ui.sidebar.classList.toggle('open', open);
  ui.overlay.classList.toggle('show', open);
  $('#openNav').setAttribute('aria-expanded', String(open));
  if (open) $('#closeNav').focus();
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.add('show');
  state.toastTimer = window.setTimeout(() => ui.toast.classList.remove('show'), 2400);
}

function readSavedConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    return Array.isArray(saved) ? saved.filter((item) => (
      item && typeof item.id === 'string' && typeof item.title === 'string' && Array.isArray(item.messages)
    )).slice(0, MAX_SAVED_CONVERSATIONS) : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.conversations));
  } catch {
    showToast('瀏覽器儲存空間不足，無法保存歷史對話。');
  }
}

function activeConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeConversationId) || null;
}

function createConversation(firstPrompt) {
  const conversation = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: firstPrompt.slice(0, 28),
    updatedAt: Date.now(),
    messages: []
  };
  state.conversations.unshift(conversation);
  state.activeConversationId = conversation.id;
  return conversation;
}

function addHistoryMessage(message) {
  const conversation = activeConversation() || createConversation(message.content || '新對話');
  conversation.messages.push(message);
  conversation.updatedAt = Date.now();
  if (message.role === 'user' && conversation.messages.length === 1) {
    conversation.title = message.content.slice(0, 28);
  }
  state.conversations.sort((left, right) => right.updatedAt - left.updatedAt);
  state.conversations = state.conversations.slice(0, MAX_SAVED_CONVERSATIONS);
  saveConversations();
  renderConversationHistory();
}

function renderConversationHistory() {
  const history = $('#historyList');
  if (!history) return;
  if (!state.conversations.length) {
    history.innerHTML = '<div class="history-empty">暫無已保存的對話</div>';
    return;
  }
  history.innerHTML = state.conversations.map((conversation) => {
    const savedAt = new Date(conversation.updatedAt).toLocaleString('zh-HK', {
      month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    return `<div class="history-row"><button type="button" class="history-item${conversation.id === state.activeConversationId ? ' selected' : ''}" data-conversation-id="${escapeHtml(conversation.id)}"><span>${escapeHtml(conversation.title)}</span><small>${escapeHtml(savedAt)} · ${conversation.messages.length} 條訊息</small></button><button type="button" class="history-delete" data-delete-conversation="${escapeHtml(conversation.id)}" aria-label="刪除對話 ${escapeHtml(conversation.title)}"><svg><use href="#i-trash" /></svg></button></div>`;
  }).join('');
}

function deleteConversation(id) {
  const conversation = state.conversations.find((item) => item.id === id);
  if (!conversation) return;
  if (state.generating && state.activeConversationId === id) {
    showToast('請先停止生成，再刪除這則對話');
    return;
  }
  if (!window.confirm(`確定刪除對話「${conversation.title}」嗎？此操作無法復原。`)) return;
  state.conversations = state.conversations.filter((item) => item.id !== id);
  saveConversations();
  if (state.activeConversationId === id) {
    state.activeConversationId = null;
    resetChat();
  } else {
    renderConversationHistory();
  }
  showToast('已刪除對話');
}

function loadConversation(id) {
  const conversation = state.conversations.find((item) => item.id === id);
  if (!conversation || state.generating) return;
  state.activeConversationId = id;
  ui.thread.replaceChildren();
  ui.welcome.hidden = true;
  ui.thread.hidden = false;
  let latestPrompt = '';
  conversation.messages.forEach((message) => {
    if (message.role === 'user') {
      latestPrompt = message.content;
      ui.thread.insertAdjacentHTML('beforeend', `<article class="message user"><div class="bubble">${escapeHtml(message.content)}</div></article>`);
      return;
    }
    const container = document.createElement('article');
    container.className = 'message assistant';
    container.innerHTML = '<div class="bot-avatar"><svg><use href="#i-spark" /></svg></div><div class="answer"></div>';
    ui.thread.appendChild(container);
    const model = models[message.modelKey] || models.qwen;
    if (message.status === 'error') {
      renderApiError(container, latestPrompt, message.content, model, false);
    } else {
      renderApiAnswer(container, latestPrompt, {
        answer: message.content,
        sources: message.sources || []
      }, model, false);
    }
  });
  renderConversationHistory();
  scrollBottom(true);
  setNav(false);
}

function stopGeneration(message = '已停止生成') {
  if (!state.generating) return;
  state.runId += 1;
  state.generating = false;
  state.controller?.abort();
  state.controller = null;
  setSendState(false);
  const pending = $('.message.pending', ui.thread);
  if (pending) pending.remove();
  showToast(message);
}

function setSendState(generating) {
  state.generating = generating;
  ui.send.classList.toggle('stop', generating);
  ui.send.setAttribute('aria-label', generating ? '停止生成' : '發送');
  ui.send.innerHTML = generating ? '<span class="stop-square"></span>' : '<svg><use href="#i-send" /></svg>';
}

function setRetrievalState(status, text) {
  state.indexStatus = status;
  const indicator = $('#retrievalState');
  if (!indicator) return;
  indicator.className = `retrieval-state ${status}`;
  indicator.textContent = text;
}

function formatElapsed(seconds) {
  const value = Number(seconds) || 0;
  if (value < 60) return `${value}秒`;
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return rest ? `${minutes}分${rest}秒` : `${minutes}分鐘`;
}

function updateIndexBanner(visible, progress = {}) {
  const banner = $('#indexBanner');
  if (!banner) return;
  banner.hidden = !visible;
  if (!visible) return;
  const phaseMap = { scanning: '掃描文檔', parsing: '解析文檔', chunking: '切塊', embedding: '向量化' };
  const phase = phaseMap[progress.phase] || '處理知識庫';
  const fileName = String(progress.current_file || '').split(/[/\\]/).pop();
  const counts = progress.total ? `${progress.done || 0}/${progress.total}` : '';
  $('#indexBannerTitle').textContent = progress.message || `正在${phase}`;
  $('#indexBannerDetail').textContent = fileName
    ? `${phase} ${counts} · ${fileName}。期間可繼續提問或切換模型。`
    : `${phase}${counts ? ` ${counts}` : ''}。期間可繼續提問、切換模型或管理文檔。`;
  $('#indexBannerTime').textContent = formatElapsed(progress.elapsed_seconds);
  const fill = $('#indexBannerFill');
  if (fill) fill.style.width = `${Math.max(8, Math.min(100, progress.percent || 0))}%`;
}

$('#openNav').addEventListener('click', () => setNav(true));
$('#closeNav').addEventListener('click', () => setNav(false));
ui.overlay.addEventListener('click', () => setNav(false));

function closestEl(event, selector) {
  if (event.target?.closest) {
    const match = event.target.closest(selector);
    if (match) return match;
  }
  const path = typeof event.composedPath === 'function' ? event.composedPath() : [];
  return path.find((node) => node instanceof Element && node.matches(selector)) || null;
}

function placeModelMenu() {
  const rect = ui.modelSelect.getBoundingClientRect();
  ui.modelMenu.style.top = `${Math.round(rect.bottom + 8)}px`;
  ui.modelMenu.style.right = `${Math.max(12, Math.round(window.innerWidth - rect.right))}px`;
}

function setModelMenuOpen(open) {
  ui.modelMenu.hidden = !open;
  ui.modelSelect.setAttribute('aria-expanded', String(open));
  if (open) {
    if (ui.modelMenu.parentElement !== document.body) document.body.appendChild(ui.modelMenu);
    placeModelMenu();
  }
}

ui.modelSelect.addEventListener('click', (event) => {
  event.preventDefault();
  event.stopPropagation();
  setModelMenuOpen(ui.modelMenu.hidden);
});
window.addEventListener('resize', () => {
  if (!ui.modelMenu.hidden) placeModelMenu();
});

ui.modelMenu.addEventListener('click', (event) => {
  event.stopPropagation();
  const button = closestEl(event, '[data-model]');
  if (button?.dataset.model) selectModel(button.dataset.model);
});

function selectModel(key, silent = false) {
  const model = models[key];
  if (!model) return;
  const changed = state.selectedModel !== key;
  state.selectedModel = key;
  try { localStorage.setItem('knowledge-rag-model', key); } catch { /* 浏览器禁用存储时仍可正常切换 */ }
  $$('#modelMenu [data-model]').forEach((button) => button.classList.toggle('selected', button.dataset.model === key));
  if ($('#modelLabel')) $('#modelLabel').textContent = model.label;
  const icon = $('#modelIcon use');
  if (icon) {
    icon.setAttribute('href', model.icon);
    icon.setAttribute('xlink:href', model.icon);
  }
  const isLocal = model.provider === 'local';
  if ($('#activeModelName')) $('#activeModelName').textContent = model.name;
  if ($('#modelNotice')) {
    if (isLocal && state.localError) {
      $('#modelNotice').innerHTML = `<svg><use href="#i-lock" /></svg> 本地模型載入失敗：${escapeHtml(String(state.localError).slice(0, 160))}`;
    } else {
      $('#modelNotice').innerHTML = isLocal
        ? '<svg><use href="#i-lock" /></svg> 本地模式下，檢索資料與問題均不會發送到雲端。'
        : '<svg><use href="#i-cloud" /></svg> HKPC GPU Platform 雲端模式會把問題和召回的文檔片段發送到已配置的 API。';
    }
  }
  if ($('#modelDisclaimer')) {
    $('#modelDisclaimer').textContent = isLocal
      ? '目前選擇本地模型；回答由 AI 生成，請透過引用原文覆核重要資訊'
      : '目前選擇 HKPC GPU Platform 雲端模型；問題與檢索片段將發送至雲端 API';
  }
  if (key === 'local') {
    fetch('/api/local/warmup', { method: 'POST' }).catch(() => {});
  }
  if (!silent) {
    setModelMenuOpen(false);
    if (changed) showToast(`已切換至${model.label}`);
  }
}

async function checkServiceHealth() {
  if (window.location.protocol === 'file:') {
    selectModel(state.selectedModel, true);
    return;
  }
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    $('#serviceDot').className = 'status-dot';
    $('#serviceStatus').textContent = '模型服務已連接';
    const local = health.models?.local || {};
    state.localError = local.error || '';
    $('#localModelStatus').textContent = local.loading
      ? '載入中'
      : local.loaded
        ? '已載入'
        : local.error
          ? '失敗'
          : local.ready
            ? '可用'
            : '需配置';
    $$('.online-status').forEach((el) => {
      el.textContent = health.models?.online?.ready ? '可用' : '需配置';
    });
    syncOnlineModels(health.models?.online?.options || []);
    state.defaultedModel = true;
    const documentCount = health.knowledge_base?.document_count ?? 0;
    const indexReady = Boolean(health.knowledge_base?.index_ready);
    const indexStatus = health.knowledge_base?.index_status || 'idle';
    const progress = health.knowledge_base?.index_progress || {};
    state.libraries = health.knowledge_base?.libraries || state.libraries;
    state.activeLibrary = health.knowledge_base?.active_library || '';
    renderScopeMenu(state.libraries);
    renderLibraryChips(state.libraries);
    $('#navDocumentCount').textContent = documentCount;
    if (!documentCount) {
      setRetrievalState('empty', '無文檔');
      updateIndexBanner(false);
    } else if (indexReady) {
      setRetrievalState('ready', '已啟用');
      updateIndexBanner(false);
    } else if (indexStatus === 'error') {
      setRetrievalState('error', '索引失敗');
      updateIndexBanner(false, progress);
    } else {
      const detail = progress.total
        ? `${progress.phase === 'embedding' ? '向量化' : '切塊'} ${progress.done}/${progress.total}`
        : '準備中';
      setRetrievalState('building', detail);
      updateIndexBanner(true, progress);
    }
    window.clearTimeout(state.indexTimer);
    if ((documentCount && !indexReady) || local.loading) {
      state.indexTimer = window.setTimeout(checkServiceHealth, 1000);
    }
  } catch {
    $('#serviceDot').className = 'status-dot demo';
    $('#serviceStatus').textContent = '請執行 main.py';
    setRetrievalState('error', '服務離線');
  }
  selectModel(state.selectedModel, true);
}

ui.knowledge.addEventListener('click', () => {
  const menu = $('#scopeMenu');
  const open = menu.hidden;
  menu.hidden = !open;
  ui.knowledge.setAttribute('aria-expanded', String(open));
});

function scopeLabel(scope) {
  if (scope === '不使用知識庫' || scope === '不使用知识库') return '不使用知識庫';
  if (!scope || scope === '全部知識空間' || scope === '全部知识空间') return '全部知識空間';
  const found = state.libraries.find((item) => item.id === scope || item.name === scope);
  return found ? found.name : scope;
}

function renderScopeMenu(libraries = []) {
  const menu = $('#scopeMenu');
  if (!menu) return;
  const items = [
    { scope: '全部知識空間', title: '全部文檔', hint: '檢索所有知識庫' },
    ...libraries.map((item) => ({
      scope: item.id,
      title: item.name,
      hint: item.document_count ? `${item.document_count} 份文檔` : '尚無文檔'
    })),
    { scope: '不使用知識庫', title: '不使用知識庫', hint: '僅使用模型回答' }
  ];
  menu.innerHTML = items.map((item) => (
    `<button data-scope="${escapeHtml(item.scope)}" class="${state.scope === item.scope ? 'selected' : ''}"><span>${escapeHtml(item.title)}</span><small>${escapeHtml(item.hint)}</small></button>`
  )).join('');
  $('.scope-label', ui.knowledge).textContent = scopeLabel(state.scope);
}

function renderLibraryChips(libraries = []) {
  const board = $('#libraryChips');
  if (!board) return;
  if (!libraries.some((item) => item.id === state.uploadLibrary) && libraries.length) {
    state.uploadLibrary = libraries[0].id;
  }
  board.innerHTML = libraries.map((item) => {
    const count = item.document_count ? `${item.document_count} 份文檔` : '尚無文檔';
    const selected = item.id === state.uploadLibrary ? ' selected' : '';
    return `<div class="library-chip${selected}"><button type="button" class="library-chip-main" data-library="${escapeHtml(item.id)}"><strong>${escapeHtml(item.name)}</strong><small>${count}</small></button><button type="button" class="library-chip-delete" data-delete-library-id="${escapeHtml(item.id)}" aria-label="刪除類型 ${escapeHtml(item.name)}"><svg><use href="#i-trash" /></svg></button></div>`;
  }).join('') || '<div class="empty-documents">尚未建立知識庫類型</div>';
  const title = $('#documentListTitle');
  if (title) title.textContent = state.uploadLibrary ? `${scopeLabel(state.uploadLibrary)}的文檔` : '已載入文檔';
}

$('#scopeMenu').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  state.scope = button.dataset.scope;
  state.useKnowledge = state.scope !== '不使用知識庫';
  if (state.useKnowledge && state.scope !== '全部知識空間') state.uploadLibrary = state.scope;
  renderScopeMenu(state.libraries);
  renderLibraryChips(state.libraries);
  ui.knowledge.classList.toggle('active', state.useKnowledge);
  $('#scopeMenu').hidden = true;
  ui.knowledge.setAttribute('aria-expanded', 'false');
  showToast(state.useKnowledge ? `檢索範圍：${scopeLabel(state.scope)}` : '本次對話將不檢索知識庫');
});

ui.input.addEventListener('input', () => {
  ui.input.style.height = 'auto';
  ui.input.style.height = `${Math.min(ui.input.scrollHeight, 120)}px`;
});

ui.input.addEventListener('keydown', (event) => {
  if (event.isComposing) return;
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    ui.composer.requestSubmit();
  }
});

const knownModelKeys = {
  'public/qwen3.8-27b': 'qwen',
  'public/deepseek-v4-flash-w8a8-mtp': 'deepseek',
  'public/minimax-m3': 'minimax',
  'public/qwen3.8-max-0902': 'qwenmax'
};

function syncOnlineModels(items = []) {
  const signature = items.map((item) => `${item.id}|${item.label}`).join('\n');
  if (syncOnlineModels.signature === signature) return;
  syncOnlineModels.signature = signature;
  Object.keys(models).forEach((key) => {
    if (key !== 'local') delete models[key];
  });
  items.forEach((item) => {
    const key = knownModelKeys[item.id] || `m-${item.id.replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48)}`;
    models[key] = { label: item.label, name: item.label, id: item.id, provider: 'online', icon: '#i-cloud' };
  });
  const menu = $('#modelMenu');
  $$('[data-model]', menu).forEach((button) => {
    if (button.dataset.model !== 'local') button.remove();
  });
  const notice = $('#modelNotice');
  Object.entries(models).forEach(([key, model]) => {
    if (model.provider !== 'online') return;
    const button = document.createElement('button');
    button.dataset.model = key;
    button.innerHTML = `<i class="model-option-icon online"><svg><use href="#i-cloud" /></svg></i><span><strong>${escapeHtml(model.name)}</strong><small>${escapeHtml(model.id)}</small></span><em class="online-status">聯網</em>`;
    menu.insertBefore(button, notice);
  });
  const fallback = models[state.selectedModel] ? state.selectedModel : (Object.keys(models).find((key) => key !== 'local') || 'local');
  selectModel(fallback, true);
}

function resetModelEditor() {
  $('#devModelLabel').value = '';
  $('#devModelId').value = '';
  $('#devModelKey').value = '';
  $('#devModelKey').placeholder = '此模型的 API Key，留空則沿用共用 Key';
  $('#devModelUrl').value = '';
  $('#devModelOriginal').value = '';
  $('#devModelSubmit').textContent = '新增模型';
  $('#devModelCancel').hidden = true;
}

function renderDevModels(data) {
  syncOnlineModels(data.models || []);
  const box = $('#devModels');
  if (!box) return;
  box.innerHTML = (data.models || []).map((item) => `
    <div class="dev-model-row${item.id === data.model ? ' current' : ''}">
      <div><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.id)} · ${item.api_key_set ? '已設定專用 Key' : '使用共用 Key'} · ${escapeHtml(item.base_url || 'HKPC 共用位址')}</small></div>
      <button type="button" data-use-model="${escapeHtml(item.id)}">${item.id === data.model ? '預設' : '設為預設'}</button>
      <button type="button" data-edit-model="${escapeHtml(item.id)}" data-key-set="${item.api_key_set ? '1' : '0'}" data-base-url="${escapeHtml(item.base_url || '')}">修改</button>
      <button type="button" data-delete-model="${escapeHtml(item.id)}">刪除</button>
    </div>
  `).join('');
}

function clampRag(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, Math.round(number)));
}

function setRagPair(numberId, rangeId, value) {
  const number = $(numberId);
  const range = $(rangeId);
  if (number) number.value = String(value);
  if (range) range.value = String(value);
}

function readRagInputs() {
  const chunkSize = clampRag($('#devChunkSize')?.value, 200, 4000, 800);
  const overlapCap = Math.min(2000, Math.floor(chunkSize / 2));
  const overlap = clampRag($('#devChunkOverlap')?.value, 0, overlapCap, Math.min(120, overlapCap));
  const topK = clampRag($('#devTopK')?.value, 1, 20, 4);
  const overlapInput = $('#devChunkOverlap');
  const overlapRange = $('#devChunkOverlapRange');
  if (overlapInput) overlapInput.max = String(overlapCap);
  if (overlapRange) overlapRange.max = String(overlapCap);
  return { chunk_size: chunkSize, chunk_overlap: overlap, top_k: topK };
}

function fillDevFields(data) {
  $('#devBaseUrl').value = data.base_url || '';
  $('#devApiKey').value = '';
  $('#devKeyHint').textContent = data.api_key_set ? '已設定，留空則不更改' : '尚未設定';
  $('#devVectorBackend').value = data.vector_backend || '';
  $('#devEmbeddingModel').value = data.embedding_model || '';
  $('#devRerankerModel').value = data.reranker_model || '';
  setRagPair('devChunkSize', 'devChunkSizeRange', data.chunk_size ?? 800);
  setRagPair('devChunkOverlap', 'devChunkOverlapRange', data.chunk_overlap ?? 120);
  setRagPair('devTopK', 'devTopKRange', data.top_k ?? 4);
  readRagInputs();
  ragState.dirty = false;
  ragState.synced = true;
  renderDevModels(data);
}

function bindRagPair(numberId, rangeId) {
  const number = $(numberId);
  const range = $(rangeId);
  if (!number || !range || number.dataset.boundRag === '1') return;
  number.dataset.boundRag = '1';
  const publish = () => {
    ragState.dirty = true;
    const params = readRagInputs();
    setRagPair('devChunkSize', 'devChunkSizeRange', params.chunk_size);
    setRagPair('devChunkOverlap', 'devChunkOverlapRange', params.chunk_overlap);
    setRagPair('devTopK', 'devTopKRange', params.top_k);
  };
  range.addEventListener('input', publish);
  number.addEventListener('change', publish);
}

function loadCustomPrompt() {
  const box = $('#devCustomPrompt');
  if (!box) return;
  box.value = localStorage.getItem('knowledge-rag-custom-prompt') || '';
}

async function loadDevSettings() {
  const response = await fetch('/api/dev-settings');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '無法讀取設定');
  fillDevFields(data);
  return data;
}

async function openDevSettings() {
  const dialog = ui.devDialog;
  if (!dialog || dialog.open) return;
  dialog.showModal();
  resetModelEditor();
  loadCustomPrompt();
  const debug = $('#devDebug');
  if (debug) debug.checked = state.debugRetrieval;
  try {
    await loadDevSettings();
  } catch (error) {
    showToast(error.message);
  }
}

document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    openDevSettings();
  }
  if (event.key === 'Escape') {
    setNav(false);
    setModelMenuOpen(false);
    if (ui.sourceDialog.open) ui.sourceDialog.close();
    if (ui.documentsDialog.open) ui.documentsDialog.close();
    if (ui.devDialog?.open) ui.devDialog.close();
  }
});

document.addEventListener('click', (event) => {
  if (!closestEl(event, '.knowledge-control')) {
    $('#scopeMenu').hidden = true;
    ui.knowledge.setAttribute('aria-expanded', 'false');
  }
  if (!closestEl(event, '.model-control') && !closestEl(event, '#modelMenu')) {
    setModelMenuOpen(false);
  }
});

function bindPromptButtons(root = document) {
  $$('[data-prompt]', root).forEach((button) => {
    if (button.dataset.boundPrompt === '1') return;
    button.dataset.boundPrompt = '1';
    button.addEventListener('click', () => {
      ui.input.value = button.dataset.prompt;
      setNav(false);
      submitPrompt();
    });
  });
}

function renderHandbookExamples() {
  const suggestions = $('#suggestionList');
  if (suggestions) {
    suggestions.innerHTML = handbookExamples.map((item) => (
      `<button data-prompt="${escapeHtml(item.prompt)}"><i class="${item.iconClass}">${item.icon}</i><span><strong>${escapeHtml(item.cardTitle)}</strong>${escapeHtml(item.cardText)}</span><svg><use href="#i-chevron" /></svg></button>`
    )).join('');
  }
  const chips = $('#chipList');
  if (chips) {
    chips.innerHTML = handbookChips.map((item) => (
      `<button data-prompt="${escapeHtml(item.prompt)}">${escapeHtml(item.label)}</button>`
    )).join('');
  }
  bindPromptButtons();
  renderConversationHistory();
}

bindPromptButtons();

$$('[data-toast]').forEach((button) => button.addEventListener('click', () => showToast(button.dataset.toast)));
$$('[data-open-documents]').forEach((button) => button.addEventListener('click', openDocuments));
$('#newChat').addEventListener('click', resetChat);
$('#historyList').addEventListener('click', (event) => {
  const remove = event.target.closest('[data-delete-conversation]');
  if (remove) {
    deleteConversation(remove.dataset.deleteConversation);
    return;
  }
  const button = event.target.closest('[data-conversation-id]');
  if (button) loadConversation(button.dataset.conversationId);
});
$('#clearHistory').addEventListener('click', () => {
  if (!state.conversations.length || !window.confirm('確定清除本瀏覽器保存的所有歷史對話嗎？')) return;
  state.conversations = [];
  state.activeConversationId = null;
  saveConversations();
  renderConversationHistory();
  resetChat();
  showToast('已清除歷史對話');
});
$('#closeSource').addEventListener('click', () => ui.sourceDialog.close());
ui.sourceDialog.addEventListener('click', (event) => { if (event.target === ui.sourceDialog) ui.sourceDialog.close(); });
$('#closeDocuments').addEventListener('click', () => ui.documentsDialog.close());
$('#closeDev')?.addEventListener('click', () => ui.devDialog?.close());
bindRagPair('devChunkSize', 'devChunkSizeRange');
bindRagPair('devChunkOverlap', 'devChunkOverlapRange');
bindRagPair('devTopK', 'devTopKRange');
$('#devCustomPrompt')?.addEventListener('input', () => {
  localStorage.setItem('knowledge-rag-custom-prompt', $('#devCustomPrompt').value);
});
$('#devDebug')?.addEventListener('change', () => {
  state.debugRetrieval = $('#devDebug').checked;
  localStorage.setItem('knowledge-rag-debug', state.debugRetrieval ? '1' : '0');
});
$('#devForm')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    const response = await fetch('/api/dev-settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: $('#devApiKey').value,
        base_url: $('#devBaseUrl').value.trim(),
        vector_backend: $('#devVectorBackend').value.trim(),
        embedding_model: $('#devEmbeddingModel').value.trim(),
        reranker_model: $('#devRerankerModel').value.trim(),
        ...readRagInputs()
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '儲存失敗');
    fillDevFields(data);
    showToast(data.reindex_started ? '設定已儲存，正在按新的切片或向量模型重建索引' : '開發者設定已儲存');
    checkServiceHealth();
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
});

$('#devModels')?.addEventListener('click', async (event) => {
  const edit = event.target.closest('[data-edit-model]');
  if (edit) {
    const row = edit.closest('.dev-model-row');
    $('#devModelLabel').value = row.querySelector('strong').textContent;
    $('#devModelId').value = edit.dataset.editModel;
    $('#devModelKey').value = '';
    $('#devModelKey').placeholder = edit.dataset.keySet === '1' ? '已設定專用 Key，留空則不更改' : '未設定，可填此模型的 API Key';
    $('#devModelUrl').value = edit.dataset.baseUrl || '';
    $('#devModelOriginal').value = edit.dataset.editModel;
    $('#devModelSubmit').textContent = '儲存修改';
    $('#devModelCancel').hidden = false;
    $('#devModelLabel').focus();
    return;
  }
  const remove = event.target.closest('[data-delete-model]');
  if (remove) {
    const modelId = remove.dataset.deleteModel;
    if (!window.confirm(`確定刪除模型「${modelId}」嗎？`)) return;
    try {
      const response = await fetch(`/api/dev-models?model_id=${encodeURIComponent(modelId)}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '刪除失敗');
      renderDevModels(data);
      showToast('已刪除模型');
    } catch (error) {
      showToast(error.message);
    }
    return;
  }
  const use = event.target.closest('[data-use-model]');
  if (!use || use.textContent === '預設') return;
  try {
    const response = await fetch('/api/dev-settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: use.dataset.useModel })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '設定失敗');
    renderDevModels(data);
    showToast('已設為預設模型');
  } catch (error) {
    showToast(error.message);
  }
});

$('#devModelCancel')?.addEventListener('click', resetModelEditor);
$('#devModelForm')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const original = $('#devModelOriginal').value;
  const payload = {
    id: original || $('#devModelId').value.trim(),
    label: $('#devModelLabel').value.trim(),
    new_id: $('#devModelId').value.trim(),
    api_key: $('#devModelKey').value,
    base_url: $('#devModelUrl').value.trim()
  };
  try {
    const response = await fetch('/api/dev-models', {
      method: original ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(original ? payload : { id: payload.new_id, label: payload.label, api_key: payload.api_key, base_url: payload.base_url })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '儲存模型失敗');
    resetModelEditor();
    renderDevModels(data);
    showToast(original ? '已修改模型' : '已新增模型');
  } catch (error) {
    showToast(error.message);
  }
});
$('#refreshDocuments').addEventListener('click', loadDocuments);
ui.documentsDialog.addEventListener('click', (event) => { if (event.target === ui.documentsDialog) ui.documentsDialog.close(); });
$('#refreshStatus').addEventListener('click', checkServiceHealth);

$('#documentInput').addEventListener('change', () => {
  const file = $('#documentInput').files[0];
  $('#uploadButton').disabled = !file;
  $('.upload-zone label strong').textContent = file ? file.name : '選擇文檔上傳';
});

$('#uploadForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#documentInput');
  if (!input.files[0]) return;
  const button = $('#uploadButton');
  button.disabled = true;
  button.textContent = '正在上傳…';
  const body = new FormData();
  body.append('file', input.files[0]);
  body.append('library', state.uploadLibrary || 'moldpdf');
  try {
    const response = await fetch('/api/documents', { method: 'POST', body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '上傳失敗');
    input.value = '';
    $('.upload-zone label strong').textContent = '選擇文檔上傳';
    showToast(`已上傳到${scopeLabel(result.library)}：${result.name}`);
    await Promise.all([loadDocuments(), checkServiceHealth()]);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.textContent = '上傳並重建索引';
    button.disabled = !input.files[0];
  }
});

ui.composer.addEventListener('submit', (event) => {
  event.preventDefault();
  if (state.generating) stopGeneration();
  else submitPrompt();
});

function resetChat() {
  if (state.generating) stopGeneration();
  ui.thread.replaceChildren();
  ui.thread.hidden = true;
  ui.welcome.hidden = false;
  ui.input.value = '';
  ui.input.style.height = 'auto';
  state.activeConversationId = null;
  renderConversationHistory();
  ui.input.focus();
  setNav(false);
}

async function submitPrompt() {
  const prompt = ui.input.value.trim();
  if (!prompt || state.generating) return;
  const buildingLibrary = state.activeLibrary;
  const scopeHitsBuilding = state.scope === '全部知識空間' || state.scope === buildingLibrary;
  if (state.useKnowledge && state.indexStatus === 'building' && scopeHitsBuilding) {
    showToast('所選知識庫仍在建立索引，本次先直接回答，其他知識庫仍可使用。');
  }
  state.lastPrompt = prompt;
  addHistoryMessage({ role: 'user', content: prompt });
  const currentRun = ++state.runId;
  ui.welcome.hidden = true;
  ui.thread.hidden = false;
  ui.thread.insertAdjacentHTML('beforeend', `<article class="message user"><div class="bubble">${escapeHtml(prompt)}</div></article>`);
  ui.input.value = '';
  ui.input.style.height = 'auto';
  setSendState(true);

  const pending = document.createElement('article');
  pending.className = 'message assistant pending';
  pending.setAttribute('aria-live', 'polite');
  const model = models[state.selectedModel];
  pending.innerHTML = `<div class="bot-avatar"><svg><use href="#i-spark" /></svg></div><div class="answer"><div class="answer-head"><strong>知匯 AI</strong><span>${state.selectedModel.toUpperCase()} · ${escapeHtml(model.name)}</span></div><div class="progress-list"><div class="active"><i></i><span>理解問題與檢索意圖</span></div><div><i></i><span>使用 Qwen3-Embedding-0.6B 檢索並重排序</span></div><div><i></i><span>調用 ${model.provider === 'online' ? 'HKPC GPU Platform 雲端模型' : '本地模型'}</span></div></div></div>`;
  ui.thread.appendChild(pending);
  scrollBottom(true);

  const apiPromise = requestModel(prompt, model);
  const steps = $$('.progress-list div', pending);
  for (let index = 0; index < steps.length; index += 1) {
    if (currentRun !== state.runId) return;
    if (index > 0) steps[index - 1].className = 'done';
    steps[index].className = 'active';
    await wait(index === 0 ? 280 : 360);
  }
  if (currentRun !== state.runId) return;
  const apiResult = await apiPromise;
  if (currentRun !== state.runId) return;
  if (apiResult?._error) renderApiError(pending, prompt, apiResult._error, model);
  else if (apiResult) renderApiAnswer(pending, prompt, apiResult, model);
  else renderApiError(pending, prompt, '請先執行 python main.py，再透過瀏覽器開啟平台網址。', model);
  state.controller = null;
  setSendState(false);
}

function setPendingHint(text) {
  const line = $('.message.pending .progress-list .active span');
  if (line) line.textContent = text;
}

function describeFetchError(error) {
  const message = String(error?.message || error || '');
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return `無法連接到服務（${window.location.origin}）。請在專案目錄執行「python main.py」，確認頁面左下角顯示「模型服務已連接」後再重試。`;
  }
  return message || '請求失敗';
}

function errorHint(message) {
  if (/索引/.test(message)) return '索引還在後台建立，可以稍後再問一次以帶上手冊原文。';
  if (/無法連接|连不上|Failed to fetch/i.test(message)) return '請保持 main.py 啟動視窗開啟，並只使用它列印的 http://127.0.0.1:連接埠 網址開啟頁面。';
  if (/model_not_found|模型.*不存在|does not exist/i.test(message)) return '此模型尚未被目前的 HKPC API Key 授權。請確認 HKPC GPU Platform 已開通該模型，或在模型選單選擇已授權模型。';
  return '請檢查模型路徑、知識庫文檔或線上 API Key，然後點擊重試。';
}

async function requestModel(prompt, model) {
  if (window.location.protocol === 'file:') return null;
  state.controller = new AbortController();
  const timer = window.setTimeout(() => state.controller?.abort('timeout'), model.provider === 'local' ? 600000 : 120000);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: state.controller.signal,
      body: JSON.stringify({
        question: prompt,
        provider: model.provider,
        model: model.id,
        knowledge_scope: state.scope,
        use_knowledge: state.useKnowledge,
        debug: state.debugRetrieval,
        ...((ragState.synced || ragState.dirty) ? readRagInputs() : {}),
        ...(customPromptText() ? { custom_prompt: customPromptText() } : {})
      })
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      if (state.controller?.signal?.reason === 'timeout') {
        return { _error: '請求逾時。請確認啟動視窗仍在運行。' };
      }
      return null;
    }
    return { _error: describeFetchError(error) };
  } finally {
    window.clearTimeout(timer);
  }
}

function inlineAnswer(text) {
  return escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
}

function formatAnswer(text) {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let list = null;
  let paragraph = [];
  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${inlineAnswer(paragraph.join(' '))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (!list) return;
    html.push(list === 'ol' ? '</ol>' : '</ul>');
    list = null;
  };
  const openList = (kind) => {
    flushParagraph();
    if (list !== kind) {
      closeList();
      list = kind;
      html.push(kind === 'ol' ? '<ol>' : '<ul>');
    }
  };
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line || /^(-{3,}|\*{3,})$/.test(line)) {
      flushParagraph();
      closeList();
      return;
    }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      html.push(`<h2>${inlineAnswer(heading[1])}</h2>`);
      return;
    }
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      openList('ol');
      html.push(`<li>${inlineAnswer(ordered[1])}</li>`);
      return;
    }
    const unordered = line.match(/^[-*•・]\s+(.+)$/);
    if (unordered) {
      openList('ul');
      html.push(`<li>${inlineAnswer(unordered[1])}</li>`);
      return;
    }
    closeList();
    paragraph.push(line);
  });
  flushParagraph();
  closeList();
  return html.join('');
}

function customPromptText() {
  const typed = $('#devCustomPrompt')?.value;
  const text = typed == null ? (localStorage.getItem('knowledge-rag-custom-prompt') || '') : typed;
  return text.trim();
}

function mapSource(source) {
  const hasRaw = source.rawScore !== undefined && source.rawScore !== null && source.rawScore !== '';
  const incoming = Number(hasRaw ? source.rawScore : source.score ?? 0);
  const rawScore = Number.isFinite(incoming) ? incoming : 0;
  const display = Math.round(rawScore * (rawScore <= 1 ? 100 : 1));
  return {
    type: source.type || 'DOC',
    title: source.title || source.source || '知識庫文檔',
    location: source.location || (Number.isInteger(source.page) ? `第 ${source.page + 1} 頁` : '相關片段'),
    rawScore,
    score: display,
    excerpt: source.excerpt || '',
    content: source.content || source.excerpt || ''
  };
}

function formatRawScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : '0.0000';
}

function renderApiAnswer(container, prompt, data, model, saveHistory = true) {
  const answer = typeof data.answer === 'string' ? data.answer : '介面未返回有效答案。';
  const sourceItems = Array.isArray(data.sources) ? data.sources.map(mapSource) : [];
  if (data.retrieval?.reindex_started) showToast('切片長度已更新，正在重建索引');
  const answerHtml = formatAnswer(answer);
  const retrieval = data.retrieval || {};
  const skipped = Boolean(retrieval.skipped);
  const retrievalLabel = skipped
    ? '索引建立中，未檢索文檔'
    : retrieval.enabled
      ? `已檢索 ${sourceItems.length} 個資料片段`
      : '未使用知識庫';
  const sourceBlock = skipped
    ? '<div class="no-source">知識庫正在後台建立索引，本次先直接回答，不影響繼續提問或管理文檔。索引完成後即可引用原文。</div>'
    : (state.useKnowledge && sourceItems.length ? renderSources(sourceItems) : '<div class="no-source">本次介面未返回可展示的引用片段。</div>');
  container.classList.remove('pending');
  container.querySelector('.answer').innerHTML = `
    <div class="answer-head"><strong>知匯 AI</strong><span>${state.selectedModel.toUpperCase()} · ${escapeHtml(model.name)}</span><em class="confidence">${retrievalLabel}</em></div>
    <div class="api-answer">${answerHtml}</div>
    ${sourceBlock}
    <div class="answer-actions"><button class="copy-answer"><svg><use href="#i-copy" /></svg>複製</button><button class="retry-answer">重新生成</button><button data-toast="感謝回饋，正式版本會將回饋寫入評測記錄">有幫助</button><button data-toast="感謝回饋，正式版本會記錄問題類型">需改進</button></div>`;
  bindAnswerActions(container, prompt, sourceItems);
  if (saveHistory) {
    addHistoryMessage({
      role: 'assistant', status: 'answer', content: answer,
      modelKey: state.selectedModel, sources: sourceItems
    });
  }
  scrollBottom(true);
}

function renderApiError(container, prompt, message, model, saveHistory = true) {
  container.classList.remove('pending');
  container.querySelector('.answer').innerHTML = `
    <div class="answer-head"><strong>知匯 AI</strong><span>${state.selectedModel.toUpperCase()} · ${escapeHtml(model.name)}</span><em class="confidence error">調用失敗</em></div>
    <div class="error-card"><strong>暫時無法生成回答</strong><p>${escapeHtml(message)}</p><small>${escapeHtml(errorHint(message))}</small></div>
    <div class="answer-actions"><button class="retry-answer">重新嘗試</button><button data-toast="請點擊左下角狀態卡重新整理服務狀態">檢查服務狀態</button></div>`;
  $('.retry-answer', container).addEventListener('click', () => { ui.input.value = prompt; submitPrompt(); });
  $$('[data-toast]', container).forEach((button) => button.addEventListener('click', () => showToast(button.dataset.toast)));
  if (saveHistory) {
    addHistoryMessage({
      role: 'assistant', status: 'error', content: message, modelKey: state.selectedModel
    });
  }
  scrollBottom(true);
}

function renderSources(sources) {
  const rows = sources.map((source, index) => {
    const debug = state.debugRetrieval
      ? `<div class="source-debug"><p>Reranker 分數 ${formatRawScore(source.rawScore)} · 相關度 ${source.score}%</p><pre>${escapeHtml(source.content || source.excerpt || '')}</pre></div>`
      : '';
    return `<div class="source-block"><button class="source-row" data-source-index="${index}"><i>${source.type}</i><span><strong>${escapeHtml(source.title)}</strong><small>${escapeHtml(source.location)}</small></span><em>${source.score}%</em><svg><use href="#i-chevron" /></svg></button>${debug}</div>`;
  }).join('');
  return `<section class="sources"><div class="sources-title"><svg><use href="#i-book" /></svg><span>參考依據</span><small>${sources.length} 個相關片段 · 點擊查看原文</small></div>${rows}</section>`;
}

function bindAnswerActions(container, prompt, sources) {
  $('.copy-answer', container).addEventListener('click', async (event) => {
    const text = $('.answer', container).innerText.replace(/複製|重新生成|有幫助|需改進/g, '').trim();
    try { await navigator.clipboard.writeText(text); event.currentTarget.textContent = '已複製'; }
    catch { showToast('瀏覽器未授權剪貼簿，請手動選擇文字'); }
  });
  $('.retry-answer', container).addEventListener('click', () => {
    ui.input.value = prompt;
    submitPrompt();
  });
  $$('[data-fill]', container).forEach((button) => button.addEventListener('click', () => {
    ui.input.value = `補充資訊：${button.dataset.fill}：`;
    ui.input.dispatchEvent(new Event('input'));
    ui.input.focus();
  }));
  $$('[data-toast]', container).forEach((button) => button.addEventListener('click', () => showToast(button.dataset.toast)));
  $$('.source-row', container).forEach((button) => button.addEventListener('click', () => openSource(sources[Number(button.dataset.sourceIndex)])));
}

function openSource(source) {
  $('#sourceTitle').textContent = source.title;
  const raw = state.debugRetrieval ? ` · Reranker ${formatRawScore(source.rawScore)}` : '';
  $('#sourceMeta').textContent = `${source.type} · ${source.location} · 檢索相關度 ${source.score}%${raw}`;
  $('#sourceExcerpt').textContent = state.debugRetrieval ? (source.content || source.excerpt) : source.excerpt;
  ui.sourceDialog.showModal();
}

function scrollBottom(force = false) {
  requestAnimationFrame(() => {
    const distance = ui.conversation.scrollHeight - ui.conversation.scrollTop - ui.conversation.clientHeight;
    if (force || distance < 180) ui.conversation.scrollTop = ui.conversation.scrollHeight;
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function openDocuments() {
  setNav(false);
  ui.documentsDialog.showModal();
  await loadDocuments();
}

async function loadDocuments() {
  const list = $('#documentList');
  list.innerHTML = '<div class="empty-documents">正在讀取文檔…</div>';
  try {
    const response = await fetch('/api/documents');
    if (!response.ok) throw new Error('文檔服務未連接');
    const data = await response.json();
    if (data.libraries) {
      state.libraries = data.libraries;
      renderScopeMenu(state.libraries);
      renderLibraryChips(state.libraries);
    }
    const visible = data.documents.filter((document) => document.library === state.uploadLibrary);
    if (!visible.length) {
      const name = scopeLabel(state.uploadLibrary);
      list.innerHTML = `<div class="empty-documents"><strong>${escapeHtml(name)}尚無文檔</strong><span>上傳後只會加入這個類型。</span></div>`;
      return;
    }
    list.innerHTML = visible.map((document) => `
      <div class="document-row">
        <i>${escapeHtml(document.type)}</i>
        <span><strong>${escapeHtml(document.name)}</strong><small><em>${escapeHtml(document.library_name || document.library || '')}</em>${formatBytes(document.size)}</small></span>
        <button data-delete-document="${escapeHtml(document.path || document.name)}" data-delete-library="${escapeHtml(document.library || '')}" aria-label="刪除 ${escapeHtml(document.name)}"><svg><use href="#i-trash" /></svg></button>
      </div>`).join('');
    $$('[data-delete-document]', list).forEach((button) => button.addEventListener('click', async () => {
      const name = button.dataset.deleteDocument;
      const library = button.dataset.deleteLibrary;
      if (!window.confirm(`確定從「${scopeLabel(library)}」刪除「${name}」嗎？此操作無法復原。`)) return;
      button.disabled = true;
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(library)}/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || '刪除失敗');
        showToast(`已刪除：${name}`);
        await Promise.all([loadDocuments(), checkServiceHealth()]);
      } catch (error) { showToast(error.message); }
    }));
  } catch (error) {
    list.innerHTML = `<div class="empty-documents"><strong>無法讀取文檔</strong><span>${escapeHtml(error.message)}，請透過 main.py 啟動平台。</span></div>`;
  }
}

$('#libraryChips')?.addEventListener('click', async (event) => {
  const remove = event.target.closest('[data-delete-library-id]');
  if (remove) {
    const libraryId = remove.dataset.deleteLibraryId;
    const item = state.libraries.find((entry) => entry.id === libraryId);
    const name = item?.name || scopeLabel(libraryId);
    const count = item?.document_count || 0;
    if (!window.confirm(`確定刪除類型「${name}」嗎？其中 ${count} 份文檔會一併刪除，此操作無法復原。`)) return;
    remove.disabled = true;
    try {
      const response = await fetch(`/api/libraries/${encodeURIComponent(libraryId)}`, { method: 'DELETE' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || '刪除類型失敗');
      if (state.scope === libraryId) {
        state.scope = '全部知識空間';
        state.useKnowledge = true;
      }
      if (state.uploadLibrary === libraryId) state.uploadLibrary = '';
      showToast(`已刪除類型：${result.name || name}`);
      await Promise.all([loadDocuments(), checkServiceHealth()]);
    } catch (error) {
      showToast(error.message);
      remove.disabled = false;
    }
    return;
  }
  const pick = event.target.closest('[data-library]');
  if (!pick) return;
  state.uploadLibrary = pick.dataset.library;
  renderLibraryChips(state.libraries);
  await loadDocuments();
});

$('#createLibraryForm')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#newLibraryName');
  const name = input?.value.trim();
  if (!name) {
    showToast('請輸入類型名稱');
    return;
  }
  try {
    const response = await fetch('/api/libraries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '新增類型失敗');
    state.uploadLibrary = result.id;
    input.value = '';
    showToast(`已新增類型：${result.name}`);
    await Promise.all([loadDocuments(), checkServiceHealth()]);
  } catch (error) {
    showToast(error.message);
  }
});

checkServiceHealth();
loadDevSettings().catch(() => {});
