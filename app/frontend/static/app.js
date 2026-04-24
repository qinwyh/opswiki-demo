/* OpsWiki Frontend – v2 */

// ── Mermaid init ──
mermaid.initialize({
  startOnLoad: false,
  theme: 'neutral',
  securityLevel: 'loose',
  flowchart: { useMaxWidth: true, htmlLabels: true },
  sequence:   { useMaxWidth: true },
});

// ── Global state ──
let currentRepoId   = null;
let currentCitations = [];  // citations for in-progress answer

// ════════════════════════════════════════════
// Phase 1 – Repo Picker
// ════════════════════════════════════════════

async function loadRepoList() {
  const grid = document.getElementById('repoGrid');
  grid.innerHTML = '<div class="repo-grid-placeholder"><div class="spinner"></div><span>加载中…</span></div>';
  try {
    const resp = await fetch('/api/repos');
    const data = await resp.json();
    if (!data.repos || data.repos.length === 0) {
      grid.innerHTML = '<div class="repo-grid-empty">暂无已索引的仓库，请在下方导入新仓库</div>';
      return;
    }
    grid.innerHTML = data.repos.map(r => `
      <div class="repo-card" onclick="selectRepo('${escapeHtml(r.repo_id)}')">
        <div class="repo-card-name">📁 ${escapeHtml(r.name)}</div>
        <div class="repo-card-path" title="${escapeHtml(r.path)}">${escapeHtml(r.path)}</div>
        <div class="repo-card-stats">
          <span class="repo-card-stat">📄 ${r.file_count} 文件</span>
          <span class="repo-card-stat">🔷 ${r.chunk_count} 块</span>
        </div>
        <div class="repo-card-status${r.status !== 'ready' ? ' indexing' : ''}">
          ${r.status === 'ready' ? '✓ 就绪' : escapeHtml(r.status)}
        </div>
      </div>
    `).join('');
  } catch (e) {
    grid.innerHTML = `<div class="repo-grid-empty" style="color:var(--error)">加载失败: ${escapeHtml(e.message)}</div>`;
  }
}

async function selectRepo(repoId) {
  try {
    const resp = await fetch(`/api/repo/${repoId}`);
    if (!resp.ok) throw new Error('仓库信息加载失败');
    const data = await resp.json();
    currentRepoId = repoId;
    enterChat(data);
  } catch (e) {
    alert('加载仓库失败: ' + e.message);
  }
}

async function importNewRepo() {
  const pathEl   = document.getElementById('newRepoPath');
  const statusEl = document.getElementById('importStatus');
  const btn      = document.getElementById('importBtn');
  const path     = pathEl.value.trim();
  if (!path) {
    statusEl.textContent = '请输入仓库路径';
    statusEl.className = 'import-status error';
    return;
  }
  btn.disabled = true;
  statusEl.textContent = '⏳ 正在扫描并建立索引，这可能需要几分钟…';
  statusEl.className = 'import-status loading';
  try {
    const resp = await fetch('/api/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '导入失败');
    statusEl.textContent = `✅ 索引完成！${data.file_count || 0} 个文件，${data.chunk_count || 0} 个块`;
    statusEl.className = 'import-status success';
    currentRepoId = data.repo_id;
    setTimeout(() => { loadRepoList(); enterChat(data); }, 700);
  } catch (e) {
    statusEl.textContent = '❌ ' + e.message;
    statusEl.className = 'import-status error';
  } finally {
    btn.disabled = false;
  }
}

function enterChat(repoData) {
  document.getElementById('pickerScreen').style.display = 'none';
  document.getElementById('appScreen').style.display   = 'block';
  const name = repoData.name || repoData.repo_id || '—';
  document.getElementById('sidebarRepoName').textContent   = name;
  document.getElementById('sidebarRepoMeta').textContent   =
    `${repoData.file_count ?? '?'} 文件 · ${repoData.chunk_count ?? '?'} 块`;
  document.getElementById('placeholderRepoName').textContent = name;
  if (repoData.repo_id || currentRepoId) {
    loadFileTree(repoData.repo_id || currentRepoId);
  }
  document.getElementById('questionInput').focus();
}

function backToPicker() {
  document.getElementById('appScreen').style.display   = 'none';
  document.getElementById('pickerScreen').style.display = 'flex';
  loadRepoList();
}

async function loadFileTree(repoId) {
  try {
    const resp = await fetch(`/api/repo/${repoId}/files`);
    const data = await resp.json();
    const el   = document.getElementById('fileTree');
    el.innerHTML = (data.files || []).map(f =>
      `<div class="file-tree-item" title="${escapeHtml(f)}">📄 ${escapeHtml(f)}</div>`
    ).join('');
  } catch (e) { console.error(e); }
}

// ════════════════════════════════════════════
// Chat / Ask
// ════════════════════════════════════════════

async function askQuestion() {
  const input    = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question) return;
  if (!currentRepoId) { backToPicker(); return; }

  input.value = '';
  autoResize(input);
  hidePlaceholder();

  appendMessage('user', question);
  const msgEl    = appendMessage('assistant', '');
  const contentEl = msgEl.querySelector('.msg-content');
  contentEl.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

  document.getElementById('askBtn').disabled = true;
  // Reset citations panel
  document.getElementById('citationsList').innerHTML = '';
  document.getElementById('citationsHint').style.display  = 'block';
  const countBadge = document.getElementById('citationsCount');
  countBadge.style.display = 'none';
  currentCitations = [];

  try {
    const resp = await fetch('/api/ask', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ repo_id: currentRepoId, question }),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullText    = '';
    let mermaidCode = '';
    let followups   = [];
    let started     = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of decoder.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload) continue;
        try {
          const msg = JSON.parse(payload);
          if (msg.type === 'citations') {
            currentCitations = msg.data;
            renderCitations(currentCitations);
          } else if (msg.type === 'token') {
            if (!started) { contentEl.innerHTML = ''; started = true; }
            fullText += msg.data;
            contentEl.innerHTML = renderMarkdown(fullText, currentCitations);
            scrollChat();
          } else if (msg.type === 'done') {
            mermaidCode = msg.mermaid  || '';
            followups   = msg.followups || [];
          }
        } catch (_) {}
      }
    }

    // ── Final render ──
    let finalHtml = renderMarkdown(fullText, currentCitations);

    if (mermaidCode) {
      const mId = 'mermaid-' + Date.now();
      finalHtml += `<div class="mermaid-container"><div class="mermaid" id="${mId}">${escapeHtml(sanitizeMermaid(mermaidCode))}</div></div>`;
    }

    if (followups.length) {
      finalHtml += '<div class="followup-section"><div class="followup-label">继续探索</div><div class="followup-chips">';
      followups.forEach(f => {
        finalHtml += `<button class="followup-btn" onclick="fillAndAsk(this)">${escapeHtml(f)}</button>`;
      });
      finalHtml += '</div></div>';
    }

    contentEl.innerHTML = finalHtml;
    // Attach this answer's citations for later ref-click lookups
    msgEl.dataset.citationsJson = JSON.stringify(currentCitations);

    await renderMermaidDiagrams();

  } catch (e) {
    contentEl.innerHTML = `<span style="color:var(--error)">错误: ${escapeHtml(e.message)}</span>`;
  } finally {
    document.getElementById('askBtn').disabled = false;
    scrollChat();
  }
}

function appendMessage(role, text) {
  const history = document.getElementById('chatHistory');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="msg-content">${role === 'user' ? escapeHtml(text) : text}</div>`;
  history.appendChild(div);
  scrollChat();
  return div;
}

function hidePlaceholder() {
  document.getElementById('chatPlaceholder')?.classList.add('hidden');
}

function scrollChat() {
  const h = document.getElementById('chatHistory');
  h.scrollTop = h.scrollHeight;
}

// ════════════════════════════════════════════
// Citations
// ════════════════════════════════════════════

function renderCitations(citations) {
  const listEl  = document.getElementById('citationsList');
  const hintEl  = document.getElementById('citationsHint');
  const countEl = document.getElementById('citationsCount');

  hintEl.style.display    = 'none';
  countEl.style.display   = 'inline-block';
  countEl.textContent     = citations.length + ' 处';

  listEl.innerHTML = citations.map((c) => {
    const refLabel   = `${c.rel_path}:${c.start_line}-${c.end_line}`;
    const symbolInfo = [c.symbol_name, c.chunk_type].filter(Boolean).join(' · ');
    return `
      <div class="citation-card" data-ref="${escapeHtml(refLabel)}" onclick="toggleCitation(this)">
        <div class="citation-card-header">
          <div class="citation-ref-label">${escapeHtml(refLabel)}</div>
          ${symbolInfo ? `<div class="citation-symbol">${escapeHtml(symbolInfo)}</div>` : ''}
        </div>
        <div class="citation-snippet">${escapeHtml(c.snippet || '')}</div>
      </div>`;
  }).join('');
}

function toggleCitation(card) {
  card.classList.toggle('expanded');
}

// Click on a cite-ref pill → highlight matching card
document.addEventListener('click', function (e) {
  const ref = e.target.closest('.cite-ref');
  if (!ref) return;
  const refLabel = ref.dataset.ref;

  // If this ref belongs to a specific message, reload that message's citations first
  const msgEl = ref.closest('.msg.assistant');
  if (msgEl?.dataset.citationsJson) {
    renderCitations(JSON.parse(msgEl.dataset.citationsJson));
  }

  // Highlight the matching card
  const cards = document.querySelectorAll('#citationsList .citation-card');
  cards.forEach(c => c.classList.remove('highlighted', 'expanded'));
  const target = Array.from(cards).find(c => c.dataset.ref === refLabel);
  if (target) {
    target.classList.add('highlighted', 'expanded');
    target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setTimeout(() => target.classList.remove('highlighted'), 3000);
  }
});

// ════════════════════════════════════════════
// Markdown & Mermaid
// ════════════════════════════════════════════

/**
 * Render markdown text. If citations provided, replace [N] markers
 * with clickable file:line pills referencing the Nth citation.
 */
function renderMarkdown(text, citations) {
  let html;
  try { html = marked.parse(text); } catch { html = escapeHtml(text); }

  if (citations && citations.length) {
    html = html.replace(/\[(\d+)\]/g, (match, numStr) => {
      const idx = parseInt(numStr, 10) - 1;
      if (idx < 0 || idx >= citations.length) return match;
      const c        = citations[idx];
      const refLabel = `${c.rel_path}:${c.start_line}-${c.end_line}`;
      return `<span class="cite-ref" data-ref="${escapeHtml(refLabel)}" title="查看引用：${escapeHtml(refLabel)}">${escapeHtml(refLabel)}</span>`;
    });
  }
  return html;
}

/** Fix common LLM-generated Mermaid issues before rendering. */
function sanitizeMermaid(code) {
  code = code.trim().replace(/\r\n/g, '\n');
  // Replace deprecated "graph" with "flowchart"
  code = code.replace(/^graph\s+(TD|LR|TB|RL|BT)\b/im, (_, dir) => `flowchart ${dir}`);
  // Strip YAML front-matter if accidentally included
  code = code.replace(/^---[\s\S]*?---\n/, '');
  return code;
}

async function renderMermaidDiagrams() {
  const els = document.querySelectorAll('.mermaid:not([data-processed])');
  for (const el of els) {
    try {
      const raw = sanitizeMermaid(el.textContent);
      const id  = (el.id || ('m' + Date.now())) + '-svg';
      const { svg } = await mermaid.render(id, raw);
      el.innerHTML = svg;
      el.setAttribute('data-processed', 'true');
    } catch (err) {
      console.warn('Mermaid render error:', err);
      const raw       = el.textContent;
      const container = el.closest('.mermaid-container');
      if (container) container.classList.add('mermaid-error');
      el.innerHTML = `
        <div class="mermaid-error-title">
          <span>⚠️ 图表渲染失败</span>
          <button class="btn-copy-code" onclick="copyToClipboard(this.dataset.code)"
                  data-code="${escapeHtml(raw)}">复制代码</button>
        </div>
        <pre style="font-size:.78em;color:var(--error);overflow-x:auto">${escapeHtml(err.message)}</pre>
        <pre style="font-size:.74em;color:var(--text-secondary);overflow-x:auto;margin-top:6px">${escapeHtml(raw)}</pre>`;
    }
  }
}

// ════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════

function escapeHtml(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function fillQuestion(el) {
  document.getElementById('questionInput').value = el.textContent;
  document.getElementById('questionInput').focus();
}

function fillAndAsk(el) {
  document.getElementById('questionInput').value = el.textContent;
  askQuestion();
}

function handleInputKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function copyToClipboard(text) {
  navigator.clipboard?.writeText(text).catch(() => {});
}

// ── Init ──
document.addEventListener('DOMContentLoaded', loadRepoList);