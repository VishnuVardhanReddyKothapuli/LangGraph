const $ = (id) => document.getElementById(id);
let history = [];
let busy = false;
const documents = new Map();

function status(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle('error', error);
}

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check your input and try again.');
  return data;
}

async function refreshStatus() {
  try {
    const data = await request('/api/status');
    $('connection').textContent = `${data.chunks} indexed passages · ${data.providers.length ? data.providers.join(' + ') : 'No AI key set'}`;
    if (!data.providers.length) status('chat-status', 'Add a Gemini or Groq API key to .env and restart to enable answers.');
    $('welcome').querySelector('.welcome-hint').textContent = data.chunks ? 'Your library is ready. Ask your first question.' : 'Start by adding a document to your library.';
  } catch (error) {
    $('connection').textContent = 'Library unavailable';
    status('chat-status', error.message, true);
  }
}

function setBusy(value) {
  busy = value;
  for (const id of ['send', 'files', 'new-chat']) $(id).disabled = value;
  $('chat-form').setAttribute('aria-busy', String(value));
}

$('files').addEventListener('change', async (event) => {
  const files = [...event.target.files];
  if (!files.length || busy) return;
  setBusy(true);
  const failures = [];
  let added = 0;
  for (const file of files) {
    status('upload-status', `Indexing ${file.name}… First use downloads embedding models.`);
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error('File exceeds 10 MB.');
      const body = new FormData();
      body.append('file', file);
      const result = await request('/api/documents', { method: 'POST', body });
      documents.set(result.name, result.chunks);
      added++;
    } catch (error) { failures.push(`${file.name}: ${error.message}`); }
  }
  $('document-list').replaceChildren();
  for (const [name, chunks] of documents) {
    const item = document.createElement('li');
    item.textContent = name;
    const detail = document.createElement('small');
    detail.textContent = `${chunks} passages · ready to explore`;
    item.append(detail);
    $('document-list').append(item);
  }
  $('file-count').textContent = String(documents.size).padStart(2, '0');
  status('upload-status', failures.length ? `${added} indexed. ${failures.join(' ')}` : `${added} document${added === 1 ? '' : 's'} ready. Ask away.`, failures.length > 0);
  $('files').value = '';
  setBusy(false);
  await refreshStatus();
});

function addMessage(role, text, result) {
  $('welcome').hidden = true;
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'YOU' : 'FOLIO';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text; // Never interpret model output or filenames as HTML.
  article.append(label, body);
  if (result) {
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = `${result.provider === 'none' ? 'Library' : result.provider} · ${result.input_tokens.toLocaleString()} input reference tokens`;
    article.append(meta);
    if (result.sources.length) {
      const details = document.createElement('details');
      details.className = 'sources';
      const summary = document.createElement('summary');
      summary.textContent = `Explore ${result.sources.length} source excerpt${result.sources.length === 1 ? '' : 's'}`;
      details.append(summary);
      for (const source of result.sources) {
        const section = document.createElement('section');
        section.className = 'source';
        const title = document.createElement('strong');
        title.textContent = `[${source.id}] ${source.source}${source.page ? ` · page ${source.page}` : ''}`;
        const excerpt = document.createElement('p');
        excerpt.textContent = source.excerpt;
        section.append(title, excerpt);
        details.append(section);
      }
      article.append(details);
    }
  }
  $('messages').append(article);
  $('conversation').scrollTop = $('conversation').scrollHeight;
  return article;
}

$('chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || busy) return;
  setBusy(true);
  const userMessage = addMessage('user', question);
  $('question').value = '';
  status('chat-status', 'Reading relevant passages and preparing your answer…');
  try {
    const result = await request('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: history.slice(-20) }),
    });
    addMessage('assistant', result.answer, result);
    history.push({ role: 'user', content: question }, { role: 'assistant', content: result.answer });
    history = history.slice(-20);
    status('chat-status', '');
  } catch (error) {
    userMessage.remove();
    if (!$('messages').children.length) $('welcome').hidden = false;
    $('question').value = question;
    status('chat-status', error.message, true);
  } finally {
    setBusy(false);
    $('question').focus();
  }
});

$('question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('chat-form').requestSubmit();
  }
});

$('new-chat').addEventListener('click', () => {
  history = [];
  $('messages').replaceChildren();
  $('welcome').hidden = false;
  $('question').value = '';
  status('chat-status', '');
  $('question').focus();
});

document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => {
  $('question').value = button.dataset.prompt;
  $('question').focus();
}));

refreshStatus();
