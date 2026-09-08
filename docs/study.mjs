import { createCard, summarizeCard } from './study-core.mjs?v=1';
import { openStudyStore } from './study-store.mjs?v=1';

const el = id => document.getElementById(id);
const status = message => { el('studyStatus').textContent = message; };
const node = (tag, value, className) => {
  const element = document.createElement(tag);
  if (value !== undefined) element.textContent = value;
  if (className) element.className = className;
  return element;
};

async function init() {
  const response = await fetch('model.json');
  if (!response.ok) throw new Error('The model could not be loaded. Cards are unavailable.');
  const model = await response.json();
  const items = model.lexemes.filter(item => item.text && /^[\p{L}'-]+$/u.test(item.text)).slice(0, 220);
  const store = await openStudyStore();
  for (const item of items) {
    const option = node('option', item.text);
    option.value = item.id;
    el('studyWord').append(option);
  }

  async function render() {
    const cards = await store.list();
    cards.sort((a, b) => summarizeCard(a, model).nextEstimateAt - summarizeCard(b, model).nextEstimateAt);
    const fragment = document.createDocumentFragment();
    for (const card of cards) {
      const summary = summarizeCard(card, model);
      const article = node('article', undefined, 'study-card');
      article.dataset.cardId = card.id;
      const heading = node('h3', card.word);
      const counts = node('p', `${summary.right} recalled · ${summary.wrong} forgotten`, 'study-counts');
      const timing = node('p', summary.lastReviewedAt === null ? 'Not reviewed yet. Try recalling the answer.' :
        `Model 90% estimate: ${new Date(summary.nextEstimateAt).toLocaleString()}. You can review sooner.`, 'study-timing');
      const answer = node('p', card.answer, 'study-answer');
      answer.hidden = true;
      const actions = node('div', undefined, 'study-actions');
      const reveal = node('button', 'Show answer');
      reveal.type = 'button';
      const right = node('button', 'Recalled');
      const wrong = node('button', 'Forgot');
      const remove = node('button', 'Remove card');
      for (const button of [right, wrong, remove]) button.type = 'button';
      right.disabled = wrong.disabled = true;
      let attempt = null, saving = false;
      reveal.addEventListener('click', () => {
        if (saving) return;
        answer.hidden = false;
        attempt ??= crypto.randomUUID();
        right.disabled = wrong.disabled = false;
        reveal.disabled = true;
        right.focus();
      });
      const grade = async outcome => {
        if (!attempt || saving) return;
        saving = true;
        right.disabled = wrong.disabled = remove.disabled = true;
        try {
          await store.review(card.id, { id: attempt, outcome, at: Date.now() });
          await render();
          status('Review saved on this device.');
          el('studyStatus').focus();
        } catch (error) {
          status(error.message);
          right.disabled = wrong.disabled = remove.disabled = false;
          saving = false;
        }
      };
      right.addEventListener('click', () => grade('recalled'));
      wrong.addEventListener('click', () => grade('forgot'));
      remove.addEventListener('click', async () => {
        if (saving || !confirm('Remove this card and its review history from this browser?')) return;
        saving = true;
        try { await store.remove(card.id); await render(); status('Card removed from this device.'); }
        catch (error) { saving = false; status(error.message); }
      });
      actions.append(reveal, right, wrong, remove);
      article.append(heading, counts, timing, answer, actions);
      fragment.append(article);
    }
    if (!cards.length) fragment.append(node('p', 'No cards yet. Add a word and your own answer above.'));
    el('studyCards').replaceChildren(fragment);
  }

  await render();
  el('studyWord').disabled = el('studyAnswer').disabled = el('studyAdd').disabled = false;
  status('Ready. Cards and review history stay in this browser.');
  el('studyForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (el('studyAdd').disabled) return;
    el('studyAdd').disabled = true;
    try {
      const item = items.find(item => item.id === el('studyWord').value);
      const card = createCard({ id: crypto.randomUUID(), item, answer: el('studyAnswer').value });
      await store.add(card);
      el('studyAnswer').value = '';
      await render();
      status('Card saved on this device. Recall the answer before revealing it.');
    } catch (error) { status(error.message); }
    finally { el('studyAdd').disabled = false; }
  });
  el('studyReload').addEventListener('click', () => render().then(() => status('Cards reloaded from this device.')).catch(error => status(error.message)));
}
init().catch(error => { status(error.message); });
