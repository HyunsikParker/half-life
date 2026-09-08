export const MAX_CARDS = 100;
export const MAX_REVIEWS = 1000;
const DAY = 86400000;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const validTime = value => Number.isSafeInteger(value) && value > 0;
const text = (value, max) => typeof value === 'string' && value.trim().length > 0 && value.length <= max;

export function validateCard(card) {
  if (card?.version !== 1 || !text(card.id, 100) || !text(card.modelId, 100) ||
      !text(card.word, 200) || !text(card.answer, 2000) || !validTime(card.createdAt) ||
      !Array.isArray(card.reviews) || card.reviews.length > MAX_REVIEWS) {
    throw new Error('Stored card is unreadable. Its data has not been changed.');
  }
  let previous = card.createdAt;
  const ids = new Set();
  for (const event of card.reviews) {
    if (!text(event.id, 100) || ids.has(event.id) || !validTime(event.at) || event.at < previous ||
        !['recalled', 'forgot'].includes(event.outcome)) throw new Error('Stored review history is unreadable.');
    ids.add(event.id);
    previous = event.at;
  }
  return card;
}

export function createCard({ id, item, answer, now = Date.now() }) {
  if (!text(answer, 2000)) throw new Error('Write an answer between 1 and 2,000 characters.');
  if (!item?.id || !item?.text) throw new Error('Choose a word from the model.');
  return validateCard({ version: 1, id, modelId: item?.id, word: item?.text,
    answer: typeof answer === 'string' ? answer.trim() : answer, createdAt: now, reviews: [] });
}

export function recordReview(card, { id, outcome, at = Date.now() }) {
  validateCard(card);
  const existing = card.reviews.find(event => event.id === id);
  if (existing) {
    if (existing.outcome !== outcome) throw new Error('This review was already recorded with another result.');
    return card;
  }
  if (card.reviews.length >= MAX_REVIEWS) throw new Error('This card has reached its 1,000-review limit.');
  if (at < (card.reviews.at(-1)?.at ?? card.createdAt)) throw new Error('Device clock moved backwards. Correct the clock before saving.');
  return validateCard({ ...card, reviews: [...card.reviews, { id, outcome, at }] });
}

export function summarizeCard(card, model, now = Date.now()) {
  validateCard(card);
  const item = model.lexemes.find(item => item.id === card.modelId);
  if (!item || ![model.right, model.wrong, model.bias, item.weight].every(Number.isFinite)) {
    throw new Error('The saved word is not available in this model.');
  }
  const right = card.reviews.filter(event => event.outcome === 'recalled').length;
  const wrong = card.reviews.length - right;
  const dot = model.right * Math.sqrt(1 + right) + model.wrong * Math.sqrt(1 + wrong) + model.bias + item.weight;
  const halfLifeDays = clamp(2 ** clamp(dot, -20, 20), 15 / (24 * 60), 274);
  const lastReviewedAt = card.reviews.at(-1)?.at ?? null;
  const nextEstimateAt = lastReviewedAt === null ? card.createdAt :
    Math.round(lastReviewedAt + (-halfLifeDays * Math.log2(0.9)) * DAY);
  const predictedRecall = lastReviewedAt === null ? null : 2 ** (-Math.max(0, now - lastReviewedAt) / DAY / halfLifeDays);
  return { right, wrong, halfLifeDays, lastReviewedAt, nextEstimateAt, predictedRecall };
}
