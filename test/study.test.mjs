import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createCard, recordReview, summarizeCard, MAX_REVIEWS } from '../docs/study-core.mjs';
import { openStudyStore } from '../docs/study-store.mjs';
const model = JSON.parse(await readFile(new URL('../docs/model.json', import.meta.url)));
const createdAt = 1788800000000;
const fresh = () => createCard({ id: 'card-1', item: model.lexemes[0], answer: 'an indefinite article', now: createdAt });

test('new cards start with no invented review history', () => {
 const card = fresh();
 const summary = summarizeCard(card, model, createdAt);
 assert.equal(summary.right, 0); assert.equal(summary.wrong, 0);
 assert.equal(summary.lastReviewedAt, null); assert.equal(summary.nextEstimateAt, createdAt);
 assert.equal(summary.predictedRecall, null);
});
test('recorded outcomes drive counts and the unchanged model estimate', () => {
 const first = recordReview(fresh(), { id: 'attempt-1', outcome: 'recalled', at: createdAt + 1000 });
 const second = recordReview(first, { id: 'attempt-2', outcome: 'forgot', at: createdAt + 2000 });
 const s = summarizeCard(second, model, createdAt + 2000);
 const expected = Math.min(274, Math.max(15/1440, 2 ** (model.right*Math.sqrt(2) + model.wrong*Math.sqrt(2) + model.bias + model.lexemes[0].weight)));
 assert.equal(first.reviews.length, 1); assert.equal(second.reviews.length, 2);
 assert.equal(s.right, 1); assert.equal(s.wrong, 1); assert.equal(s.halfLifeDays, expected);
 assert.equal(s.nextEstimateAt, Math.round(createdAt + 2000 - expected*Math.log2(.9)*86400000));
 assert.equal(s.predictedRecall, 1);
});
test('repeated delivery cannot count the same attempt twice or change its result', () => {
 const event = { id: 'attempt', outcome: 'recalled', at: createdAt + 1000 };
 const card = recordReview(fresh(), event);
 assert.equal(recordReview(card, { ...event, at: createdAt + 2000 }), card);
 assert.throws(() => recordReview(card, { ...event, outcome: 'forgot' }), /already recorded/);
});
test('clock rollback and corrupt outcomes do not mutate history', () => {
 const card = fresh();
 assert.throws(() => recordReview(card, { id: 'x', outcome: 'forgot', at: createdAt - 1 }), /clock/);
 assert.throws(() => recordReview(card, { id: 'x', outcome: 'unknown', at: createdAt }), /unreadable/);
 assert.deepEqual(card.reviews, []);
});
test('bounded history refuses an extra review without pruning prior evidence', () => {
 let card = fresh();
 for(let i=0;i<MAX_REVIEWS;i++) card = recordReview(card, {id: String(i),outcome:'recalled',at:createdAt+i});
 assert.throws(() => recordReview(card, { id: 'overflow', outcome:'forgot',at:createdAt+MAX_REVIEWS}), /limit/);
 assert.equal(card.reviews.length,MAX_REVIEWS);
});
test('blank answers and unavailable storage are explicit failures', async () => {
 assert.throws(() => createCard({id:'x',item:model.lexemes[0],answer:'  ',now:createdAt}), /Write an answer/);
 await assert.rejects(openStudyStore(null), /cannot be saved/);
});
