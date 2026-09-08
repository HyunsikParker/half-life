import { MAX_CARDS, validateCard, recordReview } from './study-core.mjs?v=1';

export function openStudyStore(factory = globalThis.indexedDB) {
  if (!factory) return Promise.reject(new Error('Browser storage is unavailable. Cards cannot be saved.'));
  return new Promise((resolve, reject) => {
    const request = factory.open('half-life-study-v1', 1);
    let abandoned = false;
    request.onupgradeneeded = () => request.result.createObjectStore('cards', { keyPath: 'id' });
    request.onerror = () => reject(new Error('Could not open browser storage. No card was saved.'));
    request.onblocked = () => { abandoned = true; reject(new Error('Close other Half-Life tabs, then reload to open storage.')); };
    request.onsuccess = () => {
      const db = request.result;
      if (abandoned) { db.close(); return; }
      db.onversionchange = () => db.close();
      const transact = (mode, operation) => new Promise((resolveDone, rejectDone) => {
        let result, failure;
        let transaction;
        try { transaction = db.transaction('cards', mode); }
        catch { rejectDone(new Error('Browser storage is unavailable. This change was not saved; reload to try again.')); return; }
        const store = transaction.objectStore('cards');
        transaction.oncomplete = () => resolveDone(result);
        transaction.onabort = () => rejectDone(failure ?? new Error('Could not save in this browser. The change was not recorded.'));
        transaction.onerror = () => { failure ??= new Error('Browser storage failed. The change was not recorded.'); };
        const step = callback => {
          try { callback(); } catch (error) { failure = error; transaction.abort(); }
        };
        step(() => operation(store, value => { result = value; }, step));
      });
      resolve(Object.freeze({
        list: () => transact('readonly', (store, done, step) => {
          const request = store.getAll();
          request.onsuccess = () => step(() => done(request.result.map(validateCard)));
        }),
        add: card => transact('readwrite', (store, done, step) => {
          validateCard(card);
          const request = store.count();
          request.onsuccess = () => step(() => {
            if (request.result >= MAX_CARDS) throw new Error('This browser has 100 cards. Remove a card before adding another.');
            store.add(card);
            done(card);
          });
        }),
        review: (id, event) => transact('readwrite', (store, done, step) => {
          const request = store.get(id);
          request.onsuccess = () => step(() => {
            if (!request.result) throw new Error('This card was removed in another tab. Reload your cards.');
            const updated = recordReview(request.result, event);
            store.put(updated);
            done(updated);
          });
        }),
        remove: id => transact('readwrite', store => { store.delete(id); }),
        close: () => db.close(),
      }));
    };
  });
}
