(async () => {
  const load = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error("Demo data unavailable");
    return response.json();
  };
  const [model, results] = await Promise.all([
    load("model.json"),
    load("results.json"),
  ]);

  const MIN_H = 15 / (24 * 60), MAX_H = 274;
  const TARGET = 0.9; // review when recall is predicted to have fallen to this
  const el = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const days = (d) => {
    if (d < 1 / 24) return `${Math.round(d * 24 * 60)} minutes`;
    if (d < 1) return `${(d * 24).toFixed(1)} hours`;
    if (d < 60) return `${d.toFixed(1)} days`;
    if (d < 730) return `${(d / 30.4).toFixed(1)} months`;
    return `${(d / 365).toFixed(1)} years`;
  };

  const halfLife = (right, wrong, lexWeight) => {
    const dot =
      model.right * Math.sqrt(1 + right) +
      model.wrong * Math.sqrt(1 + wrong) +
      model.bias +
      lexWeight;
    return clamp(Math.pow(2, clamp(dot, -20, 20)), MIN_H, MAX_H);
  };

  const leitnerWait = (right, wrong) =>
    clamp(Math.pow(2, clamp(right - wrong, -20, 20)), MIN_H, MAX_H);

  const recall = (t, h) => Math.pow(2, -t / h);

  el("nReviews").textContent = (results.train_reviews / 1e6).toFixed(1) + " million";
  el("nTest").textContent = (results.test_reviews / 1e6).toFixed(1) + " million";

  const items = model.lexemes.filter((l) => l.text && /^[\p{L}'-]+$/u.test(l.text)).slice(0, 220);
  el("item").innerHTML = items
    .map((l, i) => `<option value="${i}">${l.text} — ${l.reviews.toLocaleString()} reviews</option>`)
    .join("");

  function draw(h, leitner, suggested) {
    const W = 720, H = 270, P = { l: 46, r: 14, t: 14, b: 30 };
    const span = Math.max(leitner, suggested, h) * 1.6;
    const x = (t) => P.l + (t / span) * (W - P.l - P.r);
    const y = (p) => H - P.b - p * (H - P.t - P.b);

    let curve = "";
    for (let i = 0; i <= 240; i++) {
      const t = (i / 240) * span;
      curve += `${i ? "L" : "M"}${x(t).toFixed(1)},${y(recall(t, h)).toFixed(1)}`;
    }
    const grid = [0, 0.25, 0.5, 0.75, 1]
      .map((p) => `<line x1="${P.l}" y1="${y(p)}" x2="${W - P.r}" y2="${y(p)}"
           stroke="currentColor" opacity=".13"/>
        <text x="${P.l - 8}" y="${y(p) + 4}" font-size="11" fill="currentColor" opacity=".55"
           text-anchor="end">${p * 100}%</text>`)
      .join("");
    const mark = (t, color, label) => {
      if (t > span) return "";
      const p = recall(t, h);
      return `<line x1="${x(t)}" y1="${P.t}" x2="${x(t)}" y2="${H - P.b}" stroke="${color}"
                stroke-width="1.5" stroke-dasharray="4 3"/>
              <circle cx="${x(t)}" cy="${y(p)}" r="5" fill="${color}"/>
              <text x="${clamp(x(t), P.l + 4, W - 120)}" y="${y(p) - 12}" font-size="12"
                fill="${color}" font-weight="600">${label} · ${(p * 100).toFixed(0)}%</text>`;
    };
    const css = getComputedStyle(document.documentElement);
    el("chart").innerHTML =
      grid +
      `<path d="${curve}" fill="none" stroke="${css.getPropertyValue("--learned")}" stroke-width="2.5"/>` +
      mark(suggested, css.getPropertyValue("--warn").trim(), "model reviews") +
      mark(leitner, css.getPropertyValue("--fixed").trim(), "Leitner reviews") +
      `<text x="${P.l}" y="${H - 8}" font-size="11" fill="currentColor" opacity=".55">now</text>
       <text x="${W - P.r}" y="${H - 8}" font-size="11" fill="currentColor" opacity=".55"
         text-anchor="end">${days(span)}</text>`;
  }

  function render() {
    const right = +el("right").value, wrong = +el("wrong").value;
    el("rightVal").textContent = right;
    el("wrongVal").textContent = wrong;

    const item = items[+el("item").value] || { weight: 0 };
    const h = halfLife(right, wrong, item.weight);
    const leitner = leitnerWait(right, wrong);
    const suggested = -h * Math.log2(TARGET);

    el("hlrHalf").textContent = days(h);
    el("hlrNote").textContent =
      `model predicts ${TARGET * 100}% recall after ${days(suggested)}`;

    const pAtLeitner = recall(leitner, h) * 100;
    el("leitnerWait").textContent = days(leitner);
    el("leitnerNote").textContent =
      `model predicts ${pAtLeitner.toFixed(0)}% recall at that interval`;

    draw(h, leitner, suggested);
  }

  const NAMES = { hlr: "Half-life regression (learned)", leitner: "Leitner (fixed doubling)",
                  pimsleur: "Pimsleur (fixed exponential)", constant: "Always predict the average" };
  const rows = Object.entries(results.models).sort((a, b) => a[1].mae_recall - b[1].mae_recall);
  el("results").innerHTML =
    `<tr><th>scheduler</th><th>mean abs. error on recall</th><th>AUC</th><th>half-life error</th></tr>` +
    rows.map(([key, m], i) =>
        `<tr class="${i === 0 ? "best" : ""}"><td>${NAMES[key] || key}</td>
          <td class="num">${m.mae_recall.toFixed(4)}</td>
          <td class="num">${Number.isFinite(m.auc) ? m.auc.toFixed(3) : "—"}</td>
          <td class="num">${m.mae_half_life_days == null ? "—" : m.mae_half_life_days.toFixed(0) + " d"}</td></tr>`)
      .join("");

  ["item", "right", "wrong"].forEach((id) => (el(id).oninput = render));
  render();
  ["item", "right", "wrong"].forEach((id) => (el(id).disabled = false));
  el("loadStatus").textContent = "Model and held-out results loaded.";
})().catch(() => {
  document.getElementById("loadStatus").textContent =
    "Could not load the model and results. Reload this page to try again; no prediction is available.";
  ["item", "right", "wrong"].forEach((id) => (document.getElementById(id).disabled = true));
  ["chart", "results", "hlrNote", "leitnerNote"].forEach((id) => (document.getElementById(id).innerHTML = ""));
  ["hlrHalf", "leitnerWait"].forEach((id) => (document.getElementById(id).textContent = "—"));
});
