/* InsightAI — client logic. Vanilla JS, no dependencies.
   The server is the source of truth for validation; this file mirrors it
   so people get instant feedback. All text from the server is inserted with
   textContent, never innerHTML. */
(() => {
  'use strict';

  const data = JSON.parse(document.getElementById('app-data').textContent);
  const SCHEMA = data.schema;               // exact training order, from the model
  const TYPICAL = data.typical;
  const MAX_WEEK_HOURS = 168;
  const CIRC = 2 * Math.PI * 84;            // ring circumference (r = 84)

  const $ = (id) => document.getElementById(id);
  const form = $('predict-form');
  const panel = $('result-panel');
  const ring = $('ring');
  const arc = $('ring-arc');
  const ringValue = $('ring-value');
  const ringCaption = $('ring-caption');
  const title = $('result-title');
  const kicker = $('stage-kicker');
  const stageText = $('stage-text');
  const probList = $('prob-list');
  const driverList = $('driver-list');
  const driverNote = $('drv-note');
  const live = $('live');
  const banner = $('form-banner');
  const submitBtn = $('submit-btn');
  const submitLabel = submitBtn.querySelector('.btn__label');

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let controller = null;
  let rafId = 0;

  /* ---------- Reading and validating the form ---------- */
  const rawValue = (name) => {
    const el = form.elements[name];
    return el ? String(el.value ?? '').trim() : '';
  };

  const fmt = (n) => (Number.isInteger(n) ? String(n) : String(+n.toFixed(4)));

  function checkField(f) {
    const v = rawValue(f.name);
    if (f.kind !== 'numeric') {
      if (v === '') return 'Choose an option.';
      return f.options.some((o) => o.value === v) ? null : 'Choose one of the listed options.';
    }
    if (v === '') return f.widget === 'scale' ? `Choose a rating from ${f.min} to ${f.max}.` : 'Enter a value.';
    const x = Number(v);
    if (!Number.isFinite(x)) return 'Enter a number.';
    if (f.integer && !Number.isInteger(x)) return 'Enter a whole number.';
    if (x < f.min || x > f.max) return `Must be between ${fmt(f.min)} and ${fmt(f.max)}.`;
    return null;
  }

  function collectErrors() {
    const errors = {};
    SCHEMA.forEach((f) => { const e = checkField(f); if (e) errors[f.name] = e; });
    const a = Number(rawValue('Weekly_GenAI_Hours'));
    const b = Number(rawValue('Traditional_Study_Hours'));
    if (!errors.Weekly_GenAI_Hours && !errors.Traditional_Study_Hours && a + b > MAX_WEEK_HOURS) {
      errors.Weekly_GenAI_Hours = `GenAI and traditional study hours together exceed ${MAX_WEEK_HOURS} hours in a week.`;
    }
    return errors;
  }

  function buildPayload() {
    const payload = {};
    SCHEMA.forEach((f) => {
      const v = rawValue(f.name);
      payload[f.name] = f.kind === 'numeric' ? Number(v) : f.kind === 'boolean' ? v === 'true' : v;
    });
    return payload;
  }

  /* ---------- Error display ---------- */
  function fieldEl(name) { return form.querySelector(`[data-field="${name}"]`); }

  function setFieldError(name, message) {
    const wrap = fieldEl(name);
    if (!wrap) return;
    const out = $(`err-${name}`);
    wrap.classList.toggle('field--invalid', Boolean(message));
    wrap.querySelectorAll('input, select').forEach((el) => {
      if (message) el.setAttribute('aria-invalid', 'true'); else el.removeAttribute('aria-invalid');
    });
    out.textContent = message || '';
    out.hidden = !message;
  }

  function showErrors(errors, summary) {
    SCHEMA.forEach((f) => setFieldError(f.name, errors[f.name] || ''));
    const names = Object.keys(errors);
    if (!names.length) { banner.hidden = true; return; }
    const n = names.length;
    banner.textContent = summary || `Fix ${n} field${n === 1 ? '' : 's'} and try again.`;
    banner.hidden = false;
    const first = SCHEMA.find((f) => errors[f.name]);
    const wrap = first && fieldEl(first.name);
    const focusable = wrap && wrap.querySelector('input, select');
    if (focusable) {
      wrap.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
      focusable.focus({ preventScroll: true });
    }
  }

  function showBanner(message) { banner.textContent = message; banner.hidden = false; banner.scrollIntoView({ block: 'nearest' }); }

  /* ---------- Result panel states ---------- */
  function setState(state) {
    panel.dataset.state = state;
    ring.classList.toggle('is-loading', state === 'loading');
  }

  function resetResult() {
    cancelAnimationFrame(rafId);
    delete panel.dataset.risk;
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = String(CIRC);
    ringValue.textContent = '—';
    ringCaption.textContent = 'probability';
    kicker.textContent = 'Predicted burnout risk';
    title.textContent = 'Your prediction';
    stageText.hidden = false;
    stageText.textContent = 'Complete all 14 fields, then run the prediction to see the risk class and how likely each class is.';
    probList.replaceChildren();
    driverList.replaceChildren();
    setState('idle');
  }

  function countUp(el, target, ms) {
    cancelAnimationFrame(rafId);
    if (reduceMotion) { el.textContent = `${target.toFixed(1)}%`; return; }
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = `${(target * eased).toFixed(1)}%`;
      if (t < 1) rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  const pct = (p) => `${(p * 100).toFixed(1)}%`;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderResult(res) {
    const risk = res.prediction.toLowerCase();
    panel.dataset.risk = risk;
    setState('done');

    kicker.textContent = 'Predicted burnout risk';
    title.textContent = `${res.prediction} risk`;
    stageText.textContent = res.summary;
    ringCaption.textContent = 'probability';

    // Ring: set the start state with no transition, then animate on the next frame.
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = String(CIRC);
    void arc.getBoundingClientRect();
    arc.style.transition = reduceMotion ? 'none' : 'stroke-dashoffset 1.2s cubic-bezier(0.2, 0.8, 0.2, 1), stroke 0.4s';
    arc.style.strokeDashoffset = String(CIRC * (1 - res.probability));
    countUp(ringValue, res.probability * 100, 1200);

    // Probabilities
    probList.replaceChildren();
    const fills = [];
    res.probabilities.forEach((p) => {
      const li = el('li', `prob${p.predicted ? ' prob--top' : ''}`);
      const top = el('div', 'prob__top');
      const name = el('span', 'prob__name', p.label);
      if (p.predicted) name.append(el('span', 'prob__chip', 'Predicted'));
      top.append(name, el('span', 'prob__pct', pct(p.value)));
      const track = el('div', 'prob__track');
      const fill = el('div', 'prob__fill');
      track.append(fill);
      li.append(top, track);
      probList.append(li);
      fills.push([fill, p.value]);
    });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fills.forEach(([fill, v]) => { fill.style.width = `${(v * 100).toFixed(2)}%`; });
    }));

    // Drivers
    driverList.replaceChildren();
    driverNote.textContent = res.drivers.length
      ? `Compared with the other two classes. Bars to the right pushed toward ${res.prediction}; bars to the left pulled away from it.`
      : '';
    const maxAbs = Math.max(...res.drivers.map((d) => Math.abs(d.contribution)), 1e-9);
    const driverFills = [];
    res.drivers.forEach((d) => {
      const li = el('li', `drv drv--${d.direction}`);
      const top = el('div', 'drv__top');
      top.append(el('span', 'drv__name', d.label), el('span', 'drv__note', d.note));
      const track = el('div', 'drv__track');
      const fill = el('div', 'drv__fill');
      track.append(fill);
      const sr = el('span', 'sr-only',
        d.direction === 'toward' ? `Pushed toward ${res.prediction}.` : `Pulled away from ${res.prediction}.`);
      li.append(top, track, sr);
      driverList.append(li);
      driverFills.push([fill, (Math.abs(d.contribution) / maxAbs) * 50]);
    });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      driverFills.forEach(([fill, w]) => { fill.style.width = `${w.toFixed(2)}%`; });
    }));

    // Announce and move focus to the result
    const top = res.probabilities.find((p) => p.predicted);
    live.textContent = `Predicted burnout risk: ${res.prediction}, probability ${pct(top ? top.value : res.probability)}.`;
    title.focus({ preventScroll: true });
    // Bring the result into view if the submit button sat far below it.
    const panelTop = panel.getBoundingClientRect().top;
    if (window.matchMedia('(max-width: 1040px)').matches || panelTop < 0 || panelTop > window.innerHeight * 0.6) {
      panel.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    }
  }

  /* ---------- Submit ---------- */
  function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.classList.toggle('is-loading', busy);
    submitBtn.setAttribute('aria-busy', String(busy));
    submitLabel.textContent = busy ? 'Scoring profile' : 'Predict burnout risk';
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const errors = collectErrors();
    showErrors(errors);
    if (Object.keys(errors).length) return;

    if (controller) controller.abort();
    controller = new AbortController();
    setBusy(true);
    cancelAnimationFrame(rafId);
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = '0';
    ringValue.textContent = '…';
    title.textContent = 'Scoring profile';
    stageText.textContent = 'Running your inputs through the model.';
    setState('loading');

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(buildPayload()),
        signal: controller.signal,
      });
      let body = null;
      try { body = await resp.json(); } catch (_) { /* non-JSON error page */ }

      if (resp.ok && body && body.prediction) {
        renderResult(body);
        return;
      }
      resetResult();
      if (resp.status === 422 && body && body.fields) {
        showErrors(body.fields, body.message);
      } else {
        showBanner((body && body.message) || 'The prediction failed. Please try again.');
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      resetResult();
      showBanner('Could not reach the server. Check your connection and try again.');
    } finally {
      setBusy(false);
    }
  });

  /* ---------- Helpers: typical profile, reset, live error clearing ---------- */
  $('fill-btn').addEventListener('click', () => {
    SCHEMA.forEach((f) => {
      const value = TYPICAL[f.name];
      if (value === undefined || value === null) return;
      const control = form.elements[f.name];
      if (!control) return;
      if (f.widget === 'segmented' || f.widget === 'scale') {
        const radio = form.querySelector(`input[name="${f.name}"][value="${String(value)}"]`);
        if (radio) radio.checked = true;
      } else {
        control.value = String(value);
      }
    });
    showErrors({});
  });

  form.addEventListener('reset', () => {
    showErrors({});
    resetResult();
    live.textContent = '';
  });

  const clearOne = (e) => {
    const wrap = e.target.closest('[data-field]');
    if (wrap) setFieldError(wrap.dataset.field, '');
  };
  form.addEventListener('input', clearOne);
  form.addEventListener('change', clearOne);

  resetResult();
})();
