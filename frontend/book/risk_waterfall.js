// frontend/book/risk_waterfall.js
export function renderChurnWaterfall(root, cfg) {
  if (!root || !cfg) return;
  const cap = (n) => Math.max(0, Math.min(cfg.cap_to ?? 100, Number(n) || 0));
  const totalTarget = cap(cfg.total_pct ?? 0);

  root.classList.add('nl-waterfall');
  root.innerHTML = `<div class="nl-tooltip" id="wf-tip"></div>`;
  const tip = root.querySelector('#wf-tip');

  const row = (label) => {
    const wrap = document.createElement('div');
    wrap.className = 'wf-row';
    wrap.innerHTML = `
      <div class="wf-label">${label}</div>
      <div class="wf-track-wrap"><div class="wf-track"></div></div>
    `;
    root.appendChild(wrap);
    return wrap.querySelector('.wf-track-wrap');
  };

  // Baseline
  let cumulative = cap(cfg.baseline_pp || 0);
  const baseWrap = row('Baseline');
  const base = document.createElement('div');
  base.className = 'wf-step baseline';
  base.style.left = `0%`;
  base.style.width = `${cumulative}%`;
  base.innerHTML = `<span class="pp">${cumulative}%</span>`;
  baseWrap.appendChild(base);

  // Drivers
  (cfg.drivers || []).forEach(d => {
    const wrap = row(d.label || 'Driver');
    const start = cumulative;
    const next = cap(cumulative + Number(d.pp || 0));
    const left = Math.min(start, next);
    const width = Math.abs(next - start);
    cumulative = next;

    const cls = (d.type === 'controllable') ? 'controllable'
               : (d.type === 'protective' || (Number(d.pp) < 0)) ? 'protective'
               : (d.type === 'residual') ? 'residual'
               : 'structural';

    const step = document.createElement('div');
    step.className = `wf-step ${cls}`;
    step.style.left = `${left}%`;
    step.style.width = `${width}%`;

    const ppTxt = `${Number(d.pp) >= 0 ? '+' : ''}${Number(d.pp)}pp`;
    const liftTxt = (d.lift_x && ((Number(d.pp) >= 0 && d.lift_x >= 1) || (Number(d.pp) < 0 && d.lift_x < 1)))
      ? `<span class="lift">×${Number(d.lift_x).toFixed(1)}</span>` : '';

    step.innerHTML = `${liftTxt}<span class="pp">${ppTxt}</span>`;
    step.dataset.tip = `${d.label}\nContribution: ${ppTxt}${d.why ? `\n${d.why}` : ''}`;
    step.addEventListener('mousemove', (e) => {
      const r = root.getBoundingClientRect();
      tip.textContent = step.dataset.tip;
      tip.style.display = 'block';
      tip.style.left = `${e.clientX - r.left + 10}px`;
      tip.style.top  = `${e.clientY - r.top + 10}px`;
    });
    step.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

    wrap.appendChild(step);
  });

  // Reconcile to header with residual if needed (±1pp)
  const allowResidual = Boolean(cfg.reconcile);  // default false
  const diff = Math.round(totalTarget - cumulative);
  if (allowResidual && Math.abs(diff) >= 1) {
    const wrap = row('Other modeled factors');
    const start = cumulative;
    const next = cap(cumulative + diff);
    const step = document.createElement('div');
    step.className = `wf-step residual`;
    step.style.left = `${Math.min(start, next)}%`;
    step.style.width = `${Math.abs(next - start)}%`;
    step.innerHTML = `<span class="pp">${diff >= 0 ? '+' : ''}${diff}pp</span>`;
    wrap.appendChild(step);
    cumulative = next;
  }

  // Footer
  const footer = document.createElement('div');
  footer.className = 'wf-total';
  footer.innerHTML = `
    <div class="legend">Red = controllable (fix). Amber = structural (plan). <span style="color:#15803d;font-weight:700">Green = protective</span>. Gray = residual.</div>
    <div class="val">${cumulative}%</div>
  `;
  root.appendChild(footer);
}