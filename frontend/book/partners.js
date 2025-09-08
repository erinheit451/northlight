// ---- helpers ----------------------------------------------------
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const fmtMoney = (n=0) => {
  const v = Number(n||0);
  return `${v.toLocaleString(undefined, {maximumFractionDigits:0})}`;
};
const perfClass = r => (r<=1.0 ? 'perf-good' : r<=1.2 ? 'perf-ok' : 'perf-bad');
const round2 = x => Number(x ?? 0).toFixed(2);

// ---- API calls --------------------------------------------------
async function fetchPartners(playbook="seo_dash") {
  const r = await fetch(`/api/book/partners?playbook=${encodeURIComponent(playbook)}`);
  if (!r.ok) throw new Error(`Failed to load partners: ${r.status}`);
  return r.json();
}
async function fetchPartnerDetail(name, playbook="seo_dash") {
  const r = await fetch(`/api/book/partners/${encodeURIComponent(name)}/opportunities?playbook=${encodeURIComponent(playbook)}`);
  if (!r.ok) throw new Error(`Failed to load partner ${name}: ${r.status}`);
  return r.json();
}

// ---- renderers: cards -------------------------------------------
function renderPartnerCard(p, isSelected) {
  const el = document.createElement('div');
  el.className = `partner-card ${isSelected ? 'selected':''}`;
  el.innerHTML = `
    <div class="partner-header">
      <div class="partner-name">${p.partner}</div>
      <div class="partner-budget">${fmtMoney(p.metrics.budget)} <span class="budget-label">monthly budget</span></div>
    </div>
    <div class="advertiser-breakdown">
      <div class="breakdown-title">Advertiser Distribution</div>
      <div class="breakdown-items">
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-high"></span>Single Product Advertisers</div>
          <div class="breakdown-count">${p.metrics.singleCount}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-medium"></span>Two Product Advertisers</div>
          <div class="breakdown-count">${p.metrics.twoCount}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-low"></span>3+ Product Advertisers</div>
          <div class="breakdown-count">${p.metrics.threePlusCount}</div>
        </div>
      </div>
    </div>
    <div class="opportunities">
      <div class="opportunity-card primary">
        <div class="opp-number primary">${p.metrics.crossReadyCount}</div>
        <div class="opp-label">Ready to Cross-Sell</div>
      </div>
      <div class="opportunity-card">
        <div class="opp-number success">${p.metrics.upsellReadyCount}</div>
        <div class="opp-label">Ready to Upsell</div>
      </div>
    </div>
  `;
  return el;
}

// ---- renderers: detail rows -------------------------------------
function productBadges(products) {
  return `<div class="active-products">${
    (products||[]).map(p => `<span class="product-badge">${p}</span>`).join('')
  }</div>`;
}
function perfBadge(ratio) {
  const cls = perfClass(ratio ?? 2);
  const text = `${round2(ratio ?? 2)}× goal`;
  return `
    <div class="performance-metric">
      <div class="perf-value ${cls}">${text}</div>
      <div class="perf-label">CPL vs Goal</div>
    </div>
  `;
}
function advertiserRow(a, actionLabel, actionSub) {
  return `
    <div class="advertiser-row">
      <div class="advertiser-info">
        <div class="advertiser-name">${a.name}</div>
        <div class="advertiser-meta">${fmtMoney(a.budget)}/mo • AM: ${a.am ?? '—'} • ${a.months ?? 0} months</div>
      </div>
      ${productBadges(a.products)}
      ${perfBadge(a.cplRatio)}
      <div class="action-wrapper">
        <button class="action-btn">${actionLabel}</button>
        <span class="impact-text">${actionSub}</span>
      </div>
    </div>
  `;
}
function campaignRow(a, recommended, changeLabel, good) {
  const cid = (Math.abs(String(a.name).split('').reduce((h,c)=>(h<<5)-h+c.charCodeAt(0),0)) % 90000) + 10000;
  return `
    <div class="campaign-row">
      <div class="advertiser-info">
        <div class="advertiser-name">${a.name} — Search</div>
        <div class="advertiser-meta">${a.products?.length ?? 0} products active • CID: ${cid}</div>
      </div>
      <div class="budget-current">${fmtMoney(a.budget)}</div>
      <div class="budget-recommended">${fmtMoney(recommended)}<div class="budget-change">${changeLabel}</div></div>
      ${perfBadge(a.cplRatio)}
      <div class="action-wrapper">
        <button class="action-btn ${good ? 'success':''}">${good ? 'Increase Budget' : 'Fix Budget'}</button>
        <span class="impact-text">${good ? 'Room to scale' : 'Too low to work'}</span>
      </div>
    </div>
  `;
}

// ---- detail panel population ------------------------------------
function fillDetailPanel(detail) {
  $('#detail-panel').hidden = false;
  $('#panel-title').textContent = `${detail.partner} — Action Plan`;
  $('#panel-playbook').textContent = `${detail.playbook.label}`;

  // Single product → Cross-sell
  const single = detail.groups.singleReady || [];
  $('#count-single').textContent = `${single.length} of ${detail.counts.single} ready`;
  $('#table-single').innerHTML = single.map(a =>
    advertiserRow(a, `Add ${missingOf(a, detail.playbook.elements) || 'SEO'}`, '-25% churn risk')
  ).join('');

  // Two product → complete bundle
  const two = detail.groups.twoReady || [];
  $('#count-two').textContent = `${two.length} of ${detail.counts.two} ready`;
  $('#table-two').innerHTML = two.map(a =>
    advertiserRow(a, `Add ${missingOf(a, detail.playbook.elements) || 'DASH'}`, 'Reach 3+ products')
  ).join('');

  // Upsell (budget increase)
  const upsell = detail.groups.scaleReady || [];
  $('#count-upsell').textContent = `${upsell.length} campaigns`;
  $('#table-upsell').innerHTML = upsell.map(a =>
    campaignRow(a, Math.round((a.budget||0)*1.25), '+25%', true)
  ).join('');

  // Too low (set to minimum)
  const tooLow = detail.groups.tooLow || [];
  $('#count-toolow').textContent = `${tooLow.length} campaigns`;
  $('#table-toolow').innerHTML = tooLow.map(a =>
    campaignRow(a, detail.playbook.min_sem || 2500, `min ${fmtMoney(detail.playbook.min_sem || 2500)}`, false)
  ).join('');
}

function missingOf(a, triadElements) {
  const missing = (triadElements || []).filter(t => !(a.products||[]).includes(t));
  return missing[0];
}

// ---- boot -------------------------------------------------------
(async function boot() {
  const grid = $('#partners-grid');

  // 1) Load partner cards
  let partners = [];
  try {
    partners = await fetchPartners("seo_dash");
  } catch (e) {
    grid.innerHTML = `<div style="padding:12px;color:#dc3545">Error loading partners: ${String(e.message||e)}</div>`;
    return;
  }
  if (!partners.length) {
    grid.innerHTML = `<div style="padding:12px;">No partners found.</div>`;
    return;
  }

  // 2) Render cards and hook click -> load detail
  partners.forEach((p, i) => {
    const card = renderPartnerCard(p, i===0);
    card.addEventListener('click', async () => {
      $$('.partner-card', grid).forEach(el => el.classList.remove('selected'));
      card.classList.add('selected');
      await loadDetail(p.partner);
    });
    grid.appendChild(card);
  });

  // 3) Load first partner's details by default
  await loadDetail(partners[0].partner);

  async function loadDetail(name) {
    $('#detail-panel').hidden = true;
    // optional: show a tiny skeleton
    const panel = $('#detail-panel');
    panel.hidden = false;
    $('#panel-title').textContent = `${name} — Loading…`;
    $('#panel-playbook').textContent = '';

    try {
      const d = await fetchPartnerDetail(name, "seo_dash");
      fillDetailPanel(d);
    } catch (e) {
      $('#panel-title').textContent = `${name}`;
      $('#panel-playbook').textContent = '';
      $('#group-single').outerHTML = `<div style="padding:16px;color:#dc3545">Error loading details: ${String(e.message||e)}</div>`;
    }
  }
})();