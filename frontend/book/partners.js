// ---- helpers ----------------------------------------------------
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const fmtMoney = (n=0) => {
  const v = Number(n||0);
  return `$${v.toLocaleString(undefined, {maximumFractionDigits:0})}`;
};
const perfClass = r => (r<=1.0 ? 'perf-good' : r<=1.2 ? 'perf-ok' : 'perf-bad');
const round2 = x => Number(x ?? 0).toFixed(2);

// ---- API calls --------------------------------------------------
const API_BASES = [
  'http://localhost:8001',    // Local development server
  window.location.origin,     // Current page origin (for production)
  'https://northlight-wsgw.onrender.com',
  'https://northlight-api-dev.onrender.com',
];

async function fetchFromAny(path) {
  for (const base of API_BASES) {
    try {
      const url = base + path;
      const r = await fetch(url);
      if (r.ok) return r;
    } catch (e) { continue; }
  }
  throw new Error('All API bases failed');
}

async function fetchPartners(playbook="seo_dash") {
  const r = await fetchFromAny(`/api/book/partners?playbook=${encodeURIComponent(playbook)}`);
  return r.json();
}
async function fetchPartnerDetail(name, playbook="seo_dash") {
  const r = await fetchFromAny(`/api/book/partners/${encodeURIComponent(name)}/opportunities?playbook=${encodeURIComponent(playbook)}`);
  return r.json();
}

// ---- renderers: card skeleton ----------------------------------
function renderPartnerCardShell(p) {
  const card = document.createElement('article');
  card.className = 'partner-card';
  card.setAttribute('data-partner', p.partner);

  card.innerHTML = `
    <div class="partner-header" role="button" tabindex="0" aria-expanded="false" aria-controls="detail-${cssId(p.partner)}">
      <div class="partner-name">${escapeHtml(p.partner)}</div>
      <div class="partner-budget">${fmtMoney(p.metrics.budget)} <span class="budget-label">monthly budget</span></div>
      <span class="caret">▸</span>
    </div>

    <div class="advertiser-breakdown">
      <div class="breakdown-items">
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-high"></span>Single Product</div>
          <div class="breakdown-count">${p.metrics.singleCount}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-medium"></span>Two Product</div>
          <div class="breakdown-count">${p.metrics.twoCount}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-label"><span class="risk-dot risk-low"></span>3+ Products</div>
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

    <section id="detail-${cssId(p.partner)}" class="detail-drop" hidden>
      <div class="panel-header">
        <div class="panel-title">${escapeHtml(p.partner)} — Action Plan</div>
        <div class="playbook-info" data-playbook>—</div>
      </div>

      <!-- Single Product -->
      <section class="action-group" data-group="single">
        <div class="group-header">
          <div class="group-title">Single Product Advertisers — Ready for Cross-Sell</div>
          <div class="group-count" data-count="single">0</div>
        </div>
        <div class="table-header advertiser">
          <div>Advertiser</div><div>Active Products</div><div>Performance</div><div>Action</div>
        </div>
        <div class="advertiser-table" data-table="single"></div>
      </section>

      <!-- Two Product -->
      <section class="action-group" data-group="two">
        <div class="group-header">
          <div class="group-title">Two Product Advertisers — Complete Bundle</div>
          <div class="group-count" data-count="two">0</div>
        </div>
        <div class="table-header advertiser">
          <div>Advertiser</div><div>Active Products</div><div>Performance</div><div>Action</div>
        </div>
        <div class="advertiser-table" data-table="two"></div>
      </section>

      <!-- Upsell -->
      <section class="action-group" data-group="upsell">
        <div class="group-header">
          <div class="group-title">Campaigns Ready for Budget Increase</div>
          <div class="group-count" data-count="upsell">0 campaigns</div>
        </div>
        <div class="table-header campaign">
          <div>Campaign</div><div>Current</div><div>Recommended</div><div>Performance</div><div>Action</div>
        </div>
        <div class="advertiser-table" data-table="upsell"></div>
      </section>

      <!-- Budget Inadequate -->
      <section class="action-group" data-group="toolow">
        <div class="group-header">
          <div class="group-title">Budget Inadequate — Raise to Minimum Viable</div>
          <div class="group-count" data-count="toolow">0 campaigns</div>
        </div>
        <div class="table-header campaign">
          <div>Campaign</div><div>Current</div><div>Recommended</div><div>Performance</div><div>Action</div>
        </div>
        <div class="advertiser-table" data-table="toolow"></div>
      </section>
    </section>
  `;

  return card;
}

// ---- detail row builders ----------------------------------------
function productBadges(products) {
  return `<div class="active-products">${
    (products||[]).map(p => `<span class="product-badge">${escapeHtml(p)}</span>`).join('')
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
  // a.advertiser is the advertiser display; Partner is the card
  return `
    <div class="advertiser-row">
      <div class="advertiser-info">
        <div class="advertiser-name">${escapeHtml(a.advertiser || a.name || '—')}</div>
        <div class="advertiser-meta">${fmtMoney(a.budget)}/mo • AM: ${escapeHtml(a.am ?? '—')} • ${a.months ?? 0} months</div>
      </div>
      ${productBadges(a.products)}
      ${perfBadge(a.cplRatio)}
      <div class="action-wrapper">
        <button class="action-btn">${escapeHtml(actionLabel)}</button>
        <span class="impact-text">${escapeHtml(actionSub)}</span>
      </div>
    </div>
  `;
}

function campaignRow(a, recommended, changeLabel, good) {
  // a.advertiser (advertiser) → a.name (campaign) → channel/cid in meta (hierarchy)
  const cid = escapeHtml(a.cid || deriveCidFromName(a.name));
  const meta = `${(a.products?.length ?? 0)} products • ${escapeHtml(a.channel || 'Search')} • CID: ${cid}`;
  return `
    <div class="campaign-row">
      <div class="advertiser-info">
        <div class="advertiser-name">${escapeHtml(a.advertiser || '—')} — ${escapeHtml(a.name || 'Campaign')}</div>
        <div class="advertiser-meta">${meta}</div>
      </div>
      <div class="budget-current">${fmtMoney(a.budget)}</div>
      <div class="budget-recommended">${fmtMoney(recommended)}<div class="budget-change">${escapeHtml(changeLabel)}</div></div>
      ${perfBadge(a.cplRatio)}
      <div class="action-wrapper">
        <button class="action-btn ${good ? 'success':''}">${good ? 'Increase Budget' : 'Fix Budget'}</button>
        <span class="impact-text">${good ? 'Room to scale' : 'Too low to work'}</span>
      </div>
    </div>
  `;
}

function deriveCidFromName(name) {
  const s = String(name || '');
  let h = 0;
  for (let i=0;i<s.length;i++) { h = ((h<<5)-h) + s.charCodeAt(i); h |= 0; }
  return String(Math.abs(h) % 90000 + 10000);
}

// ---- fill card detail -------------------------------------------
function missingOf(a, triadElements) {
  const missing = (triadElements || []).filter(t => !(a.products||[]).includes(t));
  return missing[0];
}

function fillCardDetail(cardEl, detail) {
  const drop = $(`.detail-drop`, cardEl);
  drop.hidden = false;
  drop.classList.add('show');

  $('[data-playbook]', cardEl).textContent = String(detail.playbook?.label || '—');

  // Single product → Cross-sell
  const single = detail.groups?.singleReady || [];
  $('[data-count="single"]', cardEl).textContent = `${single.length} of ${detail.counts?.single ?? 0} ready`;
  $('[data-table="single"]', cardEl).innerHTML = single.map(a =>
    advertiserRow(a, `Add ${missingOf(a, detail.playbook?.elements) || 'SEO'}`, '-25% churn risk')
  ).join('');

  // Two product → complete bundle
  const two = detail.groups?.twoReady || [];
  $('[data-count="two"]', cardEl).textContent = `${two.length} of ${detail.counts?.two ?? 0} ready`;
  $('[data-table="two"]', cardEl).innerHTML = two.map(a =>
    advertiserRow(a, `Add ${missingOf(a, detail.playbook?.elements) || 'DASH'}`, 'Reach 3+ products')
  ).join('');

  // Upsell (budget increase; SEM or any that qualify server-side)
  const upsell = detail.groups?.scaleReady || [];
  $('[data-count="upsell"]', cardEl).textContent = `${upsell.length} campaigns`;
  $('[data-table="upsell"]', cardEl).innerHTML = upsell.map(a =>
    campaignRow(a, Math.round((a.budget||0)*1.25), '+25%', true)
  ).join('');

  // Budget Inadequate (SEM viability)
  const tooLow = detail.groups?.tooLow || [];
  $('[data-count="toolow"]', cardEl).textContent = `${tooLow.length} campaigns`;
  $('[data-table="toolow"]', cardEl).innerHTML = tooLow.map(a => {
    const rec = Number(a.recommended_budget || detail.playbook?.min_sem || 2500);
    const change = rec > (a.budget||0) ? `min ${fmtMoney(rec)}` : `check settings`;
    return campaignRow(a, Math.round(rec), change, false);
  }).join('');
}

// ---- utilities --------------------------------------------------
function cssId(s){ return String(s||'').toLowerCase().replace(/\s+/g,'-').replace(/[^a-z0-9\-]/g,''); }
function escapeHtml(s){
  return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

// ---- boot -------------------------------------------------------
(async function boot() {
  const grid = $('#partners-grid');

  // Load partner cards
  let partners = [];
  try {
    partners = await fetchPartners("seo_dash");
  } catch (e) {
    grid.innerHTML = `<div style="padding:12px;color:#dc3545">Error loading partners: ${escapeHtml(e.message||e)}</div>`;
    return;
  }
  if (!partners.length) {
    grid.innerHTML = `<div style="padding:12px;">No partners found.</div>`;
    return;
  }

  // Render cards
  partners.forEach((p) => {
    const card = renderPartnerCardShell(p);
    grid.appendChild(card);

    // Toggle + lazy load detail
    const header = $('.partner-header', card);
    const caret  = $('.caret', card);
    const drop   = $('.detail-drop', card);
    let loaded   = false;

    const toggle = async () => {
      const isOpen = drop.classList.contains('show');
      if (isOpen) {
        drop.classList.remove('show');
        drop.hidden = true;
        card.classList.remove('selected');
        header.setAttribute('aria-expanded', 'false');
        caret.classList.remove('open');
        return;
      }
      // open
      card.classList.add('selected');
      caret.classList.add('open');
      header.setAttribute('aria-expanded', 'true');

      if (!loaded) {
        try {
          const d = await fetchPartnerDetail(p.partner, "seo_dash");
          fillCardDetail(card, d);
          loaded = true;
        } catch (e) {
          drop.hidden = false;
          drop.classList.add('show');
          drop.innerHTML = `<div style="padding:16px;color:#dc3545">Error loading details: ${escapeHtml(e.message||e)}</div>`;
        }
      } else {
        drop.hidden = false;
        drop.classList.add('show');
      }
    };

    header.addEventListener('click', toggle);
    header.addEventListener('keydown', (e)=>{ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); toggle(); }});
  });
})();
