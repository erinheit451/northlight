/* ===========================
   ENV + STORAGE
   =========================== */
const PROD_API = 'https://northlight-wsgw.onrender.com';
const DEV_API  = 'https://northlight-api-dev.onrender.com';
const host = window.location.hostname;
// localhost / develop / PR previews treated as dev
const isDev = host.includes('localhost') || host.includes('127.0.0.1') || host.includes('develop') || host.includes('pr-');
const API = isDev ? DEV_API : PROD_API;

const STORAGE_KEY = 'northlight_inputs_v1';

console.log(`API endpoint set to: ${API}`);

/* ===========================
   UTILITIES
   =========================== */
function fmtPct1(v) { return (v == null) ? "—" : (v*100).toFixed(1) + "%"; }
function fmtMoney(v){ return (v == null) ? "—" : "$" + Number(v).toFixed(2); }
function pill(text, cls) { return `<span class="pill ${cls}">${text}</span>`; }

function ensureTargetRange(metricData, unit, metricName) {
  if (metricData?.target_range?.low != null && metricData?.target_range?.high != null) {
    return metricData.target_range;
  }
  const m = metricData?.median ?? metricData?.value ?? (unit === '%' ? 0.05 : (metricName === 'CPC' ? 3 : 50));
  const p25 = metricData?.p25, p75 = metricData?.p75;
  if (p25 != null && p75 != null) return { low: Math.max(0, p25), high: Math.max(p75, p25) };
  const span = Math.max(0.00001, m * 0.15); // ±15%
  return { low: Math.max(0, m - span), high: m + span };
}

function deriveVerdict(value, targetRange, direction) {
  if (value == null || targetRange?.low == null || targetRange?.high == null) return 'unknown';
  const { low, high } = targetRange;
  if (value >= low && value <= high) return 'on_target';
  if (direction === 'lower-is-better') {
    return (value < low) ? 'exceeds_target' : 'outside_target';
  } else {
    return (value > high) ? 'exceeds_target' : 'outside_target';
  }
}

function deltaPctFromRange(value, targetRange) {
  const { low, high } = targetRange;
  if (value < low) return ((low - value) / Math.max(low, 1e-9)) * 100;
  if (value > high) return ((value - high) / Math.max(high, 1e-9)) * 100;
  return 0;
}

function roundUpToNiceCurrency(x) {
  const targets = [0.5, 1, 2, 3, 5, 7.5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000];
  const want = x * 1.15;
  for (const t of targets) if (want <= t) return t;
  return Math.ceil(want / 100) * 100;
}

function roundUpToNicePercent(x) {
  const targets = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00];
  const want = x * 1.15;
  for (const t of targets) if (want <= t) return t;
  return Math.min(1.00, Math.ceil(want * 20) / 20);
}

/* ===========================
   BENCHMARK RENDERING
   =========================== */
function renderVerdictBenchmark(containerEl, metricData, metricName, metricUnit, direction, opts = {}) {
  const { hidePeer = false } = opts;
  const value = metricData?.value;

  if (value == null) {
    containerEl.innerHTML = `<div class="benchmark-verdict" style="padding: 12px;">No data for ${metricName}</div>`;
    return;
  }

  const target_range = ensureTargetRange(metricData, metricUnit === '%' ? '%' : '$', metricName);
  const localVerdict = deriveVerdict(value, target_range, direction);
  const rawDelta = (typeof metricData?.delta_from_target === 'number') ? Math.abs(metricData.delta_from_target) : deltaPctFromRange(value, target_range);
  const median = metricData?.median;
  const peer_multiple = metricData?.peer_multiple;

  const verdictLabels = { "outside_target": "Outside Target", "on_target": "On Target", "exceeds_target": "Excellent", "unknown": "Benchmark" };
  const verdictIcons  = { "outside_target": "⚠️", "on_target": "🎯", "exceeds_target": "🎉", "unknown": "📊" };

  const valueFormatter = metricUnit === '$' ? fmtMoney : fmtPct1;
  const primaryValue   = valueFormatter(value);
  const verdictClass   = (localVerdict || "").replace(/_/g, "-");

  let deltaText = "";
  if (target_range) {
    const below = value < target_range.low;
    const above = value > target_range.high;
    if (below || above) {
      const isGood = (direction === 'lower-is-better' && below) || (direction === 'higher-is-better' && above);
      const deltaClass = isGood ? 'negative' : 'positive';
      const rangeText = `(${valueFormatter(target_range.low)}–${valueFormatter(target_range.high)})`;
      const deltaDescription = below ? "below target range" : "above target range";
      deltaText = `<span class="delta-text ${deltaClass}">${Math.round(Math.abs(rawDelta))}% ${deltaDescription} ${rangeText}</span>`;
    } else {
      deltaText = `<span class="delta-text">within target range (${valueFormatter(target_range.low)}–${valueFormatter(target_range.high)})</span>`;
    }
  }

  const peerText = (!hidePeer && peer_multiple && median) ? `<span class="peer-context">${peer_multiple.toFixed(1)}× peer median</span>` : "";
  const barData = calculateBarData(value, target_range, median, direction, metricUnit, metricName);

  containerEl.innerHTML = `
    <div class="benchmark-verdict ${verdictClass}">
      <div class="status-header ${verdictClass}">
        ${verdictIcons[localVerdict] || "📊"} ${verdictLabels[localVerdict] || "Benchmark"}
      </div>
      <div class="primary-value">Your ${metricName} is ${primaryValue}</div>
      ${(deltaText || peerText) ? `<div class="verdict-context">${deltaText} ${peerText}</div>` : ''}
      <div class="bar-container">${generateBar(barData)}</div>
    </div>`;
}

function calculateBarData(userValue, targetRange, median, direction, metricUnit, metricName) {
  if (!targetRange || targetRange.low == null || targetRange.high == null) {
    targetRange = ensureTargetRange({ value: userValue, median }, metricUnit, metricName);
  }

  const baseTop = Math.max( userValue ?? 0, targetRange.high ?? 0, median ?? 0 );
  let minScale = 0, maxScale;
  if (metricUnit === '%') {
    maxScale = roundUpToNicePercent(baseTop);
  } else {
    if (metricName === 'CPC') {
      maxScale = roundUpToNiceCurrency(baseTop);
    } else {
      maxScale = Math.max(baseTop * 1.4, targetRange.high * 1.6, (median || 0) * 1.5);
      maxScale = roundUpToNiceCurrency(maxScale);
    }
  }

  const totalRange = maxScale - minScale;
  if (totalRange <= 0) return null;

  let userSection = 'target-range';
  if (userValue < targetRange.low) {
    userSection = (direction === 'lower-is-better') ? 'excellent' : 'needs-improvement';
  } else if (userValue > targetRange.high) {
    userSection = (direction === 'lower-is-better') ? 'needs-improvement' : 'excellent';
  }

  const valueFormatter = metricUnit === '$' ? fmtMoney : fmtPct1;
  const clamp01 = v => Math.max(0, Math.min(1, v));
  const userPosition = clamp01((userValue - minScale) / totalRange) * 100;

  let excellentWidth = clamp01((targetRange.low - minScale) / totalRange) * 100;
  let targetWidth    = clamp01((targetRange.high - targetRange.low) / totalRange) * 100;
  let improvementWidth = Math.max(0, 100 - excellentWidth - targetWidth);

  const sum = excellentWidth + targetWidth + improvementWidth;
  if (sum !== 100 && sum > 0) {
    excellentWidth = (excellentWidth / sum) * 100;
    targetWidth = (targetWidth / sum) * 100;
    improvementWidth = (improvementWidth / sum) * 100;
  }

  const sectionWidths = (direction === 'lower-is-better')
    ? { 'excellent': excellentWidth, 'target-range': targetWidth, 'needs-improvement': improvementWidth }
    : { 'needs-improvement': excellentWidth, 'target-range': targetWidth, 'excellent': improvementWidth };

  const scalePoints = [
    { value: minScale }, { value: targetRange.low },
    { value: targetRange.high }, { value: maxScale }
  ].map(p => ({
    label: valueFormatter(p.value),
    position: clamp01((p.value - minScale) / totalRange) * 100
  }));

  return { userValue, userPosition, userSection, sectionWidths, scalePoints, valueFormatter, direction, metricName };
}

function generateBar(data) {
  if (!data) return '';
  const scaleHtml = data.scalePoints.map(p => `<div class="scale-point" style="left: ${p.position}%;">${p.label}</div>`).join('');
  const sections = (data.direction === 'lower-is-better')
    ? ['excellent', 'target-range', 'needs-improvement']
    : ['needs-improvement', 'target-range', 'excellent'];

  return `
    <div class="bar-scale">${scaleHtml}</div>
    <div class="bar-track">
      ${sections.map(key => `<div class="zone-bg ${key}" style="flex-basis: ${(data.sectionWidths[key] || 0)}%"></div>`).join('')}
      <div class="user-marker ${data.userSection}" style="left: ${data.userPosition}%;">
        <div class="user-label">
          <span class="label-title">Your ${data.metricName}</span>
          ${data.valueFormatter(data.userValue)}
        </div>
      </div>
    </div>
    <div class="bar-labels">
      ${sections.map(key => {
        const labelText = key.replace(/-/g, ' ');
        return `<div class="label ${key} ${data.userSection === key ? 'active' : ''}" style="text-transform: capitalize; flex-basis: ${(data.sectionWidths[key] || 0)}%">${labelText}</div>`;
      }).join('')}
    </div>
  `;
}

/* ===========================
   PRIMARY STATUS / DIAGNOSIS / SCENARIO
   =========================== */
function renderPrimaryStatusBlock(d) {
  const container = document.getElementById("primaryStatusBlock");
  const ga = d.goal_analysis || {};
  const userGoal = d.input?.goal_cpl;

  container.className = "primary-status-block";
  if (!userGoal) {
    container.innerHTML = '<div class="context-text">Enter a goal CPL to see analysis.</div>';
    return;
  }

  const range = (ga.realistic_range && ga.realistic_range.low != null && ga.realistic_range.high != null)
    ? ga.realistic_range
    : ensureTargetRange({ value: userGoal, median: d?.benchmarks?.cpl?.median }, '$', 'CPL');

  const { low, high } = range;
  const recommended = ga.recommended_cpl ?? null;
  let verdictClass = 'on-target';
  if (ga.goal_scenario === 'goal_too_aggressive') verdictClass = 'outside-target';
  if (ga.goal_scenario === 'goal_in_range')      verdictClass = 'on-target';
  if (ga.goal_scenario === 'goal_conservative')  verdictClass = 'on-target';

  let deltaHtml = '';
  if (userGoal < low || userGoal > high) {
    const pct = Math.round(deltaPctFromRange(userGoal, range));
    const below = userGoal < low;
    const deltaClass = 'positive';
    const desc = below ? "below typical range" : "above typical range";
    deltaHtml = `<span class="delta-text ${deltaClass}">${pct}% ${desc} (${fmtMoney(low)}–${fmtMoney(high)})</span>`;
  } else {
    deltaHtml = `<span class="delta-text">within typical range (${fmtMoney(low)}–${fmtMoney(high)})</span>`;
  }

  const titleIcon = (verdictClass === 'outside-target') ? '⚠️' : '🎯';
  const titleText = 'Goal CPL check';
  const conservativeNote = (ga.goal_scenario === 'goal_conservative') ? `<span class="pill ok">Conservative goal</span>` : '';
  const recChip = recommended ? `<div class="recommendation-chip">Suggested target: ${fmtMoney(recommended)}</div>` : '';

  container.classList.add(verdictClass);
  container.innerHTML = `
    <div class="status-header ${verdictClass}">${titleIcon} ${titleText}</div>
    <div class="primary-value">Goal CPL: ${fmtMoney(userGoal)}</div>
    <div class="verdict-context">${deltaHtml} ${conservativeNote}</div>
    <div class="goal-comparison">
      <div class="goal-item">
        <div class="goal-k">Your goal</div>
        <div class="goal-v">${fmtMoney(userGoal)}</div>
      </div>
      <div class="goal-item">
        <div class="goal-k">Typical</div>
        <div class="goal-v">${fmtMoney(low)} – ${fmtMoney(high)}</div>
      </div>
      <div class="goal-item">
        <div class="goal-k">Recommended</div>
        <div class="goal-v">${recommended ? fmtMoney(recommended) : '—'}</div>
      </div>
    </div>
    ${recChip}
  `;
}

function statusFromBench(metric, metricName, unit, direction) {
  if (!metric || metric.value == null) return 'UNKNOWN';
  const tr = ensureTargetRange(metric, unit, metricName);
  const verdict = deriveVerdict(metric.value, tr, direction);
  if (verdict === 'on_target' || verdict === 'exceeds_target') return 'GOOD';
  const delta = deltaPctFromRange(metric.value, tr);
  return (delta <= 15) ? 'AVG' : 'WEAK';
}

function createDiagnosisCard(status, title, message, listItems = [], kicker = '', provisionalInfo = '') {
  const listHtml = listItems.length ? `<ul>${listItems.map(item => `<li>${item}</li>`).join('')}</ul>` : '';
  return `
    <div class="card diagnosis-card">
      ${provisionalInfo}
      <div class="status-header ${status}">${title}</div>
      <p>${message}</p>
      ${listHtml}
      ${kicker}
    </div>
  `;
}

function renderDiagnosis(d) {
  const el = document.getElementById("diagnosisCard");
  const { input, goal_analysis: ga, benchmarks: bm, targets, overall } = d;
  const status = overall?.goal_status;

  // --- Pre-checks ---
  if ((input.clicks || 0) === 0) {
    el.innerHTML = createDiagnosisCard('bad', '🛑 No Traffic',
      'There are no clicks recorded for this period. Expand coverage or bids before optimizing.',
      ['Broaden match types/targets; check ad eligibility and daily budgets.']);
    return;
  }

  const crSuspicious = (() => {
    const v = bm.cr?.value;
    if (v == null) return false;
    const tr = ensureTargetRange(bm.cr, '%', 'CR');
    return v > Math.max(tr.high * 1.8, 0.15);
  })();

  if ((input.leads === 0 && (input.clicks || 0) >= 100) || crSuspicious) {
    el.innerHTML = createDiagnosisCard('bad', '🚨 Verify Tracking',
      (input.leads === 0) ? 'Clicks but zero conversions. Likely tracking failure.' : 'Suspiciously high CR. You may be counting non-lead events.',
      ['Fire a test conversion; verify thank-you tag/pixel.', 'Confirm you're counting qualified leads only.']);
    return;
  }

  if (input.budget != null && input.budget < 500) {
    el.innerHTML = createDiagnosisCard('bad', '💰 Budget Too Low',
      'Under $500 isn't enough signal. Increase budget or <strong>Run Grader</strong> to find the right budget.');
    return;
  }

  let softBudgetBanner = (input.budget != null && input.budget < 1000)
    ? `<div class="pill ok" style="margin-bottom:8px;">Budget constrained ($${input.budget}) — insights are provisional</div>`
    : '';
  const provisional = ((input.leads || 0) < 15 || (input.clicks || 0) < 300);
  const provisionalTag = provisional ? '<div class="pill ok" style="margin-bottom:8px;">Provisional – low data volume</div>' : softBudgetBanner;

  const cplStatus = statusFromBench(bm.cpl, 'CPL', '$', 'lower-is-better');
  const cpcStatus = statusFromBench(bm.cpc, 'CPC', '$', 'lower-is-better');
  const crStatus  = statusFromBench(bm.cr,  'CR',  '%', 'higher-is-better');
  const goalScenario = ga.goal_scenario;
  const performanceIsGood = cplStatus !== 'WEAK' && cpcStatus !== 'WEAK' && crStatus !== 'WEAK';

  if (performanceIsGood) {
    if (status === 'achieved' || status === 'on_track') {
      el.innerHTML = createDiagnosisCard('good', '🚀 Ready to Scale',
        `CPL is meeting or beating the goal. ${goalScenario === 'goal_conservative' ? 'Your goal is conservative — you can scale with confidence.' : 'Performance is solid.'}`,
        ['Increase budget (start +30–50%).', 'Add adjacent ad groups / geos.'], '', provisionalTag);
    } else {
      const rec = ga.recommended_cpl;
      const rng = ga.realistic_range;
      el.innerHTML = createDiagnosisCard('ok', '🎯 Realign Goal Expectations',
        `Your performance is good, but your goal is outside the typical range. A more realistic target is <strong>${fmtMoney(rec)}</strong> (Typical: ${fmtMoney(rng?.low)}–${fmtMoney(rng?.high)}).`,
        ['<strong>Primary Action:</strong> Use this benchmark data to reset the CPL goal with your client.',
         '<strong>Next Step:</strong> Run a deeper ROAS calculation to find the true breakeven CPL.'], '', provisionalTag);
    }
    return;
  }

  if (status === 'behind') {
    const needCr = targets?.target_cr ?? ((bm.cpc?.value && input?.goal_cpl) ? (bm.cpc.value / input.goal_cpl) : null);
    const needCpc = targets?.target_cpc ?? ((input?.goal_cpl != null && bm.cr?.value != null) ? input.goal_cpl * bm.cr.value : null);
    const kicker = (goalScenario === 'goal_too_aggressive') ? `<div class="diag-kicker">*Note: Your CPL goal is also aggressive for this market. Re-evaluate it after improving performance.</div>` : '';

    const trCPC = ensureTargetRange(bm.cpc, '$', 'CPC');
    const trCR  = ensureTargetRange(bm.cr,  '%', 'CR');
    const cpcExtreme = bm.cpc?.value > trCPC.high * 1.5;
    const crDeep     = bm.cr?.value  < trCR.low  * 0.5;

    if (crStatus === 'WEAK' && cpcStatus !== 'WEAK') {
      el.innerHTML = createDiagnosisCard('bad', '🔧 Fix Conversion Rate',
        `Your CR of ${fmtPct1(bm.cr?.value)} is the main bottleneck. You need ~<strong>${needCr != null ? fmtPct1(Math.max(0, needCr)) : '—'}</strong> to hit your CPL goal.`,
        ['Audit page speed & mobile experience (aim &lt;2.5s LCP).', 'Ensure message match: ad → headline; CTA above fold.'], kicker, provisionalTag);
    } else if (cpcStatus === 'WEAK' && crStatus !== 'WEAK') {
      el.innerHTML = createDiagnosisCard('bad', '💰 Reduce Traffic Cost',
        `Your pages convert well, but the CPC of ${fmtMoney(bm.cpc?.value)} is too high. You need ~<strong>${needCpc != null ? fmtMoney(Math.max(0, needCpc)) : '—'}</strong> to hit your goal.`,
        ['Add negatives; cut waste by geo/device/daypart.', 'Tighten match types; refresh RSAs.'], kicker, provisionalTag);
    } else {
      if (cpcExtreme && !crDeep) {
        el.innerHTML = createDiagnosisCard('bad', '🔥 Start with CPC, then CR',
          'Both levers need work, but your CPC is an extreme outlier. Start by fixing traffic cost.',
          ['Aggressively prune queries and tighten targeting to control CPCs.', 'Once CPC is in a healthier range, shift focus to landing page optimization.'], kicker, provisionalTag);
      } else {
        el.innerHTML = createDiagnosisCard('bad', '🔥 Start with CR, then CPC',
          'Both levers need work. Start with CR for faster lift, then address CPC.',
          ['Apply LP quick wins; verify tracking quality.', 'Then prune queries / restructure bids.'], kicker, provisionalTag);
      }
    }
    return;
  }

  el.innerHTML = createDiagnosisCard('', '📊 Monitor Performance',
    'Your campaign is performing within average ranges. Continue to test incremental CR and CPC improvements to increase efficiency.', [], '', provisionalTag);
}

function renderScenarioBuilder(d) {
  const el = document.getElementById('scenarioBuilder');
  const bm = d.benchmarks || {};
  const goal = d.input?.goal_cpl ?? null;

  const hasCPC = bm.cpc && (bm.cpc.value != null || bm.cpc.median != null);
  const hasCR  = bm.cr  && (bm.cr.value  != null || bm.cr.median  != null);
  if (!hasCPC || !hasCR) { el.style.display = 'none'; return; }

  let state = {
    budget: d.input?.budget ?? 5000,
    cpc:    bm.cpc?.value ?? bm.cpc?.median ?? 3.00,
    cr:     bm.cr?.value  ?? bm.cr?.median  ?? 0.04
  };
  const current = { ...state };

  const p25cpc = bm.cpc?.p25, p75cpc = bm.cpc?.p75;
  const p25cr  = bm.cr?.p25,  p75cr  = bm.cr?.p75;

  const bounds = {
    budgetMin: 0,
    budgetMax: Math.max((d.input?.budget ?? 5000) * 2, 10000),
    cpcMin: Math.max(0.2, (p25cpc ?? state.cpc) * 0.5),
    cpcMax: (p75cpc ?? state.cpc) * 1.8 || 20,
    crMin: Math.max(0.001, (p25cr ?? state.cr) * 0.3),
    crMax: Math.min(1.0, (p75cr ?? state.cr) * 2.0)
  };
  if (bounds.cpcMin >= bounds.cpcMax) { bounds.cpcMin = Math.max(0.2, state.cpc * 0.5); bounds.cpcMax = state.cpc * 2.0; }
  if (bounds.crMin  >= bounds.crMax)  { bounds.crMin  = Math.max(0.001, state.cr  * 0.5); bounds.crMax  = Math.min(1.0, state.cr  * 2.0); }

  const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

  const compute = () => {
    const clicks = (state.budget > 0 && state.cpc > 0) ? state.budget / state.cpc : 0;
    const leads  = (state.cr > 0 && state.cpc > 0) ? state.budget * state.cr / state.cpc : 0;
    const cpl    = (state.cr > 0) ? state.cpc / state.cr : null;
    return { clicks, leads, cpl };
  };

  const updateState = (key, rawValue, isFromNumeric) => {
    let num = parseFloat(rawValue);
    if (!isFinite(num)) return;
    if (key === 'cr' && isFromNumeric) num /= 100; // numeric CR input is in %
    state[key] = clamp(num, bounds[`${key}Min`], bounds[`${key}Max`]);
    render();
  };

  el.oninput = (e) => {
    const keyMap = { sb_budget: 'budget', sb_cpc: 'cpc', sb_cr: 'cr' };
    if (keyMap[e.target.id]) updateState(keyMap[e.target.id], e.target.value, false);
  };
  el.onchange = (e) => {
    const keyMap = { sb_budget_n: 'budget', sb_cpc_n: 'cpc', sb_cr_n: 'cr' };
    if (keyMap[e.target.id]) updateState(keyMap[e.target.id], e.target.value, true);
  };
  el.onclick = (e) => {
    if (e.target.id === 'sb_reset') { state = { ...current }; render(); }
    if (goal && e.target.id === 'sb_snap_cr' && goal > 0 && state.cpc > 0) { state.cr = clamp(state.cpc / goal, bounds.crMin, bounds.crMax); render(); }
    if (goal && e.target.id === 'sb_snap_cpc' && goal > 0 && state.cr > 0) { state.cpc = clamp(goal * state.cr, bounds.cpcMin, bounds.cpcMax); render(); }
  };

  function render() {
    const { clicks, leads, cpl } = compute();
    const banners = [];
    if (clicks < 300 || leads < 15) banners.push(`<span class="pill ok" style="margin-bottom:8px;">Provisional – low volume</span>`);
    if (state.budget >= 500 && state.budget < 1000) banners.push(`<span class="pill ok" style="margin-bottom:8px;">Budget constrained (${fmtMoney(state.budget)})</span>`);

    // slider track progress
    const budgetProgress = (state.budget - bounds.budgetMin) / (bounds.budgetMax - bounds.budgetMin) * 100;
    const cpcProgress = (state.cpc - bounds.cpcMin) / (bounds.cpcMax - bounds.cpcMin) * 100;
    const crProgress = (state.cr - bounds.crMin) / (bounds.crMax - bounds.crMin) * 100;

    el.style.display = 'block';
    el.innerHTML = `
      <h3 style="margin:0 0 12px;">Scenario Builder</h3>
      ${banners.join(' ')}
      <div class="scenario-builder-input-grid" style="margin-top:8px;">
        <div>
          <div class="label-row">
            <label>Budget</label>
            <div class="numeric-input-wrapper symbol-before" data-symbol="$">
              <input type="number" id="sb_budget_n" class="numeric-input" value="${Math.round(state.budget)}" />
            </div>
          </div>
          <input type="range" id="sb_budget" min="${bounds.budgetMin}" max="${bounds.budgetMax}" step="50" value="${state.budget}" style="--progress: ${budgetProgress}%">
          <div class="slider-labels">
            <span>${fmtMoney(bounds.budgetMin)}</span>
            <span>${fmtMoney(bounds.budgetMax)}</span>
          </div>
        </div>
        <div>
          <div class="label-row">
            <label>CPC</label>
            <div class="numeric-input-wrapper symbol-before" data-symbol="$">
              <input type="number" id="sb_cpc_n" step="0.01" class="numeric-input" value="${state.cpc.toFixed(2)}" />
            </div>
          </div>
          <input type="range" id="sb_cpc" min="${bounds.cpcMin.toFixed(2)}" max="${bounds.cpcMax.toFixed(2)}" step="0.01" value="${state.cpc}" style="--progress: ${cpcProgress}%">
          <div class="slider-labels">
            <span>${fmtMoney(bounds.cpcMin)}</span>
            <span>${fmtMoney(bounds.cpcMax)}</span>
          </div>
        </div>
        <div>
          <div class="label-row">
            <label>CR</label>
            <div class="numeric-input-wrapper symbol-after" data-symbol="%">
              <input type="number" id="sb_cr_n" step="0.1" class="numeric-input" value="${(state.cr * 100).toFixed(1)}" />
            </div>
          </div>
          <input type="range" id="sb_cr" min="${bounds.crMin.toFixed(3)}" max="${bounds.crMax.toFixed(3)}" step="0.001" value="${state.cr}" style="--progress: ${crProgress}%">
          <div class="slider-labels">
            <span>${fmtPct1(bounds.crMin)}</span>
            <span>${fmtPct1(bounds.crMax)}</span>
          </div>
        </div>
      </div>
      <div class="row" style="margin-top:20px;">
        <div class="card">
          <div class="small">Leads</div>
          <div class="primary-value" style="margin-top:4px;">${Math.round(leads)}</div>
          <div class="small">Clicks: ${Math.round(clicks)}</div>
        </div>
        <div class="card">
          <div class="small">CPL ${goal && cpl != null && cpl > goal ? `(Goal: ${fmtMoney(goal)})` : ''}</div>
          <div class="primary-value" style="margin-top:4px; color:${(goal && cpl != null && cpl <= goal) ? 'var(--good)' : 'var(--fg)'}">${cpl != null ? fmtMoney(cpl) : '—'}</div>
          <div class="small">Formula: CPC ÷ CR</div>
        </div>
      </div>
      <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;">
        ${goal ? `<button id="sb_snap_cr" class="btn-small" type="button">Solve for Goal via CR</button>
                  <button id="sb_snap_cpc" class="btn-small" type="button">Solve for Goal via CPC</button>` : ''}
        <button id="sb_reset" class="btn-secondary" type="button">Reset to Current</button>
      </div>
    `;
  }
  render(); // initial
}

/* ===========================
   COPY SUMMARY
   =========================== */
function buildCopySummary(d) {
  const bm = d.benchmarks || {};
  const ga = d.goal_analysis || {};
  const rng = ga.realistic_range || {};
  const parts = [];

  parts.push(`Northlight Benchmark Summary`);
  parts.push(`Category: ${d.input?.category || '—'} / ${d.input?.subcategory || '—'}`);
  if (d.input?.goal_cpl != null) {
    parts.push(`Goal CPL: ${fmtMoney(d.input.goal_cpl)} ${rng.low!=null && rng.high!=null ? `(Typical: ${fmtMoney(rng.low)}–${fmtMoney(rng.high)})` : ''}`);
  }
  if (ga?.recommended_cpl != null) parts.push(`Recommended Target: ${fmtMoney(ga.recommended_cpl)}`);

  const linesFor = (name, obj, unit, dir) => {
    if (!obj) return;
    const tr = ensureTargetRange(obj, unit, name);
    const verdict = deriveVerdict(obj.value, tr, dir);
    const verdictText = verdict === 'exceeds_target' ? 'Excellent' :
                        verdict === 'on_target' ? 'On Target' :
                        verdict === 'outside_target' ? 'Outside Target' : '—';
    const val = unit === '$' ? fmtMoney(obj.value) : fmtPct1(obj.value);
    const low = unit === '$' ? fmtMoney(tr.low) : fmtPct1(tr.low);
    const high = unit === '$' ? fmtMoney(tr.high) : fmtPct1(tr.high);
    parts.push(`${name}: ${val} (Target Range: ${low}–${high}) – ${verdictText}`);
  };

  linesFor('CPL', bm.cpl, '$', 'lower-is-better');
  linesFor('CPC', bm.cpc, '$', 'lower-is-better');
  linesFor('CR',  bm.cr,  '%', 'higher-is-better');

  return parts.join('\n');
}

/* ===========================
   INPUT PERSISTENCE
   =========================== */
function loadInputsFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function saveInputsToStorage() {
  const payload = {
    category: document.getElementById("category").value || null,
    subcategory: document.getElementById("subcategory").value || null,
    goal_cpl: document.getElementById("goal_cpl").value ? parseFloat(document.getElementById("goal_cpl").value) : null,
    budget: document.getElementById("budget").value ? parseFloat(document.getElementById("budget").value) : null,
    clicks: document.getElementById("clicks").value ? parseFloat(document.getElementById("clicks").value) : null,
    leads: document.getElementById("leads").value ? parseFloat(document.getElementById("leads").value) : null
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}
function bindPersistence() {
  ['goal_cpl','budget','clicks','leads'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', saveInputsToStorage);
    el.addEventListener('change', saveInputsToStorage);
  });
  document.getElementById('category').addEventListener('change', () => { saveInputsToStorage(); });
  document.getElementById('subcategory').addEventListener('change', () => { saveInputsToStorage(); });
}

/* ===========================
   META FETCH + FORM BUILD
   =========================== */
async function fetchMeta() {
  const resultsSection = document.getElementById('results');
  const catSel = document.getElementById("category");
  const subSel = document.getElementById("subcategory");
  try {
    const res = await fetch(`${API}/benchmarks/meta`);
    if (!res.ok) throw new Error(`API request failed with status: ${res.status}`);
    const meta = await res.json();
    if (!Array.isArray(meta) || meta.length === 0) throw new Error("API returned empty or invalid category data.");

    const cats = [...new Set(meta.map(x => x.category))].sort();
    catSel.innerHTML = cats.map(c => `<option value="${c}">${c}</option>`).join("");

    function buildSubcats(cat, preferred) {
      const subs = meta.filter(x => x.category === cat).map(x => x.subcategory).sort();
      subSel.innerHTML = subs.map(s => `<option value="${s}">${s}</option>`).join("");
      if (preferred && subs.includes(preferred)) subSel.value = preferred;
    }

    // Rehydrate saved inputs now that we have options
    const saved = loadInputsFromStorage();
    if (saved?.category && cats.includes(saved.category)) {
      catSel.value = saved.category;
      buildSubcats(saved.category, saved.subcategory);
    } else {
      buildSubcats(catSel.value);
    }

    if (saved) {
      if (saved.goal_cpl != null) document.getElementById('goal_cpl').value = saved.goal_cpl;
      if (saved.budget != null)   document.getElementById('budget').value   = saved.budget;
      if (saved.clicks != null)   document.getElementById('clicks').value   = saved.clicks;
      if (saved.leads != null)    document.getElementById('leads').value    = saved.leads;
    }

    // Keep subcats in sync when category changes
    catSel.addEventListener("change", e => {
      buildSubcats(e.target.value);
      saveInputsToStorage();
    });

    // ARIA state
    resultsSection.setAttribute('aria-busy','false');
  } catch (err) {
    console.error("Could not load benchmark categories:", err);
    catSel.innerHTML = `<option disabled selected>Error: Data unavailable</option>`;
    subSel.innerHTML = `<option disabled selected>-</option>`;
    document.getElementById('results').setAttribute('aria-busy','false');
  }
}

/* ===========================
   EXECUTION
   =========================== */
let lastPayload = null;

async function runDiag() {
  const resultsEl = document.getElementById("results");
  resultsEl.style.display = "block";
  resultsEl.setAttribute('aria-busy','true');

  const payload = {
    category: document.getElementById("category").value,
    subcategory: document.getElementById("subcategory").value,
    budget: parseFloat(document.getElementById("budget").value) || null,
    clicks: parseFloat(document.getElementById("clicks").value) || null,
    leads: parseFloat(document.getElementById("leads").value) || null,
    goal_cpl: document.getElementById("goal_cpl").value ? parseFloat(document.getElementById("goal_cpl").value) : null
  };
  lastPayload = payload;
  saveInputsToStorage();

  try {
    const res = await fetch(`${API}/diagnose`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const t = await res.text();
      alert(`Error ${res.status}: ${t}`);
      resultsEl.setAttribute('aria-busy','false');
      return;
    }
    const data = await res.json();
    renderResults(data);

    // Enable copy + auto-scroll to results top for UX polish
    document.getElementById('copySummaryBtn').disabled = false;
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Also move keyboard focus for accessibility
    resultsEl.setAttribute('tabindex','-1');
    resultsEl.focus({ preventScroll: true });

  } catch (e) {
    console.error(e);
    alert("Network error running diagnostics.");
  } finally {
    resultsEl.setAttribute('aria-busy','false');
  }
}

function renderResults(d) {
  const bm = d.benchmarks || {};

  const metricsToRender = [
    { id: "cpl-verdict-container", data: bm.cpl, name: "CPL", unit: "$", dir: 'lower-is-better' },
    { id: "cpc-verdict-container", data: bm.cpc, name: "CPC", unit: "$", dir: 'lower-is-better' },
    { id: "cr-verdict-container",  data: bm.cr,  name: "CR",  unit: "%", dir: 'higher-is-better' }
  ];

  metricsToRender.forEach(metric => {
    renderVerdictBenchmark(
      document.getElementById(metric.id),
      metric.data, metric.name, metric.unit, metric.dir, { hidePeer: true }
    );
  });

  renderPrimaryStatusBlock(d);
  renderDiagnosis(d);
  renderScenarioBuilder(d);

  // Wire copy summary to current result
  const copyBtn = document.getElementById('copySummaryBtn');
  copyBtn.onclick = async () => {
    const text = buildCopySummary(d);
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = 'Copy Summary'; }, 1200);
    } catch {
      // Fallback if clipboard blocked
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = 'Copy Summary'; }, 1200);
    }
  };
}

/* ===========================
   BOOT
   =========================== */
document.getElementById("footerYear").textContent = new Date().getFullYear();
document.getElementById("runBtn").addEventListener("click", runDiag);
document.getElementById("resetBtn").addEventListener("click", () => {
  ['goal_cpl','budget','clicks','leads'].forEach(id => document.getElementById(id).value = '');
  saveInputsToStorage();
});
bindPersistence();
fetchMeta();