// frontend/book/script.js
import { renderChurnWaterfall } from './risk_waterfall.js';

// Store account data for waterfall rendering
let accountsData = [];

// Function to store accounts data when it's loaded
function storeAccountsData(accounts) {
  accountsData = accounts || [];
}

// Function to get account data by CID
function getAccountByCid(cid) {
  return accountsData.find(acc => acc.campaign_id === cid);
}

// Convert account data to waterfall config
function buildWaterfallFromAccount(account) {
  if (!account) return null;
  
  const driversData = account.risk_drivers_json;
  let drivers = null;
  
  // Parse drivers data (it can be object or string)
  if (typeof driversData === 'object' && driversData !== null) {
    drivers = driversData;
  } else if (typeof driversData === 'string') {
    try {
      drivers = JSON.parse(driversData);
    } catch (e) {
      console.warn('Failed to parse risk drivers JSON:', e);
    }
  }
  
  if (!drivers || !drivers.baseline) return null;
  
  // Extract headline total (the % shown in the header)
  const churnProb = account.churn_prob_90d;
  const totalPct = churnProb ? Math.round(churnProb * 100) : 0;
  
  const config = {
    total_pct: totalPct,
    baseline_pp: Math.max(0, Math.min(100, Math.round(drivers.baseline))),
    drivers: [],
    cap_to: 100,
    show_ranges: false
  };
  
  // Convert drivers to waterfall format with lift ratios
  if (Array.isArray(drivers.drivers)) {
    config.drivers = drivers.drivers.map(d => {
      const impact = Math.round(Number(d.impact || 0));
      const name = d.name || 'Risk Factor';
      const type =
        impact < 0 ? 'protective' :
        /CPL|Leads|deficit|Underpacing/i.test(name) ? 'controllable' :
        'structural';
      return {
        label: name,
        pp: impact,             // keep negatives
        type,
        why: getDriverExplanation(name),
        lift_x: estimateLiftRatio(name, impact)
      };
    });
  }
  
  return config;
}

function getDriverExplanation(driverName) {
  const explanations = {
    'High CPL (≥3× goal)': '3× goal historically elevates churn vs cohort.',
    'New Account (≤1m)': 'First 30 days show elevated hazard vs matured accounts.',
    'Single Product': 'Fewer anchors → higher volatility.',
    'Off-pacing': 'Under/over-spend drives instability and lead gaps.',
    'Below expected leads': 'Lead scarcity increases cancel probability.',
    'Zero Leads (30d)': 'Extended periods without leads indicate conversion issues.',
    'CPL above goal': 'Cost per lead exceeding target reduces campaign viability.',
    'Zero Leads (early)': 'Early zero-lead streak is an emerging conversion issue.',
    'Severe lead deficit (≤25% plan)': 'Very low leads relative to plan despite adequate spend.',
    'Lead deficit (≤50% plan)': 'Leads significantly behind plan.',
    'Underpacing': 'Low utilization creates delivery instability and fewer leads.',
    'Good volume / CPL (protective)': 'Healthy throughput or efficiency reduces churn likelihood.',
    'New & performing (protective)': 'Early-tenure but meeting plan; historically safer.'
  };
  return explanations[driverName] || 'Risk factor affecting churn probability.';
}

function estimateLiftRatio(driverName, impactPp) {
  // Rough estimates based on typical hazard ratios
  const baseLifts = {
    'High CPL (≥3× goal)': 3.2,
    'New Account (≤1m)': 4.1,
    'Single Product': 1.3,
    'Zero Leads (30d)': 3.2,
    'Underpacing': 1.15,
    'Lead deficit (≤50% plan)': 1.6,
    'Severe lead deficit (≤25% plan)': 2.8,
    'Good volume / CPL (protective)': 0.70,
    'New & performing (protective)': 0.75
  };
  
  const baseLift = baseLifts[driverName];
  if (baseLift && impactPp !== 0) {
    const scaleFactor = Math.min(2.0, Math.max(0.5, Math.abs(impactPp) / 20.0));
    return Math.round(baseLift * scaleFactor * 10) / 10;
  }
  return baseLift || null;
}

// Function to detect when cards are expanded and render waterfalls
function setupWaterfallObserver() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
        const target = mutation.target;
        if (target.classList.contains('card-expanded') && target.classList.contains('show')) {
          // Card was expanded, check for waterfall mount point
          const mount = target.querySelector('#churn-waterfall');
          if (mount && !mount.hasChildNodes()) {
            // Get the campaign ID from the card
            const card = target.closest('.priority-card');
            if (card) {
              const cidElement = card.querySelector('.detail-item strong');
              const cid = cidElement ? cidElement.textContent.trim() : null;
              if (cid && cid !== 'N/A') {
                renderWaterfallForCid(cid, mount);
              }
            }
          }
        }
      }
    });
  });

  // Start observing the cards container
  const cardsContainer = document.getElementById('cardsContainer');
  if (cardsContainer) {
    observer.observe(cardsContainer, {
      attributes: true,
      attributeFilter: ['class'],
      subtree: true
    });
  }
}

function renderWaterfallForCid(cid, mount) {
  const account = getAccountByCid(cid);
  if (!account) {
    console.warn('No account data found for CID:', cid);
    return;
  }
  
  const config = buildWaterfallFromAccount(account);
  if (config) {
    renderChurnWaterfall(mount, config);
  } else {
    // Show fallback message
    mount.innerHTML = '<div style="color:#64748b;font-size:12px;text-align:center;padding:20px;">Waterfall data not available for this account.</div>';
  }
}

// Listen for data updates from the main page
function setupDataListener() {
  window.addEventListener('accountsDataUpdated', (event) => {
    const { accounts } = event.detail;
    storeAccountsData(accounts);
    console.log('Waterfall module: Received accounts data update', accounts.length, 'accounts');
  });
}

document.addEventListener('DOMContentLoaded', () => {
  console.log('Waterfall module: Initializing');
  
  // Setup observer for waterfall rendering
  setupWaterfallObserver();
  
  // Setup data listener
  setupDataListener();
});