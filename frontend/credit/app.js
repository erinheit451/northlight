const $ = (s, r=document)=>r.querySelector(s);
const $$ = (s, r=document)=>Array.from(r.querySelectorAll(s));

$("#calc").addEventListener("click", async () => {
  const products = $$(".prod").filter(x=>x.checked).map(x=>x.value);
  const payload = {
    category: $("#category").value,
    subcategory: $("#subcategory").value || null,
    geo: { city: $("#city").value||null, state: $("#state").value||null, zip: $("#zip").value||null },
    budget: Number($("#budget").value||0),
    products,
    contract_type: $("#contract").value,
    io_length_cycles: Number($("#io_len").value||0) || null,
    goal: { type: $("#goal_type").value, value: Number($("#goal_value").value||0) },
    landing_page_url: $("#lp_url").value || null
  };

  // Add economics if provided
  const txValue = Number($("#tx_value").value||0);
  const closeRate = Number($("#close_rate").value||0);
  if (txValue > 0 && closeRate > 0) {
    payload.economics = {
      transaction_value: txValue,
      close_rate_pct: closeRate
    };
  }

  const r = await fetch("/api/score/campaign", {
    method: "POST",
    headers: {"content-type":"application/json"},
    body: JSON.stringify(payload)
  });
  if (!r.ok) {
    $("#results").innerHTML = `<div class="err">Error ${r.status}</div>`;
    return;
  }
  const data = await r.json();
  $("#results").innerHTML = renderResults(data);
});

function renderReportSections(list){
  if (!list || !list.length) return "";
  const html = list.map(r => `
    <div class="section">
      <h4>${r.factor}: ${r.verdict} <span>${r.points>0?'+':''}${r.points}</span></h4>
      <p>${escapeHTML(r.explanation)}</p>
    </div>`).join("");
  return `<div class="card"><h3>Credit Report</h3>${html}</div>`;
}

function renderResults(d){
  const top = d.top_contributors.map(c=>`<li>${c.name}: <b>${c.points > 0 ? '+' : ''}${c.points}</b></li>`).join("");
  const fixes = d.fix_list.map(f=>`<li>${f.action} <span>+${f.points_gain}</span></li>`).join("");
  
  const econHtml = d.economics && d.economics.rev_per_lead > 0 ? `
    <div class="card">
      <h3>Economics</h3>
      <p>Revenue per lead: $${d.economics.rev_per_lead}</p>
      <p>Break-even CPL: $${d.economics.breakeven_cpl}</p>
      <p>ROAS at goal: ${d.economics.roas_at_goal}×</p>
    </div>` : '';

  return `
    <div class="card">
      <h2>Score: ${d.score} <span class="grade ${d.grade}">${d.grade}</span></h2>
      <div class="meta">Confidence: ${d.confidence}</div>
      <div class="meta">Suggested Goal: ${d.suggested_goal ? '$'+Math.round(d.suggested_goal) : '—'}</div>
    </div>

    ${renderReportSections(d.report_sections)}

    <div class="card"><h3>Why</h3><ul>${top||'<li>—</li>'}</ul></div>

    ${econHtml}

    <div class="card"><h3>Launch Terms</h3><pre>${escapeHTML(JSON.stringify(d.terms, null, 2))}</pre></div>

    <div class="card"><h3>Fix-List</h3><ul>${fixes||'<li>—</li>'}</ul></div>
  `;
}
function escapeHTML(s){return s.replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}