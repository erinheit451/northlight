document.addEventListener('DOMContentLoaded', () => {
    console.log('Growth Opportunities app starting...');
    
    // DOM Elements
    const loadingEl = document.getElementById('loadingState');
    const errorEl = document.getElementById('errorState');
    const overviewPage = document.getElementById('overviewPage');
    const partnerDetailPage = document.getElementById('partnerDetailPage');
    const partnerGrid = document.getElementById('partnerGrid');
    const advertiserTbody = document.getElementById('advertiserTbody');
    const backButton = document.getElementById('backButton');
    const breadcrumbHome = document.getElementById('breadcrumbHome');
    const partnerTitle = document.getElementById('partnerTitle');
    const partnerBreadcrumb = document.getElementById('partnerBreadcrumb');

    let allData = [];
    let filteredData = []; // This will hold the data currently being viewed

    // Helper Functions
    const formatMoney = (amount) => {
        if (!amount) return '$0';
        return '$' + amount.toLocaleString(undefined, {maximumFractionDigits: 0});
    };

    const formatPercent = (decimal) => {
        if (decimal === null || decimal === undefined) return '0%';
        return Math.round(decimal * 100) + '%';
    };

    const normalizeProductName = (campaign) => {
        const financeProduct = (campaign.finance_product || '').toLowerCase();
        const offerName = (campaign.offer_name || '').toLowerCase();
        if (offerName.includes('youtube')) return 'YouTube';
        if (offerName.includes('dash')) return 'Dash';
        if (financeProduct === 'search') return 'Search';
        if (financeProduct.includes('display')) return 'Display';
        if (financeProduct === 'display social' || offerName.includes('facebook') || offerName.includes('snapchat') || offerName.includes('linkedin')) return 'Social';
        if (financeProduct === 'xmo') return 'XMO';
        if (financeProduct === 'lead conversion' || financeProduct.includes('reachedge') || financeProduct === 'totaltrack') return 'Reporting';
        if (financeProduct === 'seo') return 'SEO';
        if (financeProduct === 'website' || financeProduct === 'reachedge + reachsite') return 'Website';
        if (financeProduct === 'chat') return 'Chat';
        if (financeProduct === 'listings') return 'Listings';
        if (financeProduct === 'lsa') return 'LSA';
        if (financeProduct.includes('email') || financeProduct.includes('tem')) return 'Email';
        if (financeProduct.includes('promotion')) return 'Promotions';
        return financeProduct.charAt(0).toUpperCase() + financeProduct.slice(1);
    };

    // Page Visibility Control
    const showPage = (page) => {
        loadingEl.classList.add('hidden');
        errorEl.classList.add('hidden');
        overviewPage.classList.add('hidden');
        partnerDetailPage.classList.add('hidden');
        if (page === 'overview') overviewPage.classList.remove('hidden');
        else if (page === 'detail') partnerDetailPage.classList.remove('hidden');
        else if (page === 'error') errorEl.classList.remove('hidden');
        else loadingEl.classList.remove('hidden');
    };

    // Calculate opportunity score for partner sorting
    const calculateOpportunityScore = (partner) => {
        const greenSinglesScore = partner.green_single_targets / Math.max(partner.total_advertisers, 1);
        const spendScore = Math.min(partner.total_monthly_spend / 50000, 1);
        const singleProductScore = partner.single_product_advertisers / Math.max(partner.total_advertisers, 1);
        return (greenSinglesScore * 0.5 + spendScore * 0.3 + singleProductScore * 0.2);
    };

    const getOpportunityLevel = (score) => {
        if (score > 0.6) return 'high';
        if (score > 0.3) return 'medium';
        return 'low';
    };

    // Fetch Data Function
    const fetchPartnerData = async () => {
        console.log('Fetching partner data from /api/sales/partner-dashboard...');
        const response = await fetch('/api/sales/partner-dashboard');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        console.log('Received data for', data.length, 'partners');
        return data;
    };

    // Calculate summary data from a given list of partners
    const calculateSummaryData = (partners) => {
        const totalAdvertisers = partners.reduce((sum, p) => sum + (p.total_advertisers || 0), 0);
        const singleProductAdvertisers = partners.reduce((sum, p) => sum + (p.single_product_advertisers || 0), 0);
        const multiProductAdvertisers = totalAdvertisers - singleProductAdvertisers;
        const saturationRate = totalAdvertisers > 0 ? multiProductAdvertisers / totalAdvertisers : 0;
        let totalCampaigns = 0, greenCampaigns = 0, totalProductInstances = 0;
        const productCounts = {};
        partners.forEach(partner => {
            if (partner.advertisers && Array.isArray(partner.advertisers)) {
                partner.advertisers.forEach(advertiser => {
                    totalCampaigns += advertiser.total_campaigns || 0;
                    greenCampaigns += advertiser.green_campaigns || 0;
                    if (advertiser.campaigns && Array.isArray(advertiser.campaigns)) {
                        advertiser.campaigns.forEach(campaign => {
                            if (campaign.finance_product) {
                                const cleanedName = normalizeProductName(campaign);
                                productCounts[cleanedName] = (productCounts[cleanedName] || 0) + 1;
                                totalProductInstances++;
                            }
                        });
                    }
                });
            }
        });
        const productComposition = Object.entries(productCounts)
            .map(([name, count]) => ({ name, count, percentage: totalProductInstances > 0 ? count / totalProductInstances : 0 }))
            .sort((a, b) => b.count - a.count);
        return {
            active_partners: partners.length,
            total_advertisers: totalAdvertisers,
            total_campaigns: totalCampaigns,
            green_band_campaigns: greenCampaigns,
            saturation_rate: saturationRate,
            product_composition: productComposition
        };
    };

    // NEW: Central function to refresh the entire dashboard view
    const updateDashboardView = (partnersToDisplay) => {
        const summaryData = calculateSummaryData(partnersToDisplay);
        renderSummaryCards(summaryData);
        renderPartnerCards(partnersToDisplay);
    };

    // Render Functions
    const renderSummaryCards = (summaryData) => {
        document.getElementById('activePartners').textContent = summaryData.active_partners || '0';
        document.getElementById('totalAdvertisers').textContent = summaryData.total_advertisers || '0';
        document.getElementById('totalCampaigns').textContent = summaryData.total_campaigns || '0';
        document.getElementById('greenCampaigns').textContent = summaryData.green_band_campaigns || '0';
        document.getElementById('saturationRate').textContent = formatPercent(summaryData.saturation_rate);
        renderProductBreakdown(summaryData.product_composition);
    };

    const renderProductBreakdown = (productComposition) => {
        const container = document.getElementById('productBreakdown');
        if (!container) return;
        if (!productComposition || productComposition.length === 0) {
            container.innerHTML = '<p class="small">No product data available</p>';
            return;
        }
        container.innerHTML = productComposition.map(product => `
            <div class="product-row">
                <div class="product-name">${product.name}</div>
                <div class="product-bar-container"><div class="product-bar" style="width: ${Math.max(product.percentage * 100, 2)}%"></div></div>
                <div class="product-value">${product.count} (${formatPercent(product.percentage)})</div>
            </div>
        `).join('');
    };

    const renderPartnerCards = (partnersToDisplay) => {
        console.log('Rendering', partnersToDisplay.length, 'partner cards...');
        if (!partnersToDisplay || partnersToDisplay.length === 0) {
            partnerGrid.innerHTML = '<p class="centered-message">No partners found for the selected GM.</p>';
            return;
        }
        partnerGrid.innerHTML = '';
        partnersToDisplay.forEach(partner => {
            partner.opportunityScore = calculateOpportunityScore(partner);
            partner.opportunityLevel = getOpportunityLevel(partner.opportunityScore);
        });
        partnersToDisplay.sort((a, b) => b.opportunityScore - a.opportunityScore);
        partnersToDisplay.forEach(partner => {
            const card = document.createElement('div');
            card.className = `partner-card ${partner.opportunityLevel}-opportunity`;
            const saturationRate = partner.single_product_advertisers / Math.max(partner.total_advertisers, 1);
            card.innerHTML = `
                <div class="partner-header">
                    <div class="partner-name">${partner.partner_name || 'Unknown Partner'}</div>
                    <div class="opportunity-badge ${partner.opportunityLevel}">${partner.opportunityLevel.toUpperCase()}</div>
                </div>
                <div class="partner-stats">
                    <div class="stat-item"><span class="stat-number">${partner.total_advertisers || 0}</span><span class="stat-label">Advertisers</span></div>
                    <div class="stat-item"><span class="stat-number" style="color: var(--good);">${partner.green_single_targets || 0}</span><span class="stat-label">Green Singles</span></div>
                </div>
                <div class="partner-insights">
                    <div class="insight-row"><span class="insight-label">Single-Product:</span><span class="insight-value">${partner.single_product_advertisers || 0} (${formatPercent(saturationRate)})</span></div>
                    <div class="insight-row"><span class="insight-label">Total Monthly Spend:</span><span class="insight-value">${formatMoney(partner.total_monthly_spend || 0)}</span></div>
                    <div class="insight-row"><span class="insight-label">Red Backlog:</span><span class="insight-value">${partner.red_backlog || 0} campaigns</span></div>
                </div>
            `;
            card.addEventListener('click', () => showPartnerDetail(partner.partner_id));
            partnerGrid.appendChild(card);
        });
    };

    const renderAdvertiserTable = (partner) => {
        document.getElementById('partnerTotalAdvertisers').textContent = partner.total_advertisers || 0;
        document.getElementById('partnerSingleProduct').textContent = partner.single_product_advertisers || 0;
        document.getElementById('partnerGreenSingles').textContent = partner.green_single_targets || 0;
        document.getElementById('partnerTotalSpend').textContent = formatMoney(partner.total_monthly_spend || 0);
        advertiserTbody.innerHTML = '';
        if (!partner.advertisers || partner.advertisers.length === 0) {
            advertiserTbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No advertiser data available</td></tr>';
            return;
        }
        partner.advertisers.forEach(adv => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><div><div style="font-weight: 600;">${adv.advertiser_name || 'Unknown Advertiser'}</div><div class="small">${adv.sub_category || 'N/A'}</div></div></td>
                <td><div class="product-pills">${adv.active_products && adv.active_products.length > 0 ? adv.active_products.map(p => `<span class="product-pill ${p.toLowerCase().replace(/[^a-z]/g, '')}">${p}</span>`).join('') : '<span class="small">No products</span>'}</div></td>
                <td>${formatMoney(adv.total_spend || 0)}</td>
                <td><div class="play-container">${adv.recommended_plays && adv.recommended_plays.length > 0 ? adv.recommended_plays.map(play => `<div class="play"><div class="play-name">${play.pitch?.headline || 'Expand Product Mix'}</div></div>`).join('') : '<span class="small">Analyze opportunities</span>'}</div></td>
                <td><div class="action-buttons"><button class="action-btn">View Details</button><button class="action-btn secondary">Notes</button></div></td>
            `;
            advertiserTbody.appendChild(row);
        });
    };

    // Event Handlers
    const showPartnerDetail = (partnerId) => {
        const partner = allData.find(p => p.partner_id === partnerId);
        if (partner) {
            partnerTitle.textContent = partner.partner_name || 'Unknown Partner';
            partnerBreadcrumb.textContent = partner.partner_name || 'Unknown Partner';
            renderAdvertiserTable(partner);
            showPage('detail');
        }
    };

    // Main initialization
    const initializeApp = async () => {
        try {
            console.log('Starting app initialization...');
            allData = await fetchPartnerData();

            allData.forEach(partner => {
                if (partner.advertisers && partner.advertisers[0] && partner.advertisers[0].campaigns && partner.advertisers[0].campaigns[0]) {
                    partner.gm_name = partner.advertisers[0].campaigns[0].area;
                } else {
                    partner.gm_name = 'Unknown';
                }
            });

            filteredData = [...allData];

            const gmSelector = document.getElementById('gmSelector');
            const gms = [...new Set(allData.map(p => p.gm_name).filter(Boolean))];
            gms.sort().forEach(gm => {
                if (gm !== 'Unknown') {
                    const option = document.createElement('option');
                    option.value = gm;
                    option.textContent = gm;
                    gmSelector.appendChild(option);
                }
            });

            gmSelector.addEventListener('change', () => {
                const selectedGm = gmSelector.value;
                filteredData = (selectedGm === 'all') 
                    ? [...allData] 
                    : allData.filter(p => p.gm_name === selectedGm);
                
                updateDashboardView(filteredData);
            });
            
            updateDashboardView(filteredData);
            showPage('overview');
            console.log('App initialized successfully!');
            
        } catch (error) {
            console.error("Failed to initialize app:", error);
            showPage('error');
        }
    };

    if (backButton) backButton.addEventListener('click', () => showPage('overview'));
    if (breadcrumbHome) breadcrumbHome.addEventListener('click', (e) => { e.preventDefault(); showPage('overview'); });

    initializeApp();
});