# Partners Pipeline Redesign Plan

## Current Architecture Analysis

### Data Sources
1. **Breakout File (Master Roster)**: 2,067 campaigns - COMPLETE UNIVERSE
   - All products: Search, Display, Chat, SEO, XMO, etc.
   - All partners and advertisers
   - Complete campaign roster across product stack

2. **Health File (Performance Data)**: 800 campaigns - TUNABLE MEDIA SUBSET  
   - Only campaigns that can be performance-tuned
   - Contains budget data ($1.48MM total)
   - Primarily Search/SEM focused

### Current Problem
- Partners.html only shows 449 campaigns (intersection of both files)
- Missing 1,618 campaigns from complete product portfolio
- Can't see full product mix per advertiser/partner

## Solution Plan

### Phase 1: Complete Campaign Visibility ✅ (PRIORITY)
**Goal**: Show ALL 2,067 campaigns in partners.html for complete product mix visibility

**Changes Needed**:

1. **Primary Data Source**: Use Breakout file as the foundation
   - Ensures all campaigns appear regardless of performance data availability
   - Complete product type visibility (Search, Display, Chat, SEO, etc.)

2. **Budget Enrichment**: LEFT JOIN with Health file for budget data
   - Campaigns with performance data get actual budgets
   - Campaigns without performance data get budget = $0 or estimated value
   - Preserve complete campaign roster

3. **Aggregation Logic**: Group by Partner → Advertiser → Product Mix
   - Count campaigns across all product types
   - Sum available budgets (where data exists)
   - Show complete advertiser portfolio per partner

### Implementation Changes Required

#### File: `backend/book/partners_data.py`

**Current Logic** (BROKEN):
```python
# Currently filters to only matching campaigns
df = merge(breakout, health, on='campaign_id', how='left')
# Result: Only 449 campaigns with budget data
```

**New Logic** (COMPLETE VISIBILITY):
```python
# Step 1: Start with complete breakout roster (2,067 campaigns)
complete_roster = load_breakout_data()

# Step 2: Enrich with budget data where available  
health_budgets = load_health_data()[['campaign_id', 'campaign_budget', 'am', 'utilization']]
enriched = pd.merge(complete_roster, health_budgets, on='campaign_id', how='left')

# Step 3: Fill missing budgets with 0 (or estimation logic)
enriched['campaign_budget'] = enriched['campaign_budget'].fillna(0)

# Result: ALL 2,067 campaigns visible, with budget data where available
```

#### File: `backend/book/partners_data.py` - Aggregation Functions

**Update `aggregate_partner_metrics()`**:
```python
def aggregate_partner_metrics(df):
    # NOW INCLUDES ALL CAMPAIGNS, NOT JUST THOSE WITH BUDGET DATA
    for partner in df['partner_name'].unique():
        partner_campaigns = df[df['partner_name'] == partner]
        
        # Total budget (sum of available budget data)
        total_budget = partner_campaigns['campaign_budget'].sum()
        
        # Campaign counts by product type
        product_counts = partner_campaigns['product_type'].value_counts()
        
        # Advertiser analysis across complete portfolio
        advertiser_analysis = partner_campaigns.groupby('maid').agg({
            'product_type': 'nunique',  # Products per advertiser
            'campaign_budget': 'sum'    # Budget per advertiser
        })
```

### Expected Outcomes

#### Before Fix:
- **449 campaigns** visible in partners.html
- **Limited product mix** visibility  
- **$768K budget** captured
- **Missing majority** of advertiser portfolios

#### After Fix:
- **2,067 campaigns** visible in partners.html ✅
- **Complete product mix** across Search, Display, Chat, SEO, etc. ✅
- **All advertiser portfolios** visible per partner ✅
- **Budget data where available** + $0 for others ✅

### Phase 2: Budget Enhancement (FUTURE)
**Goal**: Better budget visibility and estimation

**Potential Enhancements**:
1. **Historical Budget Estimation**: Use past data to estimate budgets for campaigns without current data
2. **Product-Type Defaults**: Apply average budgets by product type
3. **Advertiser-Level Distribution**: Distribute known advertiser budgets across their campaigns

## Implementation Priority

### HIGH PRIORITY (Implement Now):
1. ✅ **Complete Campaign Visibility** - Show all 2,067 campaigns
2. ✅ **Product Mix Visibility** - All product types per advertiser
3. ✅ **Partner Portfolio View** - Complete advertiser roster per partner

### MEDIUM PRIORITY (Future Enhancement):
1. **Budget Estimation** - Better budget values for non-tunable campaigns  
2. **Historical Trends** - Budget progression over time
3. **Performance Context** - Which campaigns are tunable vs. brand/awareness

## Success Metrics

**Data Completeness**:
- ✅ All 2,067 campaigns visible
- ✅ All 10+ product types represented  
- ✅ All 588 advertisers visible across partners

**Partner Visibility**:
- ✅ Complete advertiser portfolios per partner
- ✅ Accurate product distribution counts
- ✅ True monthly budget totals (available data + gaps identified)

**User Experience**:
- ✅ Partners.html shows comprehensive view
- ✅ Cross-sell opportunities based on complete portfolio
- ✅ No missing campaigns or advertiser relationships

## Next Steps

1. **Implement complete campaign visibility** (modify partners_data.py)
2. **Test with all 2,067 campaigns** showing in partners.html
3. **Verify product mix accuracy** across all partners
4. **Validate advertiser portfolio completeness**
5. **Consider budget estimation** for Phase 2