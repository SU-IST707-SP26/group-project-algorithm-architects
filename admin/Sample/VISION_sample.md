# VISION.md

## Current Vision

### Project: SafeWalk NYC - Block-Level Public Violence Prediction

**Primary Stakeholder:** NYC residents and visitors who need to make informed decisions about travel safety

**Secondary Stakeholders:** 
- NYC Police Department (resource allocation)
- Local businesses (understanding foot traffic patterns)
- City planning departments (infrastructure and lighting decisions)

**Problem Statement:**
New York City residents and visitors currently lack granular, predictive information about public safety risks when planning their routes through the city. While aggregate crime statistics exist at the precinct level, these are too coarse to help individuals make informed decisions about specific routes, especially during different times of day or weather conditions.

**Solution:**
We will build a machine learning model that predicts the likelihood of violent public crimes (assaults, robberies, car-jackings) at the block level in NYC, incorporating temporal patterns, weather conditions, and socioeconomic factors. The model will provide next-day predictions at 4-hour intervals, enabling users to plan safer routes.

---

## Version History

### Version 1.0 - Initial Vision (2025-09-15)

**Project: NYC Crime Analytics Platform**

**Primary Stakeholder:** NYC Police Department precincts

**Problem Statement:**
NYPD precincts need better predictive analytics to allocate resources efficiently across all crime types, from white collar fraud to domestic violence to street crimes.

**Solution:**
Build a comprehensive crime prediction system for all crime types at the block level to enable precise police resource allocation.

**Change Reason (2025-09-22):** After initial data exploration, we discovered:
1. The full crime taxonomy includes 60+ crime types with vastly different patterns and data quality
2. Many crimes (white collar, domestic violence) have poor location data, defaulting to precinct addresses
3. The scope was too broad for meaningful predictions in our timeframe
4. Pivoted to focus on violent public crimes only, with primary stakeholder shift from police to public safety users