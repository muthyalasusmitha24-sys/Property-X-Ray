"""
wellness_engine.py  --  Property X-Ray : Wellness Score engine
=================================================================
Pipeline (applied independently to each of the 9 wellness factors):
 
    Raw parameters -> Data preprocessing -> Normalisation -> Correlation matrix
    -> Factor analysis -> Factor loadings -> Entropy weighting
    -> Bootstrap validation -> Parameter weights -> Sub-factor scores
    -> Explained-variance weights -> Factor score
    -> Confidence interval
 
Then the 9 factor scores are combined into the overall Wellness Score
(equal / buyer-profile / data-driven weights) with a confidence interval.
"""
from __future__ import annotations
 
import warnings
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------------------------------
# 1. HIERARCHY  (taken from the Wellness Score diagram)
# --------------------------------------------------------------------------
HIERARCHY: dict[str, dict] = {
    "EMO": {"name": "Emotional Wellness", "subs": {
        "Stress Reduction Environment": ["Green views", "Noise levels", "Natural light"],
        "Aesthetic Pleasure": ["Visual appeal", "Architecture quality", "Interior design"],
        "Sense of Safety & Security": ["Security features", "Safe surroundings", "Gated / access control"]}},
    "PHY": {"name": "Physical Wellness", "subs": {
        "Physical Activity Support": ["Proximity to parks", "Sports facilities", "Gyms / fitness centers"],
        "Walkability": ["Walk score", "Sidewalk quality", "Pedestrian friendliness"],
        "Safety Features": ["Lighting quality", "Traffic safety", "Crossings & signs"],
        "Environmental Quality": ["Air quality index", "Water quality", "Clean surroundings"],
        "Accessibility": ["Universal design", "Ramps & lifts", "Senior/disabled access"],
        "Healthcare Access": ["Distance to hospitals", "Clinics availability", "Emergency access"],
        "Transportation": ["Public transport access", "Frequency & reliability", "Connectivity"],
        "Ergonomics": ["Space planning", "Ventilation", "Comfort design"]}},
    "SOC": {"name": "Social Wellness", "subs": {
        "Community Engagement": ["Social events", "Community organizations", "Volunteer opportunities"],
        "Neighbourhood Safety": ["Crime rate", "Perception of safety", "Emergency services access"],
        "Social Connectivity": ["Public spaces", "Transportation links", "Digital connectivity"],
        "Inclusivity & Diversity": ["Demographic diversity", "Cultural activities"],
        "Community Support Networks": ["Support groups", "Local resources", "Mutual help"],
        "Work-Life Balance": ["Proximity to workplaces", "Leisure facilities"]}},
    "OCC": {"name": "Occupational Wellness", "subs": {
        "Work-Life Balance": ["Flexible workspace", "Work hours balance", "Personal time"],
        "Commute Efficiency": ["Commute time", "Traffic conditions", "Transport options"],
        "Coworking Availability": ["Coworking spaces", "Shared offices", "Business centers"],
        "Employment Density": ["Job opportunities", "Employment rate", "Business ecosystem"],
        "Job Opportunities": ["Career growth", "Industry presence", "Skill demand"]}},
    "ENV": {"name": "Environmental Wellness", "subs": {
        "Air Quality": ["AQI value", "Pollutant levels", "Ventilation quality"],
        "Water Quality": ["Drinking water quality", "Water source", "Contamination levels"],
        "Waste Management": ["Waste segregation", "Collection systems", "Recycling facilities"],
        "Noise Pollution": ["Noise levels (dB)", "Traffic noise", "Industrial noise"],
        "Green Cover": ["% Green cover", "Trees / vegetation", "Open spaces"],
        "Energy Efficiency": ["Energy star rating", "Renewable energy", "Efficiency systems"]}},
    "INT": {"name": "Intellectual Wellness", "subs": {
        "Educational Resources": ["Schools proximity", "Colleges proximity", "Quality of education"],
        "Library Access": ["Public libraries", "Digital resources", "Reading spaces"],
        "Skill Development": ["Training centers", "Workshops", "Online learning access"],
        "Cultural & Creative Engagement": ["Cultural events", "Museums / galleries", "Creative spaces"],
        "Innovation & Learning Ecosystem": ["Hackathons / forums", "Community learning", "Knowledge networks"]}},
    "SPI": {"name": "Spiritual Wellness", "subs": {
        "Places of Worship": ["Temples / Churches", "Mosques / Gurudwaras", "Distance & access"],
        "Meditation Spaces": ["Quiet zones", "Meditation centers", "Yoga facilities"],
        "Tranquility & Peace": ["Calm environment", "Low noise", "Stress-free zones"],
        "Cultural Heritage": ["Heritage sites", "Historical places", "Traditions & art"],
        "Mind-Body Balance": ["Wellness centers", "Yoga / fitness", "Holistic activities"]}},
    "FIN": {"name": "Financial Wellness", "subs": {
        "Financial Performance Metrics": ["Cap Rate", "Rent to Cost Ratio", "ROI"],
        "Investment Viability": ["NPV", "IRR", "Break-even"],
        "Cost & Affordability": ["Property price", "Maintenance cost", "Utility cost"],
        "Cash Flow Stability": ["Rental income", "Occupancy rate", "Debt coverage"],
        "Market & Growth Potential": ["Market trend", "Appreciation", "Future demand"]}},
    "STA": {"name": "Statutory Wellness", "subs": {
        "Legal Compliance": ["RERA registration", "Building approvals", "Land use compliance"],
        "Title & Ownership Documents": ["Title deed", "Encumbrance certificate", "Sale agreement"],
        "Building & Safety Certifications": ["Occupancy certificate", "Fire safety NOC", "Structural safety"],
        "Tax & Financial Documents": ["Property tax receipt", "Utility bills", "Financial statements"],
        "Government Verifications": ["Aadhaar / PAN", "KYC documents", "Other clearances"]}},
}
 
# Parameters where a HIGHER raw value is WORSE (direction = "lower").
# Everything else is "higher is better".  Editable in the app.
LOWER_IS_BETTER = {
    "EMO": {"Noise levels"},
    "PHY": {"Air quality index", "Distance to hospitals"},
    "SOC": {"Crime rate"},
    "OCC": {"Commute time", "Traffic conditions"},
