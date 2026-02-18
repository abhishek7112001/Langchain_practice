import pandas as pd

import numpy as np

from typing import Dict, List, Tuple

import re

import json
 
 
# Sample dataframes (replace with your actual data loading)

notifications_df = research_df

# Stock metadata mapping (Category, Sector)

def load_symbol_metadata(path):

    with open(path, "r") as f:

        metadata = json.load(f)
 
    cap_map = {k: v["Category"] for k, v in metadata.items()}

    sector_map = {k: v["Sector"] for k, v in metadata.items()}
 
    return cap_map, sector_map
 
 
stock_marketcap, stock_sector = load_symbol_metadata('/home/ec2-user/SageMaker/pipeline/symbol_metadata_updated.json')
 
# **COMPLETE COHORT USAGE** - Extract ALL cohort components

def parse_cohort_full(cohort_str: str) -> Dict[str, str]:

    """Parse FULL cohort: Category|Cluster|Sector"""

    parts = cohort_str.split('|')

    return {

        'cust_category': parts[0],

        'cust_cluster': parts[1], 

        'cust_sector': parts[2]

    }
 
# Create stock metadata with defaults for notifications

def create_enriched_metadata(marketcap_dict: Dict, sector_dict: Dict, notifications: pd.DataFrame) -> pd.DataFrame:

    """Enrich notifications with marketcap/sector + defaults"""

    all_symbols = set(notifications['symbol'].tolist()) | set(marketcap_dict.keys()) | set(sector_dict.keys())

    metadata = []

    for symbol in all_symbols:

        row = {'symbol': symbol}

        row['marketcap'] = marketcap_dict.get(symbol, 'Unknown')  # Default

        row['sector'] = sector_dict.get(symbol, 'Unknown')  # Default for banks

        metadata.append(row)

    return pd.DataFrame(metadata)
 
stock_metadata_df = create_enriched_metadata(stock_marketcap, stock_sector, notifications_df)
 
# **FULL COHORT-BASED MATCHING** [web:51]

def match_using_full_cohort(notif_row: pd.Series, cohort_row: pd.Series) -> Dict[str, float]:

    """Match using COMPLETE cohort: Category + Cluster + Sector"""

    notif_cat = notif_row['marketcap']

    notif_sector = notif_row['sector']

    cohort_parts = parse_cohort_full(cohort_row['cohort'])

    scores = {

        'category_match': 0.0,

        'cluster_match': 0.0, 

        'sector_match': 0.0,

        'total_cohort_score': 0.0

    }

    # 1. CATEGORY MATCH (40% weight) - Exact match with cohort category

    if notif_cat == cohort_parts['cust_category']:

        scores['category_match'] = 4.0

    # Penalty for mismatch

    elif notif_cat != cohort_parts['cust_category']:

        scores['category_match'] = -2.0

    # REAL CLUSTER QUALITY MATCH (scaled 0–4)

    real_cluster_quality = cohort_row.get('cluster_quality', 0.5)

    scores['cluster_match'] = real_cluster_quality * 4.0

    # 3. SECTOR MATCH (40% weight) - Exact or fuzzy match

    if (notif_sector.lower() in cohort_parts['cust_sector'].lower() or 

        cohort_parts['cust_sector'].lower() in notif_sector.lower()):

        scores['sector_match'] = 4.0

    # elif any(word in notif_sector.lower() for word in cohort_parts['cust_sector'].lower().split()):

    #     scores['sector_match'] = 2.0

    # **TOTAL COHORT SCORE** = Weighted sum of all cohort components

    # scores['total_cohort_score'] = (

    #     scores['category_match'] * 0.4 +

    #     scores['cluster_match'] * 0.2 + 

    #     scores['sector_match'] * 0.4

    # )

    # TOTAL COHORT SCORE (rebalanced)

    if scores['category_match'] > 0 or scores['sector_match'] > 0:

        scores['total_cohort_score'] = (

            scores['category_match'] * 0.3 +

            scores['cluster_match']  * 0.4 +

            scores['sector_match']   * 0.3

        )

    else:

        scores['total_cohort_score'] = 0.0

    return scores
 
def calculate_cohort_priority(row: pd.Series) -> float:

    """Priority based on cohort behavioral scores"""

    return (row['activity_score'] * 0.4 + 

            row['conviction_score'] * 0.3 + 

            row['recency_score'] * 0.2

           )

# **UPDATED RECOMMENDATION ENGINE - FULL COHORT USAGE**

def generate_cohort_recommendations(notifications_df: pd.DataFrame, cohorts_df: pd.DataFrame, 

                                  metadata_df: pd.DataFrame) -> pd.DataFrame:

    """Recommendation engine using COMPLETE cohort matching"""

    recommendations = []

    notifications_enriched = notifications_df.merge(metadata_df, on='symbol', how='left')

    for _, cohort in cohorts_df.iterrows():

        ucc = cohort['UCC']

        Symbol = cohort['Symbol']

        stock_holding_Flag = cohort['IsHolding']


        cohort_priority = calculate_cohort_priority(cohort)

        cohort_parts = parse_cohort_full(cohort['cohort'])

        for _, notif in notifications_enriched.iterrows():

            cohort_scores = match_using_full_cohort(notif, cohort)

            # FINAL SCORE = Cohort Score × Behavioral Priority

            final_score = cohort_scores['total_cohort_score'] * cohort_priority / 10.0

            if cohort_scores['total_cohort_score'] == 0:

                # Mark as no-notification with blank columns

                recommendations.append({

                    'UCC': ucc,

                    'most_interactive_user_symbol': Symbol,

                    'isHolding': stock_holding_Flag,

                    'Notification_Symbol': 'no-notification',

                    'Notif_Category': notif['marketcap'],

                    'Notif_Sector': notif['sector'],

                    'Cohort_Category': cohort_parts['cust_category'],

                    'Cohort_Cluster': cohort_parts['cust_cluster'],

                    'Cohort_Sector': cohort_parts['cust_sector'],                   

                    'Category_Match': cohort_scores['category_match'],

                    'Cluster_Match': cohort_scores['cluster_match'],

                    'Sector_Match': cohort_scores['sector_match'],

                    'Total_Cohort_Score': round(cohort_scores['total_cohort_score'], 2),

                    'Behavioral_Priority': round(cohort_priority, 2),

                    'Final_Score': round(final_score, 2),

                    'Entry_Price': notif['Entry_Avg'],

                    'Target': notif['Target']

                })

            elif final_score > 2.0:  # Cohort threshold

                recommendations.append({

                    'UCC': ucc,

                    'most_interactive_user_symbol': Symbol,

                    'isHolding': stock_holding_Flag,

                    'Notification_Symbol': notif['symbol'],

                    'Notif_Category': notif['marketcap'],

                    'Notif_Sector': notif['sector'],

                    'Cohort_Category': cohort_parts['cust_category'],

                    'Cohort_Cluster': cohort_parts['cust_cluster'],

                    'Cohort_Sector': cohort_parts['cust_sector'],

                    'Category_Match': cohort_scores['category_match'],

                    'Cluster_Match': cohort_scores['cluster_match'],

                    'Sector_Match': cohort_scores['sector_match'],

                    'Total_Cohort_Score': round(cohort_scores['total_cohort_score'], 2),

                    'Behavioral_Priority': round(cohort_priority, 2),

                    'Final_Score': round(final_score, 2),

                    'Entry_Price': notif['Entry_Avg'],

                    'Target': notif['Target']

                })

    rec_df = pd.DataFrame(recommendations)

    return rec_df.sort_values(['UCC', 'Final_Score'], ascending=[True, False])
 
 
 