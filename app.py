import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from collections import Counter
from scipy.stats import poisson
from scipy.special import factorial
from scipy.optimize import minimize
import math
import hashlib
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
import requests  # 新增用于 API 请求

# ========== Football-Data.org API 配置 ==========
FOOTBALL_DATA_API_KEY = "e081ba0cb14245668dc3ce2de29e7ff2"  # 请替换为真实 Key
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4/"

# ========== 启动诊断 ==========
try:
    _ = st.session_state.keys()
except Exception:
    st.error("❌ 请通过 'streamlit run app.py' 启动应用！")
    st.stop()

if 'success_message' in st.session_state:
    st.success(st.session_state['success_message'])
    del st.session_state['success_message']

# ========== 历史记录管理 ==========
RESULTS_FILE = "match_results.csv"

def load_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=['日期','主队','客队','主队进球','客队进球','比分','盘口','实际盘口','预测结果',
                                     'λ_主队初','λ_主队即','λ_客队初','λ_客队即','主胜概率','主先进球概率','先进球方',
                                     'odds_fingerprint'])
    try:
        df = pd.read_csv(RESULTS_FILE, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(RESULTS_FILE, encoding='gbk')
        except Exception as e:
            st.error(f"读取历史记录文件编码错误: {e}")
            return pd.DataFrame(columns=['日期','主队','客队','主队进球','客队进球','比分','盘口','实际盘口','预测结果',
                                         'λ_主队初','λ_主队即','λ_客队初','λ_客队即','主胜概率','主先进球概率','先进球方',
                                         'odds_fingerprint'])
    except PermissionError:
        st.error("❌ 无法读取历史记录文件，请关闭可能占用该文件的程序（如Excel），然后刷新页面。")
        return pd.DataFrame(columns=['日期','主队','客队','主队进球','客队进球','比分','盘口','实际盘口','预测结果',
                                     'λ_主队初','λ_主队即','λ_客队初','λ_客队即','主胜概率','主先进球概率','先进球方',
                                     'odds_fingerprint'])
    except Exception as e:
        st.error(f"读取历史记录文件出错: {e}")
        return pd.DataFrame(columns=['日期','主队','客队','主队进球','客队进球','比分','盘口','实际盘口','预测结果',
                                     'λ_主队初','λ_主队即','λ_客队初','λ_客队即','主胜概率','主先进球概率','先进球方',
                                     'odds_fingerprint'])
    required_cols = ['日期','主队','客队','主队进球','客队进球','比分','盘口','实际盘口','预测结果',
                     'λ_主队初','λ_主队即','λ_客队初','λ_客队即','主胜概率','主先进球概率','先进球方',
                     'odds_fingerprint']
    for col in required_cols:
        if col not in df.columns:
            if col == '先进球方':
                df[col] = ''
            elif col == 'odds_fingerprint':
                df[col] = ''
            else:
                df[col] = np.nan
    if 'λ_主队初' not in df.columns and 'λ_主队' in df.columns:
        df['λ_主队初'] = df['λ_主队']
        df['λ_主队即'] = df['λ_主队']
    if 'λ_客队初' not in df.columns and 'λ_客队' in df.columns:
        df['λ_客队初'] = df['λ_客队']
        df['λ_客队即'] = df['λ_客队']
    for col in ['λ_主队初', 'λ_主队即', 'λ_客队初', 'λ_客队即', '主胜概率', '主先进球概率']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if '比分' in df.columns:
        df['比分'] = df['比分'].astype(str)
    else:
        df['比分'] = ''
    if 'odds_fingerprint' not in df.columns:
        df['odds_fingerprint'] = ''
    return df

def save_result(date, home_team, away_team, home_score, away_score, handicap, actual_handicap, predict_result,
                lam_h_init, lam_h_live, lam_a_init, lam_a_live, home_prob, home_first, first_scorer, odds_fingerprint=''):
    df = load_results()
    new_row = pd.DataFrame([{
        '日期': date,
        '主队': home_team,
        '客队': away_team,
        '主队进球': home_score,
        '客队进球': away_score,
        '比分': f"{home_score}:{away_score}",
        '盘口': handicap,
        '实际盘口': actual_handicap,
        '预测结果': predict_result,
        'λ_主队初': lam_h_init,
        'λ_主队即': lam_h_live,
        'λ_客队初': lam_a_init,
        'λ_客队即': lam_a_live,
        '主胜概率': home_prob,
        '主先进球概率': home_first,
        '先进球方': first_scorer,
        'odds_fingerprint': odds_fingerprint
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    try:
        df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
    except PermissionError:
        st.error("❌ 无法保存比赛结果，请关闭可能占用 'match_results.csv' 文件的程序（如Excel），然后重试。")
        return None
    except Exception as e:
        st.error(f"保存文件时出错: {e}")
        return None
    return df

def is_abnormal_match(row):
    if row['主队进球'] >= 4 or row['客队进球'] >= 4:
        return True
    first_scorer = row.get('先进球方', '')
    if first_scorer == '主队':
        if row['主队进球'] <= row['客队进球']:
            return True
    elif first_scorer == '客队':
        if row['客队进球'] <= row['主队进球']:
            return True
    if pd.notna(row['实际盘口']) and pd.notna(row['盘口']):
        if abs(row['实际盘口'] - row['盘口']) > 1:
            return True
    return False

def is_score_col(col):
    col = str(col).strip()
    return bool(re.match(r'^\d+[:：\-]\d+$', col))

def normalize_score_col(col):
    return re.sub(r'[：\-]', ':', str(col).strip())

def parse_pasted_text(text):
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return None, "至少需要两行数据"
    headers_raw = re.split(r'\s+', lines[0].strip())
    headers = [h.strip() for h in headers_raw if h.strip()]
    drop_first_col = False
    if headers and (headers[0] in ['序', '序号', 'No', 'no']):
        drop_first_col = True
        headers = headers[1:]
    elif headers and re.match(r'^\d+$', headers[0]):
        drop_first_col = True
        headers = headers[1:]
    rows = []
    last_company = None
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r'\s+', line.strip())
        if drop_first_col and len(parts) > 0:
            if re.match(r'^\d+$', parts[0]):
                parts = parts[1:]
        if len(parts) >= 1 and parts[0] == '即':
            if last_company is None:
                return None, "无法找到“即”行对应的公司名，请确保初盘行在前且有公司名"
            row = [last_company, '即'] + parts[1:]
        else:
            if len(parts) < 2:
                continue
            company = parts[0]
            category = parts[1] if len(parts) > 1 else ''
            last_company = company
            row = parts
        if len(row) < len(headers):
            row.extend([''] * (len(headers) - len(row)))
        else:
            row = row[:len(headers)]
        rows.append(row)
    if not rows:
        return None, "未找到有效数据行"
    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        if is_score_col(col):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if '分类' not in df.columns:
        for col in df.columns:
            if df[col].astype(str).str.contains('初|即').any():
                df.rename(columns={col: '分类'}, inplace=True)
                break
    return df, None

def compute_odds_fingerprint(df):
    score_cols = [c for c in df.columns if is_score_col(c)]
    if not score_cols:
        return None
    norm_score_cols = {c: normalize_score_col(c) for c in score_cols}
    records = []
    for _, row in df.iterrows():
        company = row.iloc[0] if len(row) > 0 else ""
        cat_col = None
        for col in df.columns:
            if '分类' in col:
                cat_col = col
                break
        if cat_col is None:
            for col in df.columns:
                if df[col].astype(str).str.contains('初|即').any():
                    cat_col = col
                    break
        category = row[cat_col] if cat_col and cat_col in row else ""
        odds = {}
        for orig, std in norm_score_cols.items():
            val = row[orig]
            if pd.notna(val) and val > 0:
                odds[std] = float(val)
        if odds:
            records.append((company, category, odds))
    records.sort(key=lambda x: (x[0], x[1]))
    fingerprint_str = json.dumps(records, sort_keys=True)
    return hashlib.md5(fingerprint_str.encode()).hexdigest()

# ========== 非线性优化反推λ ==========
def optimize_lambdas(odds_dict, max_goals=4):
    scores = list(odds_dict.keys())
    odds = np.array([odds_dict[s] for s in scores])
    raw_probs = 1.0 / odds
    total_raw = np.sum(raw_probs)
    overround = 1.0 / total_raw
    fair_probs = raw_probs / total_raw

    score_index = {}
    idx = 0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            score_index[f"{x}:{y}"] = idx
            idx += 1
    target_probs = np.zeros((max_goals+1)*(max_goals+1))
    for s, p in zip(scores, fair_probs):
        if s in score_index:
            target_probs[score_index[s]] = p

    def poisson_probs(lam_h, lam_a):
        probs = []
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                probs.append(prob)
        return np.array(probs)

    def objective(params):
        lam_h, lam_a = params
        if lam_h <= 0 or lam_a <= 0:
            return 1e10
        model_probs = poisson_probs(lam_h, lam_a)
        diff = model_probs - target_probs
        return np.sum(diff**2)

    home_marginal = {}
    away_marginal = {}
    for s, p in zip(scores, fair_probs):
        x, y = map(int, s.split(':'))
        home_marginal[x] = home_marginal.get(x, 0.0) + p
        away_marginal[y] = away_marginal.get(y, 0.0) + p
    lam_h_init = sum(x * home_marginal.get(x, 0.0) for x in range(max_goals+1))
    lam_a_init = sum(y * away_marginal.get(y, 0.0) for y in range(max_goals+1))
    lam_h_init = max(0.1, min(5.0, lam_h_init))
    lam_a_init = max(0.1, min(5.0, lam_a_init))

    result = minimize(objective, [lam_h_init, lam_a_init], bounds=[(0.1, 5.0), (0.1, 5.0)], method='L-BFGS-B')
    lam_h_opt, lam_a_opt = result.x
    model_probs = poisson_probs(lam_h_opt, lam_a_opt)
    mask = target_probs > 0
    if np.any(mask):
        rmse = np.sqrt(np.mean((model_probs[mask] - target_probs[mask])**2))
    else:
        rmse = np.nan
    return lam_h_opt, lam_a_opt, overround, rmse

def compute_goal_distribution_optimized(odds_dict):
    lam_h, lam_a, overround, rmse = optimize_lambdas(odds_dict)
    raw = {bt: 1.0/odd for bt, odd in odds_dict.items() if odd>0}
    total = sum(raw.values())
    prob_bt = {bt: p/total for bt, p in raw.items()}
    home_goals = {0:0,1:0,2:0,3:0,4:0}
    away_goals = {0:0,1:0,2:0,3:0,4:0}
    away_zero = 0.0
    for bt, p in prob_bt.items():
        try:
            h,a = map(int, bt.split(':'))
            if h <= 4:
                home_goals[h] += p
            if a <= 4:
                away_goals[a] += p
            if a == 0:
                away_zero += p
        except:
            continue
    poisson_h = {g: poisson.pmf(g, lam_h) for g in range(5)} if lam_h>0 else {g:0 for g in range(5)}
    poisson_a = {g: poisson.pmf(g, lam_a) for g in range(5)} if lam_a>0 else {g:0 for g in range(5)}
    return home_goals, poisson_h, lam_h, away_zero, lam_a, away_goals, poisson_a, overround, rmse

def process_dataframe(df):
    company_col = None
    for col in df.columns:
        if '公司' in col or '序' in col:
            company_col = col
            break
    if company_col is None:
        company_col = df.columns[0]
    type_col = None
    for col in df.columns:
        if '分类' in col:
            type_col = col
            break
    if type_col is None:
        for col in df.columns:
            if df[col].astype(str).str.contains('初|即').any():
                type_col = col
                break
    score_cols = [c for c in df.columns if is_score_col(c)]
    if not score_cols:
        return None, None, "未识别到波胆列", None, None, None, (None, None, None, None), 0, 0
    norm = {c: normalize_score_col(c) for c in score_cols}
    results = []
    goal_analysis = []
    most_likely_scores = []
    over_probs = []
    under_probs = []
    init_lam_h_list = []
    init_lam_a_list = []
    live_lam_h_list = []
    live_lam_a_list = []
    init_overround_list = []
    live_overround_list = []
    init_rmse_list = []
    live_rmse_list = []
    live_company_info = []
    for idx, row in df.iterrows():
        odds = {}
        for orig in score_cols:
            std = norm[orig]
            v = row[orig]
            if pd.notna(v) and v>0:
                odds[std] = float(v)
        if not odds:
            continue
        home_goals, poisson_h, lam_h, away_zero, lam_a, away_goals, poisson_a, overround, rmse = compute_goal_distribution_optimized(odds)
        raw_home = sum(1.0/odds[bt] for bt in odds if bt in ['1:0','2:0','2:1','3:0','3:1','3:2','4:0','4:1','4:2','4:3'])
        raw_draw = sum(1.0/odds[bt] for bt in odds if bt in ['0:0','1:1','2:2','3:3','4:4'])
        raw_away = sum(1.0/odds[bt] for bt in odds if bt in ['0:1','0:2','1:2','0:3','1:3','2:3','0:4','1:4','2:4','3:4'])
        total_raw = raw_home + raw_draw + raw_away
        if total_raw > 0:
            odds_home = 1.0 / (raw_home / total_raw)
            odds_draw = 1.0 / (raw_draw / total_raw)
            odds_away = 1.0 / (raw_away / total_raw)
            home_p = raw_home / total_raw
            draw_p = raw_draw / total_raw
            away_p = raw_away / total_raw
        else:
            odds_home = odds_draw = odds_away = 0.0
            home_p = draw_p = away_p = 0.0
        min_bt = min(odds.items(), key=lambda x: x[1])[0] if odds else "未知"
        max_p = 1.0/odds[min_bt]/total_raw if total_raw>0 and min_bt in odds else 0
        home_first = home_p + 0.5*draw_p
        away_first = away_p + 0.5*draw_p
        company = row[company_col] if company_col in row else f"行{idx+1}"
        cat = row[type_col] if type_col and type_col in row and pd.notna(row[type_col]) else ""
        results.append({
            '公司': company, '分类': cat,
            '主胜': home_p, '平局': draw_p, '客胜': away_p,
            '主先进球': home_first, '客先进球': away_first,
            '最可能比分': min_bt, '比分概率': max_p,
            '返还率': overround,
            'RMSE': rmse,
            '赔率主': odds_home, '赔率平': odds_draw, '赔率客': odds_away
        })
        most_likely_scores.append(min_bt)
        goal_analysis.append({
            '标识': f"{company} ({cat})" if cat else company,
            '分类': cat,
            'λ(主队)': lam_h,
            'λ(客队)': lam_a,
            '主胜概率': home_p,
            '主先进球概率': home_first,
            '主队0球(边际)': home_goals[0],
            '客队0球(边际)': away_zero,
            '主队1球(边际)': home_goals[1],
            '主队2球(边际)': home_goals[2],
            '主队3球(边际)': home_goals[3],
            '主队4球(边际)': home_goals[4],
            '客队1球(边际)': away_goals[1],
            '客队2球(边际)': away_goals[2],
            '客队3球(边际)': away_goals[3],
            '客队4球(边际)': away_goals[4],
            '泊松主0球': poisson_h[0], '泊松主1球': poisson_h[1], '泊松主2球': poisson_h[2], '泊松主3球': poisson_h[3], '泊松主4球': poisson_h[4],
            '泊松客0球': poisson_a[0], '泊松客1球': poisson_a[1], '泊松客2球': poisson_a[2], '泊松客3球': poisson_a[3], '泊松客4球': poisson_a[4],
            '返还率': overround,
            'RMSE': rmse
        })
        if cat == '初':
            init_lam_h_list.append(lam_h)
            init_lam_a_list.append(lam_a)
            init_overround_list.append(overround)
            init_rmse_list.append(rmse)
        elif cat == '即':
            live_lam_h_list.append(lam_h)
            live_lam_a_list.append(lam_a)
            live_overround_list.append(overround)
            live_rmse_list.append(rmse)
            live_company_info.append((company, odds.copy(), rmse, odds_home, odds_draw, odds_away, lam_h, lam_a))
        over = sum(1.0/odds[bt] for bt in odds if (lambda h,a: h+a>=3)(*map(int,bt.split(':'))))
        under = total_raw - over
        if over > 0:
            over_probs.append(over/total_raw)
            under_probs.append(under/total_raw)
    if not results:
        return None, None, "未提取到有效赔率", None, None, None, (None, None, None, None), 0, 0
    most_common_score = Counter(most_likely_scores).most_common(1)[0][0] if most_likely_scores else "未知"
    avg_over = np.mean(over_probs) if over_probs else 0.5
    avg_under = np.mean(under_probs) if under_probs else 0.5
    avg_lam_h_init = np.mean(init_lam_h_list) if init_lam_h_list else None
    avg_lam_a_init = np.mean(init_lam_a_list) if init_lam_a_list else None
    avg_lam_h_live = np.mean(live_lam_h_list) if live_lam_h_list else None
    avg_lam_a_live = np.mean(live_lam_a_list) if live_lam_a_list else None
    lam_h_final = avg_lam_h_live if avg_lam_h_live is not None else (avg_lam_h_init if avg_lam_h_init is not None else 0)
    lam_a_final = avg_lam_a_live if avg_lam_a_live is not None else (avg_lam_a_init if avg_lam_a_init is not None else 0)
    avg_init_overround = np.mean(init_overround_list) if init_overround_list else None
    avg_live_overround = np.mean(live_overround_list) if live_overround_list else None
    avg_init_rmse = np.mean(init_rmse_list) if init_rmse_list else None
    avg_live_rmse = np.mean(live_rmse_list) if live_rmse_list else None
    st.session_state['avg_init_overround'] = avg_init_overround
    st.session_state['avg_live_overround'] = avg_live_overround
    st.session_state['avg_init_rmse'] = avg_init_rmse
    st.session_state['avg_live_rmse'] = avg_live_rmse
    st.session_state['live_company_info'] = live_company_info
    if type_col:
        companies = {}
        for r in results:
            comp = r['公司']; cat = r['分类']
            companies.setdefault(comp, {})[cat] = r
        compare = []
        for comp, items in companies.items():
            if '初' in items and '即' in items:
                init, live = items['初'], items['即']
                compare.append({
                    '公司': comp,
                    '主胜初': f"{init['主胜']:.2%}",
                    '主胜即': f"{live['主胜']:.2%}",
                    '主胜变化': f"{(live['主胜']-init['主胜'])*100:+.2f}%",
                    '平局初': f"{init['平局']:.2%}",
                    '平局即': f"{live['平局']:.2%}",
                    '平局变化': f"{(live['平局']-init['平局'])*100:+.2f}%",
                    '客胜初': f"{init['客胜']:.2%}",
                    '客胜即': f"{live['客胜']:.2%}",
                    '客胜变化': f"{(live['客胜']-init['客胜'])*100:+.2f}%",
                    '主先进球初': f"{init['主先进球']:.2%}",
                    '主先进球即': f"{live['主先进球']:.2%}",
                    '主先进球变化': f"{(live['主先进球']-init['主先进球'])*100:+.2f}%",
                    '客先进球初': f"{init['客先进球']:.2%}",
                    '客先进球即': f"{live['客先进球']:.2%}",
                    '客先进球变化': f"{(live['客先进球']-init['客先进球'])*100:+.2f}%",
                    '最可能初': init['最可能比分'],
                    '最可能即': live['最可能比分'],
                    '返还率初': f"{init['返还率']:.2%}" if '返还率' in init else "N/A",
                    '返还率即': f"{live['返还率']:.2%}" if '返还率' in live else "N/A",
                    'RMSE初': f"{init['RMSE']:.4f}" if 'RMSE' in init else "N/A",
                    'RMSE即': f"{live['RMSE']:.4f}" if 'RMSE' in live else "N/A",
                })
            else:
                for cat, data in items.items():
                    compare.append({
                        '公司': f"{comp} ({cat})",
                        '主胜': f"{data['主胜']:.2%}",
                        '平局': f"{data['平局']:.2%}",
                        '客胜': f"{data['客胜']:.2%}",
                        '主先进球': f"{data['主先进球']:.2%}",
                        '客先进球': f"{data['客先进球']:.2%}",
                        '最可能比分': data['最可能比分'],
                        '概率': f"{data['比分概率']:.2%}",
                        '返还率': f"{data['返还率']:.2%}" if '返还率' in data else "N/A",
                        'RMSE': f"{data['RMSE']:.4f}" if 'RMSE' in data else "N/A"
                    })
        return compare, goal_analysis, None, most_common_score, avg_over, avg_under, (avg_lam_h_init, avg_lam_a_init, avg_lam_h_live, avg_lam_a_live), lam_h_final, lam_a_final
    else:
        simple = [{
            '公司': r['公司'],
            '主胜': f"{r['主胜']:.2%}",
            '平局': f"{r['平局']:.2%}",
            '客胜': f"{r['客胜']:.2%}",
            '主先进球': f"{r['主先进球']:.2%}",
            '客先进球': f"{r['客先进球']:.2%}",
            '最可能比分': r['最可能比分'],
            '概率': f"{r['比分概率']:.2%}",
            '返还率': f"{r['返还率']:.2%}" if '返还率' in r else "N/A",
            'RMSE': f"{r['RMSE']:.4f}" if 'RMSE' in r else "N/A"
        } for r in results]
        return simple, goal_analysis, None, most_common_score, avg_over, avg_under, (avg_lam_h_init, avg_lam_a_init, avg_lam_h_live, avg_lam_a_live), lam_h_final, lam_a_final

# ========== 盘口映射 ==========
def handicap_from_diff(diff):
    options = [-2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    diff_clip = max(-1.5, min(1.5, diff))
    raw = - diff_clip * 0.5
    return min(options, key=lambda x: abs(x - raw))

def compute_global_rho():
    df = load_results()
    if df.empty or '主队进球' not in df.columns or '客队进球' not in df.columns:
        return 0.0
    home_goals = pd.to_numeric(df['主队进球'], errors='coerce').fillna(0)
    away_goals = pd.to_numeric(df['客队进球'], errors='coerce').fillna(0)
    if len(home_goals) < 2:
        return 0.0
    corr = home_goals.corr(away_goals)
    return corr if not np.isnan(corr) else 0.0

@st.cache_data
def get_global_rho():
    return compute_global_rho()

def bivariate_poisson_prob(lam1, lam2, rho, x, y):
    if rho == 0:
        return poisson.pmf(x, lam1) * poisson.pmf(y, lam2)
    total = 0.0
    for k in range(min(x, y) + 1):
        term = (rho / (lam1 * lam2)) ** k
        term *= math.comb(x, k) * math.comb(y, k) * math.factorial(k)
        total += term
    return np.exp(-(lam1 + lam2 + rho)) * (lam1 ** x) * (lam2 ** y) / (math.factorial(x) * math.factorial(y)) * total

def bivariate_poisson_vector(lam1, lam2, rho, max_goals=4):
    vec = []
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            prob = bivariate_poisson_prob(lam1, lam2, rho, x, y)
            vec.append(prob)
    return np.array(vec)

def get_similar_matches(lam_h_live, lam_a_live, home_prob, home_first, first_scorer, top_n=20, lam_h_init=None, lam_a_init=None):
    df = load_results()
    if df.empty:
        return []
    candidate = df.copy()
    if first_scorer and first_scorer != "":
        candidate = candidate[candidate['先进球方'] == first_scorer]
    if candidate.empty:
        return []
    for col in ['λ_主队即','λ_客队即','主胜概率','主先进球概率','λ_主队初','λ_客队初']:
        if col not in candidate.columns:
            candidate[col] = 0.0
        else:
            candidate[col] = pd.to_numeric(candidate[col], errors='coerce').fillna(0)
    
    curr_h_diff = (lam_h_live if lam_h_live is not None else 0) - (lam_h_init if lam_h_init is not None else 0)
    curr_a_diff = (lam_a_live if lam_a_live is not None else 0) - (lam_a_init if lam_a_init is not None else 0)
    curr_diff_vec = np.array([curr_h_diff, curr_a_diff])
    norm_diff_curr = np.linalg.norm(curr_diff_vec)
    curr_diff_unit = curr_diff_vec / norm_diff_curr if norm_diff_curr > 0 else curr_diff_vec
    
    global_rho = get_global_rho()
    lam_h_curr = lam_h_live if lam_h_live is not None else 0
    lam_a_curr = lam_a_live if lam_a_live is not None else 0
    curr_biv_vec = bivariate_poisson_vector(lam_h_curr, lam_a_curr, global_rho, max_goals=4)
    norm_biv_curr = np.linalg.norm(curr_biv_vec)
    if norm_biv_curr == 0:
        return []
    curr_biv_unit = curr_biv_vec / norm_biv_curr
    
    curr_aux_vec = np.array([
        lam_h_live if lam_h_live is not None else 0,
        lam_a_live if lam_a_live is not None else 0,
        home_prob if home_prob is not None else 0,
        home_first if home_first is not None else 0
    ])
    norm_aux_curr = np.linalg.norm(curr_aux_vec)
    curr_aux_unit = curr_aux_vec / norm_aux_curr if norm_aux_curr > 0 else curr_aux_vec
    
    def combined_similarity(row):
        hist_h_diff = row['λ_主队即'] - row['λ_主队初']
        hist_a_diff = row['λ_客队即'] - row['λ_客队初']
        hist_diff_vec = np.array([hist_h_diff, hist_a_diff])
        norm_diff_hist = np.linalg.norm(hist_diff_vec)
        if norm_diff_hist == 0:
            diff_sim = 0.0
        else:
            hist_diff_unit = hist_diff_vec / norm_diff_hist
            diff_sim = np.dot(curr_diff_unit, hist_diff_unit)
        
        lam_h = row['λ_主队即']
        lam_a = row['λ_客队即']
        hist_biv_vec = bivariate_poisson_vector(lam_h, lam_a, global_rho, max_goals=4)
        norm_biv_hist = np.linalg.norm(hist_biv_vec)
        if norm_biv_hist == 0:
            biv_sim = 0.0
        else:
            hist_biv_unit = hist_biv_vec / norm_biv_hist
            biv_sim = np.dot(curr_biv_unit, hist_biv_unit)
        
        hist_aux_vec = np.array([
            row['λ_主队即'],
            row['λ_客队即'],
            row['主胜概率'],
            row['主先进球概率']
        ])
        norm_aux_hist = np.linalg.norm(hist_aux_vec)
        if norm_aux_hist == 0:
            aux_sim = 0.0
        else:
            hist_aux_unit = hist_aux_vec / norm_aux_hist
            aux_sim = np.dot(curr_aux_unit, hist_aux_unit)
        
        base_sim = 0.4 * diff_sim + 0.4 * biv_sim + 0.2 * aux_sim
        if pd.notna(row['实际盘口']) and pd.notna(row['盘口']):
            diff_abs = abs(row['实际盘口'] - row['盘口'])
            adjust = max(0.85, min(1.0, 1 - 0.1 * diff_abs))
        else:
            adjust = 1.0
        return base_sim * adjust
    
    candidate['sim_score'] = candidate.apply(combined_similarity, axis=1)
    candidate_sorted = candidate.sort_values('sim_score', ascending=False)
    filtered = candidate_sorted[candidate_sorted['sim_score'] >= 0.85]
    similar = filtered.head(top_n)
    results = []
    for _, row in similar.iterrows():
        score_str = str(row['比分']) if pd.notna(row['比分']) else ""
        if not re.match(r'^\d+:\d+$', score_str):
            continue
        results.append({
            '日期': row['日期'],
            '主队': row['主队'],
            '客队': row['客队'],
            '比分': score_str,
            '盘口': row['盘口'],
            '实际盘口': row['实际盘口'],
            'λ_主队初': row['λ_主队初'],
            'λ_主队即': row['λ_主队即'],
            'λ_客队初': row['λ_客队初'],
            'λ_客队即': row['λ_客队即'],
            '主胜概率': row['主胜概率'],
            '主先进球概率': row['主先进球概率'],
            '预测结果': row['预测结果'],
            '先进球方': row['先进球方'],
            'sim_score': row['sim_score']
        })
    return results

def over_under_recommendation_historical(similar_matches, min_support=0.8, min_matches=6):
    if not similar_matches or len(similar_matches) < min_matches:
        return None, None
    goals = []
    for m in similar_matches:
        try:
            h, a = map(int, m['比分'].split(':'))
            total = h + a
            if total >= 7:
                continue
            goals.append(total)
        except:
            continue
    if not goals:
        return None, None
    over_count = sum(1 for g in goals if g >= 3)
    under_count = len(goals) - over_count
    over_ratio = over_count / len(goals)
    if over_ratio >= min_support:
        return f"大2.5 (支持率 {over_ratio:.0%})", over_ratio
    elif (1 - over_ratio) >= min_support:
        return f"小2.5 (支持率 {under_count/len(goals):.0%})", over_ratio
    else:
        return None, None

def get_star_rating_from_similarity(sim_score):
    if sim_score >= 0.999:
        return "⭐⭐⭐⭐⭐ (五星推荐)", sim_score
    elif sim_score >= 0.99:
        return "⭐⭐⭐⭐ (四星推荐)", sim_score
    elif sim_score >= 0.98:
        return "⭐⭐⭐ (三星推荐)", sim_score
    elif sim_score >= 0.97:
        return "⭐⭐ (二星推荐)", sim_score
    else:
        return None, sim_score

# ========== 球队近期实力分析（含比赛结果加减分调整） ==========
def adjust_lambda_by_result(lam, row, team_name):
    try:
        home_goals = int(row['比分'].split(':')[0])
        away_goals = int(row['比分'].split(':')[1])
    except:
        return lam
    actual_handicap = row.get('实际盘口', 0)
    if pd.isna(actual_handicap):
        return lam
    if row['主队'] == team_name:
        my_goals = home_goals
        opp_goals = away_goals
        is_home = True
        handicap = actual_handicap
    else:
        my_goals = away_goals
        opp_goals = home_goals
        is_home = False
        handicap = -actual_handicap
    net = my_goals - opp_goals
    if handicap > 0:
        if net > handicap:
            result = 'win'
        elif net == handicap:
            result = 'push'
        else:
            result = 'lose'
    elif handicap < 0:
        if net > handicap:
            result = 'win'
        elif net == handicap:
            result = 'push'
        else:
            result = 'lose'
    else:
        if net > 0:
            result = 'win'
        elif net == 0:
            result = 'push'
        else:
            result = 'lose'
    adjusted_lam = lam
    if result == 'win':
        adjusted_lam += 0.1
        if handicap >= 0 and net - handicap >= 1:
            adjusted_lam += 0.05
        elif handicap < 0 and net - handicap >= 1:
            adjusted_lam += 0.05
    elif result == 'lose':
        adjusted_lam -= 0.1
        if handicap >= 0 and net - handicap <= -1:
            adjusted_lam -= 0.05
        elif handicap < 0 and net - handicap <= -1:
            adjusted_lam -= 0.05
    if is_home:
        adjusted_lam += 0.05
    return max(0.1, adjusted_lam)

def get_team_recent_strength(team_name, n_matches=5, apply_result_adjust=True):
    df = load_results()
    if df.empty:
        return None, None, [], 0
    clean_team = team_name.strip()
    df['主队_clean'] = df['主队'].astype(str).str.strip()
    df['客队_clean'] = df['客队'].astype(str).str.strip()
    mask = (df['主队_clean'].str.lower() == clean_team.lower()) | (df['客队_clean'].str.lower() == clean_team.lower())
    df_team = df[mask].copy()
    if df_team.empty:
        return None, None, [], 0
    df_team['日期'] = pd.to_datetime(df_team['日期'])
    df_team = df_team.sort_values('日期', ascending=False)
    df_recent = df_team.head(n_matches)
    raw_lambdas = []
    adjusted_lambdas = []
    details = []
    for _, row in df_recent.iterrows():
        if row['主队_clean'].lower() == clean_team.lower():
            lam = row['λ_主队即']
            role = "主队"
            opponent = row['客队']
        else:
            lam = row['λ_客队即']
            role = "客队"
            opponent = row['主队']
        if pd.notna(lam) and lam > 0:
            raw_lambdas.append(lam)
            if apply_result_adjust:
                adjusted_lam = adjust_lambda_by_result(lam, row, clean_team)
            else:
                adjusted_lam = lam
            adjusted_lambdas.append(adjusted_lam)
            details.append({
                '日期': row['日期'].strftime('%Y-%m-%d'),
                '对手': opponent,
                '角色': role,
                '比分': row['比分'],
                '原始λ': f"{lam:.3f}",
                '调整值': f"{adjusted_lam - lam:+.3f}",
                '调整后λ': f"{adjusted_lam:.3f}"
            })
        else:
            details.append({
                '日期': row['日期'].strftime('%Y-%m-%d'),
                '对手': opponent,
                '角色': role,
                '比分': row['比分'],
                '原始λ': "无效",
                '调整值': "-",
                '调整后λ': "-"
            })
    if not raw_lambdas:
        return None, None, details, len(details)
    avg_raw = np.mean(raw_lambdas)
    avg_adjusted = np.mean(adjusted_lambdas) if adjusted_lambdas else avg_raw
    return avg_raw, avg_adjusted, details, len(details)

def compute_poisson_probs_from_lam(lam_h, lam_a, max_goals=4):
    home_probs = {g: poisson.pmf(g, lam_h) for g in range(max_goals+1)}
    away_probs = {g: poisson.pmf(g, lam_a) for g in range(max_goals+1)}
    max_prob = 0
    most_likely = "0:0"
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            p = home_probs[x] * away_probs[y]
            if p > max_prob:
                max_prob = p
                most_likely = f"{x}:{y}"
    return home_probs, away_probs, most_likely, max_prob

def compute_top_scores(lam_h, lam_a, max_goals=4, top_n=5):
    scores = []
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
            scores.append((f"{x}:{y}", prob))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def over_under_from_lam(lam_h, lam_a):
    total = lam_h + lam_a
    if total > 2.8:
        return f"大2.5 (预期总进球 {total:.2f} 球)"
    elif total < 2.2:
        return f"小2.5 (预期总进球 {total:.2f} 球)"
    else:
        return f"未能确定（预期总进球 {total:.2f} 球接近2.5）"

# ========== 机器学习模型（随机森林，三分类：输盘/走水/赢盘） ==========
MODEL_ASIAN = "asian_model.pkl"
MODEL_OU = "ou_model.pkl"
SCALER_ASIAN = "asian_scaler.pkl"
SCALER_OU = "ou_scaler.pkl"

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def build_features_for_match(row, df_history):
    home = row['主队']
    away = row['客队']
    lam_h = row.get('λ_主队即', 0.0)
    lam_a = row.get('λ_客队即', 0.0)
    actual_hc = row.get('实际盘口', 0.0)
    features = []
    lam_diff = lam_h - lam_a
    features.append(lam_diff)
    total_lam = lam_h + lam_a
    features.append(total_lam)
    home_matches = df_history[(df_history['主队'] == home) | (df_history['客队'] == home)].copy()
    if not home_matches.empty:
        home_matches = home_matches.sort_values('日期', ascending=False).head(5)
        home_win_rate = 0.0
        cnt = 0
        for _, m in home_matches.iterrows():
            try:
                h, a = map(int, m['比分'].split(':'))
                net = h - a
                if m['主队'] == home:
                    hc = m['实际盘口']
                else:
                    hc = -m['实际盘口']
                if pd.notna(hc):
                    if hc > 0:
                        win = net > hc
                    elif hc < 0:
                        win = net > hc
                    else:
                        win = net > 0
                    home_win_rate += win
                    cnt += 1
            except:
                continue
        if cnt > 0:
            home_win_rate /= cnt
        features.append(home_win_rate)
    else:
        features.append(0.5)
    away_matches = df_history[(df_history['主队'] == away) | (df_history['客队'] == away)].copy()
    if not away_matches.empty:
        away_matches = away_matches.sort_values('日期', ascending=False).head(5)
        away_win_rate = 0.0
        cnt = 0
        for _, m in away_matches.iterrows():
            try:
                h, a = map(int, m['比分'].split(':'))
                net = h - a
                if m['主队'] == away:
                    hc = m['实际盘口']
                else:
                    hc = -m['实际盘口']
                if pd.notna(hc):
                    if hc > 0:
                        win = net > hc
                    elif hc < 0:
                        win = net > hc
                    else:
                        win = net > 0
                    away_win_rate += win
                    cnt += 1
            except:
                continue
        if cnt > 0:
            away_win_rate /= cnt
        features.append(away_win_rate)
    else:
        features.append(0.5)
    h2h = df_history[((df_history['主队'] == home) & (df_history['客队'] == away)) |
                     ((df_history['主队'] == away) & (df_history['客队'] == home))]
    if not h2h.empty:
        net_goals = []
        for _, m in h2h.iterrows():
            try:
                h, a = map(int, m['比分'].split(':'))
                if m['主队'] == home:
                    net_goals.append(h - a)
                else:
                    net_goals.append(a - h)
            except:
                continue
        avg_net = np.mean(net_goals) if net_goals else 0
        features.append(avg_net)
    else:
        features.append(0)
    if not h2h.empty:
        total_goals = []
        for _, m in h2h.iterrows():
            try:
                h, a = map(int, m['比分'].split(':'))
                total_goals.append(h + a)
            except:
                continue
        avg_total = np.mean(total_goals) if total_goals else 1.5
        features.append(avg_total)
    else:
        features.append(1.5)
    features.append(actual_hc if pd.notna(actual_hc) else 0)
    home_recent = df_history[df_history['主队'] == home].sort_values('日期', ascending=False).head(3)
    if len(home_recent) >= 2:
        lam_changes = (home_recent['λ_主队即'].iloc[0] - home_recent['λ_主队即'].iloc[-1]) if len(home_recent) > 1 else 0
        features.append(lam_changes)
    else:
        features.append(0)
    away_recent = df_history[df_history['客队'] == away].sort_values('日期', ascending=False).head(3)
    if len(away_recent) >= 2:
        lam_changes = (away_recent['λ_客队即'].iloc[0] - away_recent['λ_客队即'].iloc[-1]) if len(away_recent) > 1 else 0
        features.append(lam_changes)
    else:
        features.append(0)
    return np.array(features).reshape(1, -1)

def train_models():
    df = load_results()
    df = df.dropna(subset=['比分', '主队', '客队', '实际盘口'])
    if df.empty or len(df) < 50:
        st.warning("历史数据不足50场，无法训练可靠的机器学习模型。请积累更多数据。")
        return False
    df['主队进球'] = df['比分'].str.split(':').str[0].astype(int)
    df['客队进球'] = df['比分'].str.split(':').str[1].astype(int)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    X = []
    y_asian = []  # 0=输盘, 1=走水, 2=赢盘
    y_ou = []
    valid_indices = []
    for idx, row in df.iterrows():
        try:
            h, a = row['主队进球'], row['客队进球']
            net = h - a
            hc = row['实际盘口']
            if pd.isna(hc):
                continue
            if hc > 0:
                if net > hc:
                    asian_label = 2
                elif net == hc:
                    asian_label = 1
                else:
                    asian_label = 0
            elif hc < 0:
                if net > hc:
                    asian_label = 2
                elif net == hc:
                    asian_label = 1
                else:
                    asian_label = 0
            else:
                if net > 0:
                    asian_label = 2
                elif net == 0:
                    asian_label = 1
                else:
                    asian_label = 0
            total_goals = h + a
            ou_label = 1 if total_goals > 2.5 else 0
            hist_df = df.drop(idx)
            features = build_features_for_match(row, hist_df)
            X.append(features.flatten())
            y_asian.append(asian_label)
            y_ou.append(ou_label)
            valid_indices.append(idx)
        except Exception as e:
            continue
    if len(X) < 30:
        st.warning(f"有效样本仅{len(X)}，不足30，无法训练模型。")
        return False
    X = np.array(X)
    y_asian = np.array(y_asian)
    y_ou = np.array(y_ou)
    scaler_asian = StandardScaler()
    X_scaled = scaler_asian.fit_transform(X)
    scaler_ou = StandardScaler()
    X_scaled_ou = scaler_ou.fit_transform(X)
    
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [30, 50, 70],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10]
    }
    rf_asian = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_asian = GridSearchCV(rf_asian, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid_asian.fit(X_scaled, y_asian)
    best_rf_asian = grid_asian.best_estimator_
    rf_ou = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_ou = GridSearchCV(rf_ou, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid_ou.fit(X_scaled_ou, y_ou)
    best_rf_ou = grid_ou.best_estimator_
    joblib.dump(best_rf_asian, MODEL_ASIAN)
    joblib.dump(best_rf_ou, MODEL_OU)
    save_scaler(scaler_asian, SCALER_ASIAN)
    save_scaler(scaler_ou, SCALER_OU)
    
    n_test = max(10, int(len(X)*0.2))
    n_test = min(n_test, len(X))
    X_test = X_scaled[-n_test:]
    y_asian_test = y_asian[-n_test:]
    y_ou_test = y_ou[-n_test:]
    if len(y_asian_test) > 0:
        acc_asian = accuracy_score(y_asian_test, best_rf_asian.predict(X_test))
        acc_ou = accuracy_score(y_ou_test, best_rf_ou.predict(X_test))
    else:
        acc_asian = 0.0
        acc_ou = 0.0
    if np.isnan(acc_asian):
        acc_asian = 0.0
    if np.isnan(acc_ou):
        acc_ou = 0.0
    
    st.success(f"模型训练完成！最佳参数: {grid_asian.best_params_}")
    st.success(f"亚洲盘三分类时序验证准确率（交叉验证）: {grid_asian.best_score_:.2%}")
    st.success(f"亚洲盘三分类最近{n_test}场验证准确率: {acc_asian:.2%}")
    st.success(f"大小球最近{n_test}场验证准确率: {acc_ou:.2%}")
    return True

def load_models():
    if os.path.exists(MODEL_ASIAN) and os.path.exists(MODEL_OU):
        try:
            rf_asian = joblib.load(MODEL_ASIAN)
            rf_ou = joblib.load(MODEL_OU)
            return rf_asian, rf_ou
        except:
            return None, None
    return None, None

def predict_with_ml(row, df_history):
    rf_asian, rf_ou = load_models()
    scaler_asian = load_scaler(SCALER_ASIAN)
    scaler_ou = load_scaler(SCALER_OU)
    if rf_asian is None or rf_ou is None or scaler_asian is None or scaler_ou is None:
        return None, None
    features = build_features_for_match(row, df_history)
    features_scaled_asian = scaler_asian.transform(features)
    features_scaled_ou = scaler_ou.transform(features)
    prob_asian = rf_asian.predict_proba(features_scaled_asian)[0]
    prob_ou = rf_ou.predict_proba(features_scaled_ou)[0][1]
    return prob_asian, prob_ou

# ========== 综合推荐函数 ==========
def get_head_to_head(team_h, team_a):
    df = load_results()
    if df.empty:
        return None
    df['主队_clean'] = df['主队'].astype(str).str.strip()
    df['客队_clean'] = df['客队'].astype(str).str.strip()
    team_h_clean = team_h.strip().lower()
    team_a_clean = team_a.strip().lower()
    mask = ((df['主队_clean'].str.lower() == team_h_clean) & (df['客队_clean'].str.lower() == team_a_clean)) | \
           ((df['主队_clean'].str.lower() == team_a_clean) & (df['客队_clean'].str.lower() == team_h_clean))
    df_h2h = df[mask].copy()
    if df_h2h.empty:
        return None
    df_h2h['日期'] = pd.to_datetime(df_h2h['日期'])
    df_h2h = df_h2h.sort_values('日期', ascending=False)
    net_goals = []
    total_goals = []
    win_push_lose = []
    for _, row in df_h2h.iterrows():
        try:
            h_score, a_score = map(int, row['比分'].split(':'))
        except:
            continue
        if row['主队_clean'].lower() == team_h_clean:
            net = h_score - a_score
            total = h_score + a_score
            actual_hc = row['实际盘口']
            if pd.notna(actual_hc):
                if actual_hc > 0:
                    if net > actual_hc:
                        win_push_lose.append(1)
                    elif net == actual_hc:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
                elif actual_hc < 0:
                    if net > actual_hc:
                        win_push_lose.append(1)
                    elif net == actual_hc:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
                else:
                    if net > 0:
                        win_push_lose.append(1)
                    elif net == 0:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
        else:
            net = a_score - h_score
            total = h_score + a_score
            actual_hc = row['实际盘口']
            if pd.notna(actual_hc):
                hc_opp = -actual_hc
                if hc_opp > 0:
                    if net > hc_opp:
                        win_push_lose.append(1)
                    elif net == hc_opp:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
                elif hc_opp < 0:
                    if net > hc_opp:
                        win_push_lose.append(1)
                    elif net == hc_opp:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
                else:
                    if net > 0:
                        win_push_lose.append(1)
                    elif net == 0:
                        win_push_lose.append(0)
                    else:
                        win_push_lose.append(-1)
        net_goals.append(net)
        total_goals.append(total)
    if not net_goals:
        return None
    avg_net = np.mean(net_goals)
    avg_total = np.mean(total_goals)
    win_rate = sum(1 for w in win_push_lose if w == 1) / len(win_push_lose) if win_push_lose else 0.5
    return {
        'avg_net': avg_net,
        'avg_total': avg_total,
        'win_rate': win_rate,
        'matches': len(net_goals)
    }

def odds_change_recommendation(lam_h_init, lam_h_live, lam_a_init, lam_a_live):
    if lam_h_init is None or lam_h_live is None:
        return None, None
    h_change = lam_h_live - lam_h_init
    a_change = lam_a_live - lam_a_init
    total_init = lam_h_init + lam_a_init
    total_live = lam_h_live + lam_a_live
    total_change = total_live - total_init
    hc_change = (h_change - a_change) * 0.3
    ou_tend = total_change
    return hc_change, ou_tend

def asian_handicap_stat(similar_matches, target_hc):
    if not similar_matches or len(similar_matches) < 6:
        return 0, None, None
    cnt = 0
    win = 0
    for m in similar_matches:
        actual_hc = m.get('实际盘口')
        if pd.isna(actual_hc):
            continue
        if round(actual_hc, 2) == round(target_hc, 2):
            cnt += 1
            try:
                h, a = map(int, m['比分'].split(':'))
                net = h - a
                if actual_hc > 0:
                    if net > actual_hc:
                        win += 1
                elif actual_hc < 0:
                    if net < actual_hc:
                        win += 1
                else:
                    if net > 0:
                        win += 1
            except:
                continue
    if cnt == 0:
        return 0, None, None
    is_home_team = True if target_hc > 0 else (False if target_hc < 0 else True)
    return cnt, win / cnt, is_home_team

def comprehensive_recommendation(team_h, team_a, lam_h_init, lam_a_init, lam_h_live, lam_a_live, similar_matches,
                                 real_h_lam=None, real_a_lam=None, use_ml=False):
    # 1. 获取硬实力调整后λ
    if real_h_lam is None or real_a_lam is None:
        _, real_h_lam, _, _ = get_team_recent_strength(team_h, n_matches=5, apply_result_adjust=True)
        _, real_a_lam, _, _ = get_team_recent_strength(team_a, n_matches=5, apply_result_adjust=True)
    if real_h_lam is None or real_a_lam is None:
        lam_h_strength = 1.0
        lam_a_strength = 1.0
    else:
        lam_h_strength = real_h_lam
        lam_a_strength = real_a_lam

    # 2. 市场λ（波胆隐含）
    lam_h_market = lam_h_live
    lam_a_market = lam_a_live

    # 3. 计算加权λ
    if real_h_lam is not None and real_a_lam is not None:
        strength_lam_h = real_h_lam
        strength_lam_a = real_a_lam
    else:
        strength_lam_h = 1.0
        strength_lam_a = 1.0
    change_lam_h = lam_h_live
    change_lam_a = lam_a_live
    h2h = get_head_to_head(team_h, team_a)
    if h2h:
        lam_h_h2h = max(0.1, (h2h['avg_total'] + h2h['avg_net']) / 2)
        lam_a_h2h = max(0.1, (h2h['avg_total'] - h2h['avg_net']) / 2)
    else:
        lam_h_h2h = 1.0
        lam_a_h2h = 1.0
    if similar_matches and len(similar_matches) >= 6:
        lam_h_list = []
        lam_a_list = []
        for m in similar_matches:
            if 'λ_主队即' in m and pd.notna(m['λ_主队即']):
                lam_h_list.append(m['λ_主队即'])
            if 'λ_客队即' in m and pd.notna(m['λ_客队即']):
                lam_a_list.append(m['λ_客队即'])
        lam_h_similar = np.mean(lam_h_list) if lam_h_list else 1.0
        lam_a_similar = np.mean(lam_a_list) if lam_a_list else 1.0
    else:
        lam_h_similar = 1.0
        lam_a_similar = 1.0
    weights = {'strength': 0.1, 'change': 0.3, 'h2h': 0.2, 'similar': 0.4}
    total_w = sum(weights.values())
    lam_h_combined = (strength_lam_h * weights['strength'] +
                      change_lam_h * weights['change'] +
                      lam_h_h2h * weights['h2h'] +
                      lam_h_similar * weights['similar']) / total_w
    lam_a_combined = (strength_lam_a * weights['strength'] +
                      change_lam_a * weights['change'] +
                      lam_a_h2h * weights['h2h'] +
                      lam_a_similar * weights['similar']) / total_w

    # 基于λ差值查找历史相似比赛并统计
    df_history = load_results()
    lambda_stats = None
    score_recommendations = []
    if not df_history.empty:
        hist_lam_diff = df_history['λ_主队即'] - df_history['λ_客队即']
        curr_lam_diff = lam_h_combined - lam_a_combined
        diff_abs = (hist_lam_diff - curr_lam_diff).abs()
        top_n = min(30, len(df_history))
        if top_n > 0:
            idx_sorted = diff_abs.argsort()[:top_n]
            similar_lambda = df_history.iloc[idx_sorted]
            similar_lambda = similar_lambda[similar_lambda['比分'].str.match(r'\d+:\d+')]
            if len(similar_lambda) >= 5:
                total = len(similar_lambda)
                home_wins = 0
                draws = 0
                away_wins = 0
                home_win_handicap = 0
                over_25 = 0
                home_win_scores = []
                draw_scores = []
                away_win_scores = []
                for _, row in similar_lambda.iterrows():
                    try:
                        h, a = map(int, row['比分'].split(':'))
                        score = f"{h}:{a}"
                        if h > a:
                            home_wins += 1
                            home_win_scores.append(score)
                        elif h == a:
                            draws += 1
                            draw_scores.append(score)
                        else:
                            away_wins += 1
                            away_win_scores.append(score)
                        if h + a > 2.5:
                            over_25 += 1
                        actual_hc = row.get('实际盘口')
                        if pd.notna(actual_hc):
                            net = h - a
                            if actual_hc > 0:
                                if net > actual_hc:
                                    home_win_handicap += 1
                            elif actual_hc < 0:
                                if net < actual_hc:
                                    home_win_handicap += 1
                            else:
                                if net > 0:
                                    home_win_handicap += 1
                    except:
                        continue
                lambda_stats = {
                    'total': total,
                    'home_win_rate': home_wins / total,
                    'draw_rate': draws / total,
                    'away_win_rate': away_wins / total,
                    'home_win_handicap_rate': home_win_handicap / total,
                    'over_25_rate': over_25 / total
                }
                if home_win_scores:
                    most_common_home_score = Counter(home_win_scores).most_common(1)[0][0]
                else:
                    most_common_home_score = None
                if draw_scores:
                    most_common_draw_score = Counter(draw_scores).most_common(1)[0][0]
                else:
                    most_common_draw_score = None
                if away_win_scores:
                    most_common_away_score = Counter(away_win_scores).most_common(1)[0][0]
                else:
                    most_common_away_score = None

                def probs_and_odds(lam_h, lam_a, juice=0.9):
                    p_h = 0.0
                    p_d = 0.0
                    p_a = 0.0
                    for x in range(5):
                        for y in range(5):
                            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                            if x > y:
                                p_h += prob
                            elif x == y:
                                p_d += prob
                            else:
                                p_a += prob
                    total = p_h + p_d + p_a
                    if total > 0:
                        p_h /= total
                        p_d /= total
                        p_a /= total
                    odds_h = juice / p_h if p_h > 0 else 0
                    odds_d = juice / p_d if p_d > 0 else 0
                    odds_a = juice / p_a if p_a > 0 else 0
                    return (odds_h, odds_d, odds_a)

                combined_odds_compare = probs_and_odds(lam_h_combined, lam_a_combined, juice=0.9)
                market_odds_compare = probs_and_odds(lam_h_market, lam_a_market, juice=0.9)

                if combined_odds_compare[0] > market_odds_compare[0] and most_common_home_score:
                    score_recommendations.append(f"主胜 {most_common_home_score}")
                if combined_odds_compare[1] > market_odds_compare[1] and most_common_draw_score:
                    score_recommendations.append(f"平局 {most_common_draw_score}")
                if combined_odds_compare[2] > market_odds_compare[2] and most_common_away_score:
                    score_recommendations.append(f"客胜 {most_common_away_score}")

    # 综合推荐赔率
    if lambda_stats and lambda_stats['total'] >= 5:
        p_h = lambda_stats['home_win_rate']
        p_d = lambda_stats['draw_rate']
        p_a = lambda_stats['away_win_rate']
        juice = 0.9
        odds_h = juice / p_h if p_h > 0 else 0
        odds_d = juice / p_d if p_d > 0 else 0
        odds_a = juice / p_a if p_a > 0 else 0
        combined_odds = (odds_h, odds_d, odds_a)
        combined_note = "（基于历史相似比赛统计）"
    else:
        def probs_and_odds(lam_h, lam_a, juice=0.9):
            p_h = 0.0
            p_d = 0.0
            p_a = 0.0
            for x in range(5):
                for y in range(5):
                    prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                    if x > y:
                        p_h += prob
                    elif x == y:
                        p_d += prob
                    else:
                        p_a += prob
            total = p_h + p_d + p_a
            if total > 0:
                p_h /= total
                p_d /= total
                p_a /= total
            odds_h = juice / p_h if p_h > 0 else 0
            odds_d = juice / p_d if p_d > 0 else 0
            odds_a = juice / p_a if p_a > 0 else 0
            return (p_h, p_d, p_a), (odds_h, odds_d, odds_a)
        _, combined_odds = probs_and_odds(lam_h_combined, lam_a_combined, juice=0.9)
        combined_note = "（加权λ模型）"

    # 硬实力赔率和市场赔率
    def probs_and_odds_fixed(lam_h, lam_a, juice=0.9):
        p_h = 0.0
        p_d = 0.0
        p_a = 0.0
        for x in range(5):
            for y in range(5):
                prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                if x > y:
                    p_h += prob
                elif x == y:
                    p_d += prob
                else:
                    p_a += prob
        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total
            p_d /= total
            p_a /= total
        odds_h = juice / p_h if p_h > 0 else 0
        odds_d = juice / p_d if p_d > 0 else 0
        odds_a = juice / p_a if p_a > 0 else 0
        return (p_h, p_d, p_a), (odds_h, odds_d, odds_a)

    _, strength_odds = probs_and_odds_fixed(lam_h_strength, lam_a_strength, juice=0.9)
    _, market_odds = probs_and_odds_fixed(lam_h_market, lam_a_market, juice=0.9)

    # 亚盘推荐
    diff_lam = lam_h_combined - lam_a_combined
    asian_hc = handicap_from_diff(diff_lam)
    cnt, win_rate, is_home_team = asian_handicap_stat(similar_matches, asian_hc)
    
    if cnt >= 5:
        if win_rate < 0.4:
            asian_hc = -asian_hc
            reversed_flag = True
        else:
            reversed_flag = False
    else:
        reversed_flag = False
    
    if asian_hc > 0:
        hc_abs = asian_hc
        hc_desc = f"主队 -{hc_abs:.2f}"
        if cnt >= 5:
            if not reversed_flag:
                stat_team = "主队" if is_home_team else "客队"
                percent_text = f"，相似比赛{cnt}场，{stat_team}赢盘率 {win_rate:.0%}"
                if win_rate < 0.4:
                    percent_text += "（偏低，建议关注受让方）"
                elif win_rate > 0.6:
                    percent_text += "（较高）"
            else:
                percent_text = f"，相似比赛{cnt}场，原让球方赢盘率仅 {win_rate:.0%}，因此建议关注受让方"
        elif cnt > 0:
            percent_text = f"，相似比赛仅{cnt}场，数据不足"
        else:
            percent_text = "，无相似比赛数据"
        if not reversed_flag:
            asian_rec = f"推荐主队让 {hc_abs:.2f}（{hc_desc}{percent_text}）"
        else:
            asian_rec = f"推荐主队受让 {hc_abs:.2f}（主队 +{hc_abs:.2f}{percent_text}）"
    elif asian_hc < 0:
        hc_abs = -asian_hc
        hc_desc = f"客队 -{hc_abs:.2f}"
        if cnt >= 5:
            if not reversed_flag:
                stat_team = "主队" if is_home_team else "客队"
                percent_text = f"，相似比赛{cnt}场，{stat_team}赢盘率 {win_rate:.0%}"
                if win_rate < 0.4:
                    percent_text += "（偏低，建议关注受让方）"
                elif win_rate > 0.6:
                    percent_text += "（较高）"
            else:
                percent_text = f"，相似比赛{cnt}场，原让球方赢盘率仅 {win_rate:.0%}，因此建议关注受让方"
        elif cnt > 0:
            percent_text = f"，相似比赛仅{cnt}场，数据不足"
        else:
            percent_text = "，无相似比赛数据"
        if not reversed_flag:
            asian_rec = f"推荐客队让 {hc_abs:.2f}（{hc_desc}{percent_text}）"
        else:
            asian_rec = f"推荐客队受让 {hc_abs:.2f}（客队 +{hc_abs:.2f}{percent_text}）"
    else:
        if cnt >= 5:
            percent_text = f"，相似比赛{cnt}场，主队赢盘率 {win_rate:.0%}"
            if win_rate > 0.6:
                percent_text += "（主队机会较大）"
            elif win_rate < 0.4:
                percent_text += "（客队机会较大）"
        elif cnt > 0:
            percent_text = f"，相似比赛仅{cnt}场，数据不足"
        else:
            percent_text = "，无相似比赛数据"
        asian_rec = f"推荐平手盘（主队 0{percent_text}）"

    # 大小球推荐
    ml_ou_prob = None
    if use_ml:
        current_row = pd.Series({
            '主队': team_h,
            '客队': team_a,
            'λ_主队即': lam_h_live,
            'λ_客队即': lam_a_live,
            '实际盘口': handicap_from_diff(lam_h_live - lam_a_live)
        })
        hist_df = load_results()
        if not hist_df.empty:
            _, ml_ou_prob = predict_with_ml(current_row, hist_df)
            if ml_ou_prob is None:
                use_ml = False
    if use_ml and ml_ou_prob is not None:
        if ml_ou_prob >= 0.6:
            final_ou = "大2.5"
            ou_conf = ml_ou_prob
        elif ml_ou_prob <= 0.4:
            final_ou = "小2.5"
            ou_conf = 1 - ml_ou_prob
        else:
            final_ou = "不明确"
            ou_conf = 0
    else:
        if real_h_lam is not None and real_a_lam is not None:
            strength_ou = real_h_lam + real_a_lam
            strength_score = 1 if strength_ou > 2.8 else (-1 if strength_ou < 2.2 else 0)
        else:
            strength_score = 0
        _, ou_change = odds_change_recommendation(lam_h_init, lam_h_live, lam_a_init, lam_a_live)
        if ou_change is not None:
            ou_tend = "大" if ou_change > 0.1 else ("小" if ou_change < -0.1 else "中性")
            change_score = 1 if ou_tend == "大" else (-1 if ou_tend == "小" else 0)
        else:
            change_score = 0
        h2h = get_head_to_head(team_h, team_a)
        if h2h:
            h2h_ou = h2h['avg_total']
            h2h_score = 1 if h2h_ou > 2.8 else (-1 if h2h_ou < 2.2 else 0)
        else:
            h2h_score = 0
        if similar_matches and len(similar_matches) >= 6:
            goals = []
            for m in similar_matches:
                try:
                    h, a = map(int, m['比分'].split(':'))
                    goals.append(h + a)
                except:
                    continue
            if goals:
                over_ratio = sum(1 for g in goals if g >= 3) / len(goals)
                similar_ou = "大" if over_ratio >= 0.6 else ("小" if over_ratio <= 0.4 else "中性")
                similar_score = 1 if similar_ou == "大" else (-1 if similar_ou == "小" else 0)
            else:
                similar_score = 0
        else:
            similar_score = 0
        w = {'strength': 0.1, 'change': 0.3, 'h2h': 0.2, 'similar': 0.4}
        total_score = (strength_score * w['strength'] +
                       change_score * w['change'] +
                       h2h_score * w['h2h'] +
                       similar_score * w['similar'])
        if total_score > 0.2:
            final_ou = "大2.5"
            ou_conf = total_score
        elif total_score < -0.2:
            final_ou = "小2.5"
            ou_conf = -total_score
        else:
            final_ou = "不明确"
            ou_conf = 0

    # 比分预测（基于加权λ）
    max_prob = 0
    best_score = "0:0"
    for x in range(5):
        for y in range(5):
            prob = poisson.pmf(x, lam_h_combined) * poisson.pmf(y, lam_a_combined)
            if prob > max_prob:
                max_prob = prob
                best_score = f"{x}:{y}"
    pred_score = best_score
    pred_note = "（加权λ）"

    # ========== 确保 score_recommendations 非空 ==========
    if not score_recommendations:
        def probs_and_odds_local(lam_h, lam_a, juice=0.9):
            p_h = p_d = p_a = 0.0
            for x in range(5):
                for y in range(5):
                    prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                    if x > y:
                        p_h += prob
                    elif x == y:
                        p_d += prob
                    else:
                        p_a += prob
            total = p_h + p_d + p_a
            if total > 0:
                p_h /= total
                p_d /= total
                p_a /= total
            odds_h = juice / p_h if p_h > 0 else 0
            odds_d = juice / p_d if p_d > 0 else 0
            odds_a = juice / p_a if p_a > 0 else 0
            return (odds_h, odds_d, odds_a)
        
        market_odds_local = probs_and_odds_local(lam_h_market, lam_a_market, juice=0.9)
        combined_odds_local = probs_and_odds_local(lam_h_combined, lam_a_combined, juice=0.9)
        
        try:
            h, a = map(int, best_score.split(':'))
            if h > a and combined_odds_local[0] > market_odds_local[0]:
                score_recommendations.append(f"主胜 {best_score}（综合模型赔率优于市场）")
            elif h == a and combined_odds_local[1] > market_odds_local[1]:
                score_recommendations.append(f"平局 {best_score}（综合模型赔率优于市场）")
            elif h < a and combined_odds_local[2] > market_odds_local[2]:
                score_recommendations.append(f"客胜 {best_score}（综合模型赔率优于市场）")
            else:
                score_recommendations.append(f"模型最可能比分 {best_score}，但市场赔率更优，建议结合临场信息")
        except:
            score_recommendations.append("无法解析最可能比分，请参考其他指标")
    
    return (combined_odds, strength_odds, market_odds, asian_rec, final_ou, ou_conf, pred_score, pred_note, lambda_stats, combined_note, score_recommendations)

# ========== Football-Data.org API 集成函数 ==========
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_matches_by_matchday(competition_code, season, matchday, max_retries=3):
    """获取指定联赛、赛季、轮次的比赛列表（带重试机制）"""
    headers = {
        'X-Auth-Token': FOOTBALL_DATA_API_KEY,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"{FOOTBALL_DATA_BASE_URL}competitions/{competition_code}/matches"
    params = {'season': season, 'matchday': matchday}

    # 使用带重试机制的 Session
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            matches = data.get('matches', [])
            matches_list = []
            for m in matches:
                home = m['homeTeam']['name']
                away = m['awayTeam']['name']
                status = m['status']
                home_score = m['score']['fullTime']['home']
                away_score = m['score']['fullTime']['away']
                if status == 'FINISHED' and home_score is not None and away_score is not None:
                    result = f"{home_score}:{away_score}"
                else:
                    result = "未开始" if status == 'SCHEDULED' else "进行中"
                matches_list.append({
                    '比赛日期': m['utcDate'],
                    '主队': home,
                    '客队': away,
                    '状态': status,
                    '比分': result,
                    '主队进球': home_score if status == 'FINISHED' else None,
                    '客队进球': away_score if status == 'FINISHED' else None
                })
            return pd.DataFrame(matches_list)
        except requests.exceptions.RequestException as e:
            st.warning(f"第 {attempt+1} 次请求失败: {e}")
            if attempt == max_retries - 1:
                st.error("多次重试后仍然失败，请检查网络或 API 状态。")
                return pd.DataFrame()
            time.sleep(1)  # 等待 1 秒后重试
    return pd.DataFrame()
def simple_predict_by_team_strength(team_h, team_a):
    """基于球队历史实力 λ 进行简单预测（无需波胆赔率）"""
    _, lam_h, _, _ = get_team_recent_strength(team_h, n_matches=5, apply_result_adjust=True)
    _, lam_a, _, _ = get_team_recent_strength(team_a, n_matches=5, apply_result_adjust=True)
    if lam_h is None or lam_a is None:
        return None, None, None, None
    # 计算胜平负概率
    p_h = p_d = p_a = 0.0
    for x in range(5):
        for y in range(5):
            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
            if x > y:
                p_h += prob
            elif x == y:
                p_d += prob
            else:
                p_a += prob
    total = p_h + p_d + p_a
    if total > 0:
        p_h /= total
        p_d /= total
        p_a /= total
    # 大小球概率
    total_goals_exp = lam_h + lam_a
    ou = "大2.5" if total_goals_exp > 2.8 else ("小2.5" if total_goals_exp < 2.2 else "不明确")
    # 最可能比分
    max_prob = 0
    most_likely = "0:0"
    for x in range(5):
        for y in range(5):
            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
            if prob > max_prob:
                max_prob = prob
                most_likely = f"{x}:{y}"
    return (p_h, p_d, p_a), ou, most_likely, (lam_h, lam_a)

# ========== Streamlit 主界面 ==========
st.set_page_config(layout="wide")
st.title("⚽ 足球预测 + 泊松分析 + 盘口推导 + 历史预警")

if 'parsed_fingerprints' not in st.session_state:
    st.session_state.parsed_fingerprints = []
if 'parsed_results' not in st.session_state:
    st.session_state.parsed_results = {}
if 'force_overwrite' not in st.session_state:
    st.session_state.force_overwrite = False

# 创建 4 个 Tab
tab1, tab2, tab3, tab4 = st.tabs(["📈 赔率分析", "📝 录入比赛", "📊 历史记录（只读）", "🆕 自动获取比赛"])

# ========== Tab1 赔率分析（原有完整功能） ==========
with tab1:
    st.markdown("### 波胆赔率分析")
    st.info("粘贴波胆赔率数据（制表符或空格分隔），至少包含公司、分类（初/即）以及各波胆比分列。")
    auto_parse = st.checkbox("自动解析（粘贴后自动执行）", value=True)
    pasted = st.text_area("粘贴数据", height=200, key="pasted_area")
    if auto_parse and pasted:
        if 'last_pasted' not in st.session_state or st.session_state.last_pasted != pasted:
            st.session_state.last_pasted = pasted
            df_source, err = parse_pasted_text(pasted)
            if err:
                st.error(err)
            else:
                fp = compute_odds_fingerprint(df_source)
                if fp is not None and fp in st.session_state.parsed_fingerprints:
                    st.warning("⚠️ 该组赔率数据已解析过，自动解析已跳过。如需重新解析，请使用「手动解析」并勾选强制覆盖。")
                else:
                    st.session_state.df_source = df_source
                    if fp:
                        st.session_state.parsed_fingerprints.append(fp)
                    st.success("自动解析成功！")
                    st.rerun()
    if st.button("🔍 手动解析"):
        if pasted:
            df_source, err = parse_pasted_text(pasted)
            if err:
                st.error(err)
            else:
                fp = compute_odds_fingerprint(df_source)
                if fp is not None and fp in st.session_state.parsed_fingerprints:
                    st.warning("⚠️ 该组赔率数据已经解析过，是否强制覆盖？")
                    hist = st.session_state.parsed_results.get(fp)
                    if hist:
                        st.markdown("**📜 上次解析结果预览**：")
                        st.info(f"当时预测的平均λ：主 {hist['lam_h_live']:.3f} | 客 {hist['lam_a_live']:.3f}，最可能比分 {hist['most_common']}")
                        if hist.get('compare'):
                            st.dataframe(pd.DataFrame(hist['compare']), use_container_width=True)
                    df_history = load_results()
                    if 'odds_fingerprint' in df_history.columns:
                        matched = df_history[df_history['odds_fingerprint'] == fp]
                        if not matched.empty:
                            st.markdown("**📋 与该赔率对应的历史比赛结果**：")
                            matched_sorted = matched.sort_values('日期', ascending=False)
                            for _, row in matched_sorted.iterrows():
                                st.write(f"📅 {row['日期']} : {row['主队']} {row['比分']} {row['客队']}  (先进球: {row['先进球方']})")
                        else:
                            st.info("未找到该赔率对应的已录入比赛结果（可能尚未录入）")
                    if st.button("✅ 强制覆盖", key="force_parse_btn"):
                        st.session_state.df_source = df_source
                        st.session_state.force_overwrite = True
                        st.rerun()
                else:
                    st.session_state.df_source = df_source
                    if fp:
                        st.session_state.parsed_fingerprints.append(fp)
                    st.success("解析成功！")
                    st.rerun()
        else:
            st.warning("请粘贴数据")
    if st.session_state.get('force_overwrite', False):
        st.session_state.force_overwrite = False
        if 'df_source' in st.session_state:
            st.rerun()
    if 'df_source' in st.session_state and st.session_state.df_source is not None:
        df_source = st.session_state.df_source
        current_fp = compute_odds_fingerprint(df_source)
        if current_fp:
            st.session_state.current_fingerprint = current_fp
        res = process_dataframe(df_source)
        if res[2]:
            st.error(res[2])
        else:
            comp, goal, _, most_common, avg_over, avg_under, lam_tup, lam_h_final, lam_a_final = res
            lam_h_init, lam_a_init, lam_h_live, lam_a_live = lam_tup
            st.session_state['lam_h_init'] = lam_h_init
            st.session_state['lam_a_init'] = lam_a_init
            st.session_state['lam_h_live'] = lam_h_live
            st.session_state['lam_a_live'] = lam_a_live
            st.session_state['lam_h_final'] = lam_h_final
            st.session_state['lam_a_final'] = lam_a_final
            st.session_state['most_common_score'] = most_common
            avg_home_prob = np.mean([g['主胜概率'] for g in goal]) if goal else 0
            avg_home_first = np.mean([g['主先进球概率'] for g in goal]) if goal else 0
            st.session_state['avg_home_prob'] = avg_home_prob
            st.session_state['avg_home_first'] = avg_home_first
            fp = compute_odds_fingerprint(df_source)
            if fp:
                st.session_state.parsed_results[fp] = {
                    'lam_h_live': lam_h_live,
                    'lam_a_live': lam_a_live,
                    'most_common': most_common,
                    'compare': comp,
                    'goal_analysis': goal,
                    'lam_h_init': lam_h_init,
                    'lam_a_init': lam_a_init,
                    'lam_h_final': lam_h_final,
                    'lam_a_final': lam_a_final,
                    'avg_home_prob': avg_home_prob,
                    'avg_home_first': avg_home_first
                }
            similar_matches = get_similar_matches(lam_h_live, lam_a_live, avg_home_prob, avg_home_first, "主队", top_n=20,
                                                  lam_h_init=lam_h_init, lam_a_init=lam_a_init)
            st.session_state['similar_for_rec'] = similar_matches
            
            st.markdown("## 📊 球队近期真实实力 & 综合推荐")
            st.markdown("**说明**：以下推荐基于当前解析的本场比赛波胆赔率。")
            st.info(f"**当前解析的本场比赛市场λ**：主队 {lam_h_live:.3f} | 客队 {lam_a_live:.3f}")
            
            col_team1, col_team2 = st.columns(2)
            with col_team1:
                team_h = st.text_input("主队名称", key="strength_h")
            with col_team2:
                team_a = st.text_input("客队名称", key="strength_a")
            
            use_ml = st.checkbox("使用机器学习模型（随机森林）", value=False, key="use_ml")
            if use_ml:
                st.caption("机器学习模型需要至少50场有效历史比赛才能训练，如未训练请先点击下方按钮训练。")
            st.markdown("---")
            if st.button("🎯 生成综合推荐（基于以上队名）", key="comp_rec_inside"):
                if team_h and team_a:
                    with st.spinner("分析中..."):
                        avg_raw_h, avg_adj_h, details_h, cnt_h = get_team_recent_strength(team_h, n_matches=5, apply_result_adjust=True)
                        avg_raw_a, avg_adj_a, details_a, cnt_a = get_team_recent_strength(team_a, n_matches=5, apply_result_adjust=True)
                        st.session_state['strength_h_raw'] = avg_raw_h
                        st.session_state['strength_h_adj'] = avg_adj_h
                        st.session_state['strength_h_details'] = details_h
                        st.session_state['strength_a_raw'] = avg_raw_a
                        st.session_state['strength_a_adj'] = avg_adj_a
                        st.session_state['strength_a_details'] = details_a
                        (combined_odds, strength_odds, market_odds, asian_rec, final_ou, ou_conf, pred_score, pred_note, lambda_stats, combined_note, score_recs) = comprehensive_recommendation(
                            team_h, team_a,
                            lam_h_init, lam_a_init,
                            lam_h_live, lam_a_live,
                            similar_matches,
                            real_h_lam=avg_adj_h,
                            real_a_lam=avg_adj_a,
                            use_ml=use_ml
                        )
                        st.markdown("### 📌 综合推荐结果")
                        
                        st.markdown("#### 🎲 亚洲盘推荐")
                        st.write(asian_rec)
                        
                        st.markdown("#### 📊 胜平负赔率（返还率90%）")
                        col_combined, col_strength, col_market = st.columns(3)
                        with col_combined:
                            st.markdown("**综合推荐赔率**")
                            st.write(f"主胜: {combined_odds[0]:.2f}")
                            st.write(f"平局: {combined_odds[1]:.2f}")
                            st.write(f"客胜: {combined_odds[2]:.2f}")
                            st.caption(combined_note)
                        with col_strength:
                            st.markdown("**硬实力赔率**（近5场调整后λ）")
                            st.write(f"主胜: {strength_odds[0]:.2f}")
                            st.write(f"平局: {strength_odds[1]:.2f}")
                            st.write(f"客胜: {strength_odds[2]:.2f}")
                        with col_market:
                            st.markdown("**市场赔率**（波胆隐含λ）")
                            st.write(f"主胜: {market_odds[0]:.2f}")
                            st.write(f"平局: {market_odds[1]:.2f}")
                            st.write(f"客胜: {market_odds[2]:.2f}")
                        
                        st.markdown("#### ⚽ 大小球与比分")
                        st.write(f"大小球推荐: {final_ou} (置信度 {ou_conf:.0%})")
                        st.write(f"预测比分: {pred_score} {pred_note}")
                        
                        if lambda_stats and lambda_stats['total'] >= 5:
                            st.markdown("#### 📈 基于λ值的历史相似比赛共性（综合λ差值）")
                            st.write(f"共匹配 {lambda_stats['total']} 场历史比赛：")
                            st.write(f"**主胜概率**: {lambda_stats['home_win_rate']:.1%} | **平局概率**: {lambda_stats['draw_rate']:.1%} | **客胜概率**: {lambda_stats['away_win_rate']:.1%}")
                            st.write(f"**主队赢盘率**: {lambda_stats['home_win_handicap_rate']:.1%} | **大2.5球概率**: {lambda_stats['over_25_rate']:.1%}")
                        else:
                            st.info("基于λ值的相似历史比赛不足5场，无法提供统计共性。")
                        
                        st.markdown("#### 🎯 综合推荐 vs 市场赔率对比推荐的比分")
                        if score_recs:
                            for rec in score_recs:
                                st.write(f"- {rec}")
                        else:
                            st.info("暂无基于赔率比较的比分推荐，请参考上方的预测比分。")
                        
                        if avg_adj_h is not None and avg_adj_a is not None:
                            st.markdown("**📊 硬实力摘要**")
                            st.write(f"主队 {team_h} 修正后 λ: {avg_adj_h:.3f} | 客队 {team_a} 修正后 λ: {avg_adj_a:.3f}")
                        else:
                            st.info("主队或客队历史数据不足（少于5场有效比赛），无法计算硬实力。")
                        st.caption("注：综合推荐仅供参考，请结合实时市场变化谨慎决策。")
                        if use_ml:
                            st.caption("推荐基于机器学习模型（随机森林）生成。")
                        else:
                            st.caption("推荐权重：相似历史比赛40% | 赔率变化30% | 历史交锋20% | 球队硬实力10%")
                else:
                    st.warning("请填写主队和客队名称")
            
            if st.session_state.get('strength_h_adj') is not None:
                st.markdown("---")
                st.markdown("**📊 已查询的硬实力结果**")
                col_h, col_a = st.columns(2)
                with col_h:
                    st.write(f"**主队 {team_h}**")
                    val_h = st.session_state.get('strength_h_adj')
                    if val_h is not None:
                        st.write(f"修正后 λ: {val_h:.3f}")
                    else:
                        st.write("修正后 λ: 暂无数据")
                    details_h = st.session_state.get('strength_h_details', [])
                    if details_h:
                        with st.expander(f"查看最近 {len(details_h)} 场详情"):
                            st.dataframe(pd.DataFrame(details_h), use_container_width=True)
                with col_a:
                    st.write(f"**客队 {team_a}**")
                    val_a = st.session_state.get('strength_a_adj')
                    if val_a is not None:
                        st.write(f"修正后 λ: {val_a:.3f}")
                    else:
                        st.write("修正后 λ: 暂无数据")
                    details_a = st.session_state.get('strength_a_details', [])
                    if details_a:
                        with st.expander(f"查看最近 {len(details_a)} 场详情"):
                            st.dataframe(pd.DataFrame(details_a), use_container_width=True)
            else:
                st.info("点击上方按钮生成综合推荐后，此处将显示球队硬实力详情。")
            
            st.markdown("---")
            st.markdown("#### 🆚 硬实力 vs 本场市场波胆对比")
            cur_lam_h = st.session_state.get('lam_h_live', None)
            cur_lam_a = st.session_state.get('lam_a_live', None)
            real_h_lam = st.session_state.get('strength_h_adj', None)
            real_a_lam = st.session_state.get('strength_a_adj', None)
            if cur_lam_h is None or cur_lam_a is None:
                st.info("请先在「波胆赔率分析」中解析本场比赛。")
            elif real_h_lam is None or real_a_lam is None:
                st.info("请先点击上方「生成综合推荐」按钮获取球队硬实力。")
            else:
                home_strength_score = (real_h_lam - real_a_lam) * 0.8 + 0.2
                away_strength_score = (real_a_lam - real_h_lam) * 0.8 + 0.1
                market_score = cur_lam_h - cur_lam_a
                
                st.subheader("📊 硬实力得分 vs 市场预期得分")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("主队硬实力得分", f"{home_strength_score:+.3f}", 
                              delta=f"vs 市场 {market_score:+.3f}" if market_score is not None else None)
                    st.caption("计算方式：(主队近5场λ - 客队近5场λ) × 0.8 + 主场加成(0.2)")
                with col_s2:
                    st.metric("客队硬实力得分", f"{away_strength_score:+.3f}",
                              delta=f"vs 市场 {-market_score:+.3f}" if market_score is not None else None)
                    st.caption("计算方式：(客队近5场λ - 主队近5场λ) × 0.8 + 客场加成(0.1)")
                
                st.markdown("**💡 分析结论**")
                diff_strength = home_strength_score - away_strength_score
                diff_market = market_score
                if diff_strength > diff_market + 0.2:
                    st.warning(f"主队硬实力得分 ({diff_strength:+.2f}) 明显高于市场预期 ({diff_market:+.2f})，市场可能低估主队。")
                elif diff_strength < diff_market - 0.2:
                    st.info(f"主队硬实力得分 ({diff_strength:+.2f}) 明显低于市场预期 ({diff_market:+.2f})，市场可能高估主队。")
                else:
                    st.success(f"主队硬实力与市场预期基本一致（差值 {diff_strength - diff_market:+.2f})")
                
                st.caption(f"注：主队近5场平均λ = {real_h_lam:.3f}，客队近5场平均λ = {real_a_lam:.3f}；市场主队λ = {cur_lam_h:.3f}，市场客队λ = {cur_lam_a:.3f}")
            
            st.markdown("### 🤖 机器学习模型训练")
            if st.button("重新训练机器学习模型（基于历史数据）"):
                with st.spinner("训练中，请稍候..."):
                    success = train_models()
                    if success:
                        st.success("模型训练完成并已保存！")
                    else:
                        st.error("训练失败，请确保历史数据足够（至少50场有效比赛）。")
            
            st.success("解析完成")

# ========== Tab2 录入比赛（原有完整功能） ==========
with tab2:
    st.markdown("### 录入比赛结果")
    home_prob_avg = st.session_state.get('avg_home_prob', 0.35)
    home_first_avg = st.session_state.get('avg_home_first', 0.5)
    lam_h_init = st.session_state.get('lam_h_init', 0.0)
    lam_a_init = st.session_state.get('lam_a_init', 0.0)
    lam_h_live = st.session_state.get('lam_h_live', 0.0)
    lam_a_live = st.session_state.get('lam_a_live', 0.0)
    lam_h_final = st.session_state.get('lam_h_final', 0.0)
    lam_a_final = st.session_state.get('lam_a_final', 0.0)
    most_common = st.session_state.get('most_common_score', "2:1")
    if 'avg_home_prob' in st.session_state:
        st.info(f"赔率分析平均值：主胜概率={home_prob_avg:.4f}，主先进球概率={home_first_avg:.4f}，建议参考盘口={handicap_from_diff(lam_h_final - lam_a_final):+.2f}")
    else:
        st.info("请先在「赔率分析」中解析数据，此处将自动填充平均概率和λ值")
    col_rec1, col_rec2 = st.columns([1, 3])
    with col_rec1:
        if st.button("📚 从历史记录推荐"):
            if 'avg_home_prob' in st.session_state:
                first_scorer_sel = st.session_state.get('first_scorer_sel', '')
                similar_list = get_similar_matches(lam_h_live, lam_a_live, home_prob_avg, home_first_avg, first_scorer_sel, top_n=10,
                                                   lam_h_init=lam_h_init, lam_a_init=lam_a_init)
                if similar_list:
                    st.session_state['similar_list'] = similar_list
                    st.session_state['similar_index'] = 0
                    st.success(f"找到 {len(similar_list)} 条相似历史比赛（综合相似度≥0.85），将显示第一条，可点击前后按钮浏览")
                    st.rerun()
                else:
                    st.warning("未找到足够相似的历史比赛（综合相似度≥0.85）")
            else:
                st.warning("请先在「赔率分析」中解析数据")
    if 'similar_list' in st.session_state and st.session_state.similar_list:
        idx = st.session_state.get('similar_index', 0)
        total = len(st.session_state.similar_list)
        if 0 <= idx < total:
            rec = st.session_state.similar_list[idx]
            sim = rec['sim_score']
            star_text, _ = get_star_rating_from_similarity(sim)
            if star_text is None:
                star_text = "不推荐 (相似度不足)"
            st.markdown(f"**推荐 {idx+1}/{total} - {star_text} (综合相似度 {sim:.4f})**")
            st.write(f"日期：{rec['日期']}  {rec['主队']} vs {rec['客队']} 比分：{rec['比分']}")
            st.write(f"参考盘口：{rec['盘口']}  实际盘口：{rec['实际盘口']}")
            st.write(f"λ_主队初：{rec['λ_主队初']:.3f}  λ_主队即：{rec['λ_主队即']:.3f}  λ_客队初：{rec['λ_客队初']:.3f}  λ_客队即：{rec['λ_客队即']:.3f}")
            col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
            with col_btn1:
                if idx > 0:
                    if st.button("◀ 上一个"):
                        st.session_state.similar_index = idx - 1
                        st.rerun()
                else:
                    st.button("◀ 上一个", disabled=True)
            with col_btn2:
                if idx + 1 < total:
                    if st.button("下一个 ▶"):
                        st.session_state.similar_index = idx + 1
                        st.rerun()
                else:
                    st.button("下一个 ▶", disabled=True)
            with col_btn3:
                if st.button("📥 使用此推荐填充表单"):
                    st.session_state['rec_handicap'] = rec['盘口']
                    st.session_state['rec_actual_handicap'] = rec['实际盘口']
                    st.session_state['rec_lam_h_init'] = rec['λ_主队初']
                    st.session_state['rec_lam_h_live'] = rec['λ_主队即']
                    st.session_state['rec_lam_a_init'] = rec['λ_客队初']
                    st.session_state['rec_lam_a_live'] = rec['λ_客队即']
                    st.session_state['rec_predict'] = rec['预测结果']
                    st.session_state['rec_first_scorer'] = rec['先进球方']
                    st.success("已填充参考盘口、λ、预测结果和先进球方（概率保持赔率分析值）")
                    st.rerun()
    handicap_options = [-2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("日期", datetime.now())
        home_team = st.text_input("主队", key="home_input")
        home_score = st.number_input("主队进球", min_value=0, step=1, value=0)
        default_handicap = st.session_state.get('rec_handicap', handicap_from_diff(lam_h_final - lam_a_final))
        if default_handicap not in handicap_options:
            default_handicap = 0.0
        handicap_index = handicap_options.index(default_handicap) if default_handicap in handicap_options else handicap_options.index(0)
        handicap = st.selectbox("参考盘口", handicap_options, index=handicap_index)
        default_actual = st.session_state.get('rec_actual_handicap', 0)
        if default_actual not in handicap_options:
            default_actual = 0.0
        actual_index = handicap_options.index(default_actual) if default_actual in handicap_options else handicap_options.index(0)
        actual_handicap = st.selectbox("实际盘口（开赛时）", handicap_options, index=actual_index, key="actual_hc")
        default_lam_h_init = st.session_state.get('rec_lam_h_init', lam_h_init)
        lam_h_init = st.number_input("λ_主队初", value=float(default_lam_h_init), step=0.001, format="%.3f")
        default_lam_h_live = st.session_state.get('rec_lam_h_live', lam_h_live)
        lam_h_live = st.number_input("λ_主队即", value=float(default_lam_h_live), step=0.001, format="%.3f")
        home_prob = st.number_input("主胜概率", value=float(home_prob_avg), step=0.0001, format="%.4f")
        first_scorer_options = ["主队", "客队", "无进球"]
        default_first_scorer = st.session_state.get('rec_first_scorer', '')
        first_scorer_index = first_scorer_options.index(default_first_scorer) if default_first_scorer in first_scorer_options else 0
        first_scorer = st.radio("先进球方", first_scorer_options, index=first_scorer_index, key="first_scorer_sel")
    with col2:
        away_team = st.text_input("客队", key="away_input")
        away_score = st.number_input("客队进球", min_value=0, step=1, value=0)
        default_predict = st.session_state.get('rec_predict', most_common)
        predict_result = st.text_input("预测结果（APP推演）", value=default_predict)
        default_lam_a_init = st.session_state.get('rec_lam_a_init', lam_a_init)
        lam_a_init = st.number_input("λ_客队初", value=float(default_lam_a_init), step=0.001, format="%.3f")
        default_lam_a_live = st.session_state.get('rec_lam_a_live', lam_a_live)
        lam_a_live = st.number_input("λ_客队即", value=float(default_lam_a_live), step=0.001, format="%.3f")
        home_first = st.number_input("主先进球概率", value=float(home_first_avg), step=0.0001, format="%.4f")
    if st.button("💾 保存比赛", key="save_btn"):
        if not home_team or not away_team:
            st.error("请填写主队和客队名称")
        else:
            fingerprint = st.session_state.get('current_fingerprint', '')
            result = save_result(date.strftime("%Y-%m-%d"), home_team, away_team, home_score, away_score,
                        handicap, actual_handicap, predict_result,
                        lam_h_init, lam_h_live, lam_a_init, lam_a_live, home_prob, home_first, first_scorer,
                        odds_fingerprint=fingerprint)
            if result is not None:
                for key in ['rec_handicap', 'rec_actual_handicap', 'rec_lam_h_init', 'rec_lam_h_live', 'rec_lam_a_init', 'rec_lam_a_live', 'rec_predict', 'rec_first_scorer', 'similar_list', 'similar_index']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state['success_message'] = "✅ 比赛结果保存成功！"
                st.rerun()

# ========== Tab3 历史记录（原有完整功能） ==========
with tab3:
    st.markdown("### 历史记录（仅查看）")
    dfh = load_results()
    if dfh.empty:
        st.info("暂无记录")
    else:
        st.subheader("筛选条件")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_actual = st.selectbox("筛选实际盘口", ["全部"] + sorted(dfh['实际盘口'].dropna().unique().astype(str)), key="actual_filter")
        with col_f2:
            lam_h_min = st.number_input("主队λ即最小值", value=0.0, step=0.1, key="lam_h_min")
            lam_h_max = st.number_input("主队λ即最大值", value=5.0, step=0.1, key="lam_h_max")
        with col_f3:
            lam_a_min = st.number_input("客队λ即最小值", value=0.0, step=0.1, key="lam_a_min")
            lam_a_max = st.number_input("客队λ即最大值", value=5.0, step=0.1, key="lam_a_max")
        ignore_lam = st.checkbox("忽略 λ 筛选（显示所有记录）", value=False)
        show_abnormal_only = st.checkbox("🔍 仅显示异常比赛（单边4+球 / 先进球后被反超/逼平 / 盘口差>1）", value=False)
        handicap_diff_threshold = st.slider("🎯 盘口差异阈值（绝对值）", min_value=0.0, max_value=2.0, value=0.0, step=0.1, help="仅显示 |实际盘口 - 参考盘口| ≥ 此值的比赛，0.0表示不限制")
        df_display = dfh.copy()
        if '日期' in df_display.columns:
            df_display['日期'] = pd.to_datetime(df_display['日期'])
        if filter_actual != "全部":
            df_display = df_display[df_display['实际盘口'].astype(str) == filter_actual]
        if not ignore_lam:
            df_display = df_display[(df_display['λ_主队即'] >= lam_h_min) & (df_display['λ_主队即'] <= lam_h_max)]
            df_display = df_display[(df_display['λ_客队即'] >= lam_a_min) & (df_display['λ_客队即'] <= lam_a_max)]
        df_display['盘口差'] = df_display['实际盘口'] - df_display['盘口']
        if handicap_diff_threshold > 0:
            df_display = df_display[df_display['盘口差'].abs() >= handicap_diff_threshold]
        if show_abnormal_only:
            df_display['is_abnormal'] = df_display.apply(is_abnormal_match, axis=1)
            abnormal_count = df_display['is_abnormal'].sum()
            df_display = df_display[df_display['is_abnormal']]
            if abnormal_count > 0:
                st.info(f"📌 已筛选出 {abnormal_count} 场异常比赛")
            else:
                st.info("📌 当前筛选条件下没有异常比赛")
        if df_display.empty:
            st.info("没有符合条件的记录")
        else:
            display_data = df_display.copy()
            display_data['λ_主队初_fmt'] = display_data['λ_主队初'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            display_data['λ_主队即_fmt'] = display_data['λ_主队即'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            display_data['λ_客队初_fmt'] = display_data['λ_客队初'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            display_data['λ_客队即_fmt'] = display_data['λ_客队即'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            display_data['主胜概率_fmt'] = display_data['主胜概率'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            display_data['主先进球概率_fmt'] = display_data['主先进球概率'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            display_data['盘口差_fmt'] = display_data['盘口差'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
            display_data['权重分'] = display_data['盘口差'].abs() * 0.1
            display_data['权重分_fmt'] = display_data['权重分'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            display_data.rename(columns={'盘口': '参考盘口', '实际盘口': '实际盘口'}, inplace=True)
            show_cols = ['日期','主队','客队','比分','参考盘口','实际盘口','盘口差_fmt','权重分_fmt','预测结果',
                         'λ_主队初_fmt','λ_主队即_fmt','λ_客队初_fmt','λ_客队即_fmt',
                         '主胜概率_fmt','主先进球概率_fmt','先进球方','主队进球','客队进球']
            st.dataframe(display_data[show_cols], use_container_width=True)
            csv_hist = df_display.to_csv(index=False).encode('utf-8-sig')
            st.download_button("导出当前筛选结果", csv_hist, "match_results.csv", "text/csv")

# ========== Tab4 自动获取比赛（新增） ==========
with tab4:
    st.markdown("### 从 Football-Data.org 获取比赛数据")
    st.info("免费版 API 仅支持 **2021、2022、2023 赛季** 的数据。如果您需要 2024 赛季，请升级订阅或改用其他数据源。")

    # 联赛代码映射
    comp_dict = {
        "英超 (Premier League)": "PL",
        "西甲 (La Liga)": "PD",
        "德甲 (Bundesliga)": "BL1",
        "意甲 (Serie A)": "SA",
        "法甲 (Ligue 1)": "FL1",
    }
    comp_name = st.selectbox("选择联赛", list(comp_dict.keys()))
    competition_code = comp_dict[comp_name]

    season = st.text_input("赛季 (请使用 2021, 2022, 2023)", value="2023")
    matchday = st.number_input("比赛轮次 (Matchday)", min_value=1, step=1, value=1)

    # 带重试的请求函数
    import requests
    import time
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    def fetch_matches_with_retry(competition_code, season, matchday, max_retries=3):
        headers = {
            'X-Auth-Token': FOOTBALL_DATA_API_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = f"{FOOTBALL_DATA_BASE_URL}competitions/{competition_code}/matches"
        params = {'season': season, 'matchday': matchday}

        session = requests.Session()
        retries = Retry(total=max_retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        for attempt in range(max_retries):
            try:
                response = session.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    error_msg = response.json().get('message', '')
                    if "restricted" in error_msg.lower():
                        st.error(f"❌ 权限不足：免费版不支持 {season} 赛季的数据。请使用 2021、2022 或 2023 赛季。")
                    else:
                        st.error(f"❌ 访问被拒绝 (403): {error_msg}")
                    return None
                else:
                    st.warning(f"第 {attempt+1} 次请求返回状态码 {response.status_code}")
            except Exception as e:
                st.warning(f"第 {attempt+1} 次请求异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        st.error("多次重试后仍然失败，请检查网络或 API 状态。")
        return None

    # 测试连接按钮
    if st.button("🔌 测试 API 连接"):
        with st.spinner("测试中..."):
            test_result = fetch_matches_with_retry("PL", "2023", 1)
            if test_result:
                st.success("✅ API 连接成功！可以正常获取数据。")
            else:
                st.error("❌ 连接失败，请检查 API Key 或赛季范围（免费版仅支持 2021-2023）。")

    # 获取比赛按钮
    if st.button("📥 获取比赛并预测"):
        with st.spinner("正在请求 API ..."):
            data = fetch_matches_with_retry(competition_code, season, matchday)

        if data is None:
            st.error("未获取到任何比赛，请检查联赛代码、赛季和轮次是否正确。")
        else:
            matches = data.get('matches', [])
            if not matches:
                st.warning(f"该轮次（第 {matchday} 轮）没有比赛，请尝试其他轮次或赛季。")
            else:
                st.success(f"成功获取 {len(matches)} 场比赛！")
                matches_list = []
                for m in matches:
                    home = m['homeTeam']['name']
                    away = m['awayTeam']['name']
                    status = m['status']
                    home_score = m['score']['fullTime']['home']
                    away_score = m['score']['fullTime']['away']
                    if status == 'FINISHED' and home_score is not None:
                        result_str = f"{home_score}:{away_score}"
                    else:
                        result_str = "未开始" if status == 'SCHEDULED' else "进行中"
                    matches_list.append({
                        '比赛日期': m['utcDate'],
                        '主队': home,
                        '客队': away,
                        '状态': status,
                        '比分': result_str,
                        '主队进球': home_score if status == 'FINISHED' else None,
                        '客队进球': away_score if status == 'FINISHED' else None
                    })
                df_matches = pd.DataFrame(matches_list)
                st.dataframe(df_matches, use_container_width=True)

                # 批量录入按钮
                col_batch1, col_batch2 = st.columns(2)
                with col_batch1:
                    if st.button("📦 批量录入已结束比赛"):
                        new_count = 0
                        skip_count = 0
                        df_existing = load_results()
                        for idx, row in df_matches.iterrows():
                            if row['状态'] == 'FINISHED':
                                date_str = row['比赛日期'][:10] if pd.notna(row['比赛日期']) else datetime.now().strftime("%Y-%m-%d")
                                home = row['主队']
                                away = row['客队']
                                home_score = row['主队进球'] if pd.notna(row['主队进球']) else 0
                                away_score = row['客队进球'] if pd.notna(row['客队进球']) else 0

                                exists = df_existing[
                                    (df_existing['日期'] == date_str) &
                                    (df_existing['主队'] == home) &
                                    (df_existing['客队'] == away)
                                ].shape[0] > 0
                                if not exists:
                                    save_result(
                                        date=date_str,
                                        home_team=home,
                                        away_team=away,
                                        home_score=home_score,
                                        away_score=away_score,
                                        handicap=0.0,
                                        actual_handicap=0.0,
                                        predict_result="",
                                        lam_h_init=0.0,
                                        lam_h_live=0.0,
                                        lam_a_init=0.0,
                                        lam_a_live=0.0,
                                        home_prob=0.0,
                                        home_first=0.0,
                                        first_scorer="",
                                        odds_fingerprint=""
                                    )
                                    new_count += 1
                                else:
                                    skip_count += 1
                        st.success(f"批量录入完成：新增 {new_count} 场，跳过已存在 {skip_count} 场。")
                        st.rerun()

                st.markdown("---")
                st.subheader("🔮 基于球队历史实力的预测（泊松模型）")
                for idx, row in df_matches.iterrows():
                    home = row['主队']
                    away = row['客队']
                    status = row['状态']
                    result_str = row['比分']

                    st.markdown(f"#### {home} vs {away} ({status})")
                    if status == 'FINISHED':
                        st.write(f"**实际结果**：{result_str}")
                        # 单场录入按钮（可选，通常批量已够用）
                        if st.button(f"📝 单场录入", key=f"single_save_{idx}"):
                            date_str = row['比赛日期'][:10] if pd.notna(row['比赛日期']) else datetime.now().strftime("%Y-%m-%d")
                            home_score_val = row['主队进球'] if pd.notna(row['主队进球']) else 0
                            away_score_val = row['客队进球'] if pd.notna(row['客队进球']) else 0
                            # 简单去重检查（略，可参考批量逻辑）
                            save_result(
                                date=date_str,
                                home_team=home,
                                away_team=away,
                                home_score=home_score_val,
                                away_score=away_score_val,
                                handicap=0.0,
                                actual_handicap=0.0,
                                predict_result="",
                                lam_h_init=0.0,
                                lam_h_live=0.0,
                                lam_a_init=0.0,
                                lam_a_live=0.0,
                                home_prob=0.0,
                                home_first=0.0,
                                first_scorer="",
                                odds_fingerprint=""
                            )
                            st.success(f"已保存 {home} {home_score_val}:{away_score_val} {away}")
                            st.rerun()
                    else:
                        # 调用预测函数（请确保 simple_predict_by_team_strength 已定义）
                        try:
                            probs, ou_rec, likely_score, (lam_h, lam_a) = simple_predict_by_team_strength(home, away)
                            if probs is None:
                                st.warning(f"球队 {home} 或 {away} 历史数据不足，无法预测。")
                            else:
                                p_h, p_d, p_a = probs
                                st.write(f"**预测胜平负**：主胜 {p_h:.1%} | 平局 {p_d:.1%} | 客胜 {p_a:.1%}")
                                st.write(f"**大小球**：{ou_rec} (预期总进球 {lam_h+lam_a:.2f})")
                                st.write(f"**最可能比分**：{likely_score}")
                        except NameError:
                            st.warning("预测函数 simple_predict_by_team_strength 未定义，请确认该函数已存在。")
                    st.markdown("---")
