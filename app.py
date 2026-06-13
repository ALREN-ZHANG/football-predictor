import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import shutil
import time
import hashlib
import json
import requests
import unicodedata
from collections import Counter
from scipy.stats import poisson
from scipy.optimize import minimize
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========== 密码保护配置 ==========
APP_PASSWORD = "admin123"  # 请修改为您自己的密码

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(layout="wide")
    st.title("🔐 登录")
    password = st.text_input("请输入密码", type="password")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("登录"):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ========== Football-Data.org API 配置 ==========
FOOTBALL_DATA_API_KEY = "e081ba0cb14245668dc3ce2de29e7ff2"
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

# ========== 辅助函数：清理球队名称 ==========
def clean_team_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = unicodedata.normalize('NFKC', name)
    name = re.sub(r'\s+', '', name)
    return name.strip().lower()

# ========== 历史记录管理 & 备份 ==========
RESULTS_FILE = "match_results.csv"
BACKUP_DIR = "backups"

def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

def get_backup_files():
    ensure_backup_dir()
    files = [f for f in os.listdir(BACKUP_DIR) if f.startswith("match_results_") and f.endswith(".csv")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(BACKUP_DIR, x)), reverse=True)
    return files

def backup_database():
    ensure_backup_dir()
    if not os.path.exists(RESULTS_FILE):
        return False, "数据库文件不存在，无法备份。"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"match_results_{timestamp}.csv"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    try:
        shutil.copy2(RESULTS_FILE, backup_path)
        return True, backup_name
    except Exception as e:
        return False, str(e)

def restore_database(backup_name):
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.exists(backup_path):
        return False, "备份文件不存在。"
    try:
        shutil.copy2(backup_path, RESULTS_FILE)
        return True, "恢复成功，请刷新页面。"
    except Exception as e:
        return False, str(e)

def load_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=[
            '日期','主队','客队','主队进球','客队进球','比分',
            '参考盘口','实际盘口','预测结果','先进球方',
            'sb_λ_主队初','sb_λ_主队终','sb_λ_客队初','sb_λ_客队终',
            'xl_λ_主队初','xl_λ_主队终','xl_λ_客队初','xl_λ_客队终',
            'odds_fingerprint'
        ])
    try:
        df = pd.read_csv(RESULTS_FILE, encoding='utf-8-sig')
    except:
        df = pd.read_csv(RESULTS_FILE, encoding='gbk')
    required_cols = [
        '日期','主队','客队','主队进球','客队进球','比分',
        '参考盘口','实际盘口','预测结果','先进球方',
        'sb_λ_主队初','sb_λ_主队终','sb_λ_客队初','sb_λ_客队终',
        'xl_λ_主队初','xl_λ_主队终','xl_λ_客队初','xl_λ_客队终',
        'odds_fingerprint'
    ]
    str_cols = ['日期', '主队', '客队', '比分', '预测结果', '先进球方', 'odds_fingerprint']
    for col in required_cols:
        if col not in df.columns:
            if col in str_cols:
                df[col] = ''
            else:
                df[col] = 0.0
    num_cols = [c for c in required_cols if c not in str_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['主队'] = df['主队'].apply(clean_team_name)
    df['客队'] = df['客队'].apply(clean_team_name)
    return df

def save_result(date, home_team, away_team, home_score, away_score,
                ref_handicap, actual_handicap, predict_result, first_scorer,
                sb_h_init, sb_h_live, sb_a_init, sb_a_live,
                xl_h_init, xl_h_live, xl_a_init, xl_a_live,
                odds_fingerprint=''):
    home_team = clean_team_name(home_team)
    away_team = clean_team_name(away_team)
    df = load_results()
    score_str = f"{home_score}:{away_score}"
    new_row = pd.DataFrame([{
        '日期': date, '主队': home_team, '客队': away_team,
        '主队进球': home_score, '客队进球': away_score, '比分': score_str,
        '参考盘口': ref_handicap, '实际盘口': actual_handicap,
        '预测结果': predict_result, '先进球方': first_scorer,
        'sb_λ_主队初': sb_h_init, 'sb_λ_主队终': sb_h_live,
        'sb_λ_客队初': sb_a_init, 'sb_λ_客队终': sb_a_live,
        'xl_λ_主队初': xl_h_init, 'xl_λ_主队终': xl_h_live,
        'xl_λ_客队初': xl_a_init, 'xl_λ_客队终': xl_a_live,
        'odds_fingerprint': odds_fingerprint
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    try:
        df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
        return df
    except Exception as e:
        st.error(f"保存失败: {e}")
        return None

def is_score_col(col):
    return bool(re.match(r'^\d+[:：\-]\d+$', str(col).strip()))

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
    rows, last_company = [], None
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r'\s+', line.strip())
        if drop_first_col and len(parts) > 0 and re.match(r'^\d+$', parts[0]):
            parts = parts[1:]
        if len(parts) >= 1 and parts[0] == '即':
            if last_company is None:
                return None, "无法找到“即”行对应的公司名"
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
        return None
    norm_score_cols = {c: normalize_score_col(c) for c in score_cols}
    records = []
    for _, row in df.iterrows():
        company = row[company_col] if company_col in row else ""
        if not ('SB' in company.upper() or '沙巴' in company or '小利' in company or 'XIAOLI' in company.upper()):
            continue
        category = row[type_col] if type_col and type_col in row and pd.notna(row[type_col]) else ""
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

def optimize_lambdas(odds_dict, max_goals=4):
    scores = list(odds_dict.keys())
    odds = np.array([odds_dict[s] for s in scores])
    raw_probs = 1.0 / odds
    total_raw = np.sum(raw_probs)
    overround = 1.0 / total_raw
    fair_probs = raw_probs / total_raw
    score_index = {}
    idx = 0
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            score_index[f"{x}:{y}"] = idx; idx += 1
    target_probs = np.zeros((max_goals+1)*(max_goals+1))
    for s, p in zip(scores, fair_probs):
        if s in score_index:
            target_probs[score_index[s]] = p
    def poisson_probs(lam_h, lam_a):
        probs = []
        for x in range(max_goals+1):
            for y in range(max_goals+1):
                probs.append(poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a))
        return np.array(probs)
    def objective(params):
        lam_h, lam_a = params
        if lam_h <= 0 or lam_a <= 0:
            return 1e10
        model_probs = poisson_probs(lam_h, lam_a)
        diff = model_probs - target_probs
        return np.sum(diff**2)
    home_marginal, away_marginal = {}, {}
    for s, p in zip(scores, fair_probs):
        x, y = map(int, s.split(':'))
        home_marginal[x] = home_marginal.get(x, 0.0) + p
        away_marginal[y] = away_marginal.get(y, 0.0) + p
    lam_h_init = sum(x * home_marginal.get(x, 0.0) for x in range(max_goals+1))
    lam_a_init = sum(y * away_marginal.get(y, 0.0) for y in range(max_goals+1))
    lam_h_init = max(0.1, min(5.0, lam_h_init))
    lam_a_init = max(0.1, min(5.0, lam_a_init))
    result = minimize(objective, [lam_h_init, lam_a_init], bounds=[(0.1,5.0),(0.1,5.0)], method='L-BFGS-B')
    lam_h_opt, lam_a_opt = result.x
    return lam_h_opt, lam_a_opt, overround

def extract_company_lambdas(df):
    company_col = None
    for col in df.columns:
        if '公司' in col or '序' in col:
            company_col = col; break
    if company_col is None:
        company_col = df.columns[0]
    type_col = None
    for col in df.columns:
        if '分类' in col:
            type_col = col; break
    if type_col is None:
        for col in df.columns:
            if df[col].astype(str).str.contains('初|即').any():
                type_col = col; break
    score_cols = [c for c in df.columns if is_score_col(c)]
    if not score_cols:
        return (None,)*9 + (None,)
    norm = {c: normalize_score_col(c) for c in score_cols}
    data = {}
    for _, row in df.iterrows():
        company = row[company_col] if company_col in row else ""
        is_sb = 'SB' in company.upper() or '沙巴' in company
        is_xiaoli = '小利' in company or 'XIAOLI' in company.upper()
        if not (is_sb or is_xiaoli):
            continue
        cat = row[type_col] if type_col and type_col in row and pd.notna(row[type_col]) else ""
        if not cat:
            continue
        odds = {}
        for orig in score_cols:
            std = norm[orig]
            v = row[orig]
            if pd.notna(v) and v > 0:
                odds[std] = float(v)
        if not odds:
            continue
        lam_h, lam_a, _ = optimize_lambdas(odds)
        if company not in data:
            data[company] = {}
        data[company][cat] = {'h': lam_h, 'a': lam_a}
    sb_init_h = sb_init_a = sb_live_h = sb_live_a = None
    xl_init_h = xl_init_a = xl_live_h = xl_live_a = None
    for company, cats in data.items():
        if 'SB' in company.upper() or '沙巴' in company:
            if '初' in cats:
                sb_init_h, sb_init_a = cats['初']['h'], cats['初']['a']
            if '即' in cats:
                sb_live_h, sb_live_a = cats['即']['h'], cats['即']['a']
        if '小利' in company or 'XIAOLI' in company.upper():
            if '初' in cats:
                xl_init_h, xl_init_a = cats['初']['h'], cats['初']['a']
            if '即' in cats:
                xl_live_h, xl_live_a = cats['即']['h'], cats['即']['a']
    fingerprint = compute_odds_fingerprint(df)
    return (sb_init_h, sb_init_a, sb_live_h, sb_live_a,
            xl_init_h, xl_init_a, xl_live_h, xl_live_a,
            fingerprint, None)

def get_team_recent_lam(team_name, is_home, company_type='SB', ignore_home_away=False):
    team_name = clean_team_name(team_name)
    if not team_name:
        return None
    df = load_results()
    if df.empty:
        return None
    if ignore_home_away:
        mask = (df['主队'] == team_name) | (df['客队'] == team_name)
        df_team = df[mask].copy()
        if df_team.empty:
            return None
        def get_lam(row):
            if row['主队'] == team_name:
                return row['sb_λ_主队终'] if company_type=='SB' else row['xl_λ_主队终']
            else:
                return row['sb_λ_客队终'] if company_type=='SB' else row['xl_λ_客队终']
        df_team['lam_val'] = df_team.apply(get_lam, axis=1).dropna()
        df_team = df_team.sort_values('日期', ascending=False).head(5)
        lambdas = df_team['lam_val'].tolist()
        return np.mean(lambdas) if lambdas else None
    else:
        if is_home:
            mask = (df['主队'] == team_name)
            lam_col = 'sb_λ_主队终' if company_type=='SB' else 'xl_λ_主队终'
        else:
            mask = (df['客队'] == team_name)
            lam_col = 'sb_λ_客队终' if company_type=='SB' else 'xl_λ_客队终'
        df_team = df[mask].copy()
        if df_team.empty:
            return None
        df_team = df_team.sort_values('日期', ascending=False).head(5)
        lambdas = [row[lam_col] for _, row in df_team.iterrows() if pd.notna(row[lam_col]) and row[lam_col]>0]
        return np.mean(lambdas) if lambdas else None

def get_most_likely_score(lam_h, lam_a, max_goals=4):
    max_prob = 0
    best_score = "0:0"
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
            if prob > max_prob:
                max_prob, best_score = prob, f"{x}:{y}"
    return best_score

def fetch_matches_with_retry(competition_code, season, matchday, max_retries=3):
    headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY, 'User-Agent': 'Mozilla/5.0'}
    url = f"{FOOTBALL_DATA_BASE_URL}competitions/{competition_code}/matches"
    params = {'season': season, 'matchday': matchday}
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=1, status_forcelist=[500,502,503,504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 403:
                st.error("权限不足，请使用2021-2023赛季")
                return None
        except Exception as e:
            st.warning(f"请求异常: {e}")
        time.sleep(2)
    return None

def simple_predict_by_team_strength(team_h, team_a, ignore_home_away=False):
    lam_h = get_team_recent_lam(team_h, True, 'SB', ignore_home_away)
    lam_a = get_team_recent_lam(team_a, False, 'SB', ignore_home_away)
    if lam_h is None or lam_a is None:
        return None, None, None, None
    p_h = p_d = p_a = 0.0
    for x in range(5):
        for y in range(5):
            prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
            if x > y: p_h += prob
            elif x == y: p_d += prob
            else: p_a += prob
    total = p_h + p_d + p_a
    if total > 0:
        p_h, p_d, p_a = p_h/total, p_d/total, p_a/total
    total_goals = lam_h + lam_a
    ou = "大2.5" if total_goals > 2.8 else ("小2.5" if total_goals < 2.2 else "不明确")
    score = get_most_likely_score(lam_h, lam_a)
    return (p_h, p_d, p_a), ou, score, (lam_h, lam_a)

handicap_options = [-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 
                     0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

def handicap_from_diff(diff):
    diff_clip = max(-3.0, min(3.0, diff))
    raw = - diff_clip * 0.5
    return min(handicap_options, key=lambda x: abs(x - raw))

# ========== 新增：基于四个 λ 终值查找最接近的历史比赛 ==========
def find_similar_by_lambdas(current_lambdas, top_n=4):
    """
    根据当前四个 λ 终值（SB主、SB客、小利主、小利客）查找历史最接近的比赛
    current_lambdas: dict 包含 'sb_h', 'sb_a', 'xl_h', 'xl_a'
    """
    df = load_results()
    if df.empty:
        return pd.DataFrame()
    
    # 需要的列
    required_cols = ['sb_λ_主队终', 'sb_λ_客队终', 'xl_λ_主队终', 'xl_λ_客队终']
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()
    
    # 复制并过滤掉 λ 缺失的行（任一为0或NaN）
    df_clean = df.copy()
    for col in required_cols:
        df_clean = df_clean[df_clean[col].notna() & (df_clean[col] > 0)]
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # 构造当前向量
    curr_vec = np.array([
        current_lambdas.get('sb_h', 0.0),
        current_lambdas.get('sb_a', 0.0),
        current_lambdas.get('xl_h', 0.0),
        current_lambdas.get('xl_a', 0.0)
    ])
    
    # 计算欧氏距离
    def euclidean_distance(row):
        hist_vec = np.array([
            row['sb_λ_主队终'], row['sb_λ_客队终'],
            row['xl_λ_主队终'], row['xl_λ_客队终']
        ])
        return np.linalg.norm(hist_vec - curr_vec)
    
    df_clean['distance'] = df_clean.apply(euclidean_distance, axis=1)
    df_sorted = df_clean.sort_values('distance').head(top_n)
    return df_sorted[['日期', '主队', '客队', '比分', '先进球方', 
                      'sb_λ_主队终', 'sb_λ_客队终', 'xl_λ_主队终', 'xl_λ_客队终', 'distance']]

# ========== 相似度计算函数（用于 TAB1 和 TAB2） ==========
def poisson_prob_vector(lam_h, lam_a, max_goals=4):
    vec = []
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            vec.append(poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a))
    return np.array(vec)

def compute_similarity_for_tab1(row, current_vals):
    # 用于 TAB1 相似比赛推荐
    change_sb_h = row['sb_λ_主队终'] - row['sb_λ_主队初']
    change_sb_a = row['sb_λ_客队终'] - row['sb_λ_客队初']
    change_xl_h = row['xl_λ_主队终'] - row['xl_λ_主队初']
    change_xl_a = row['xl_λ_客队终'] - row['xl_λ_客队初']
    curr_change_sb_h = current_vals['sb_h_live'] - current_vals['sb_h_init']
    curr_change_sb_a = current_vals['sb_a_live'] - current_vals['sb_a_init']
    curr_change_xl_h = current_vals['xl_h_live'] - current_vals['xl_h_init']
    curr_change_xl_a = current_vals['xl_a_live'] - current_vals['xl_a_init']
    change_dist = np.sqrt((change_sb_h - curr_change_sb_h)**2 +
                          (change_sb_a - curr_change_sb_a)**2 +
                          (change_xl_h - curr_change_xl_h)**2 +
                          (change_xl_a - curr_change_xl_a)**2)
    abs_vals_hist = np.array([
        row['sb_λ_主队初'], row['sb_λ_主队终'],
        row['sb_λ_客队初'], row['sb_λ_客队终'],
        row['xl_λ_主队初'], row['xl_λ_主队终'],
        row['xl_λ_客队初'], row['xl_λ_客队终']
    ])
    abs_vals_curr = np.array([
        current_vals['sb_h_init'], current_vals['sb_h_live'],
        current_vals['sb_a_init'], current_vals['sb_a_live'],
        current_vals['xl_h_init'], current_vals['xl_h_live'],
        current_vals['xl_a_init'], current_vals['xl_a_live']
    ])
    abs_dist = np.linalg.norm(abs_vals_hist - abs_vals_curr)
    total_dist = change_dist + abs_dist
    main_sim = 1.0 / (1.0 + total_dist)

    hist_diffs = [
        row['sb_λ_主队初'] - row['sb_λ_客队初'],
        row['sb_λ_主队终'] - row['sb_λ_客队终'],
        row['xl_λ_主队初'] - row['xl_λ_客队初'],
        row['xl_λ_主队终'] - row['xl_λ_客队终']
    ]
    curr_diffs = [
        current_vals['sb_h_init'] - current_vals['sb_a_init'],
        current_vals['sb_h_live'] - current_vals['sb_a_live'],
        current_vals['xl_h_init'] - current_vals['xl_a_init'],
        current_vals['xl_h_live'] - current_vals['xl_a_live']
    ]
    sign_match_count = sum(1 for h, c in zip(hist_diffs, curr_diffs) if (h>=0 and c>=0) or (h<=0 and c<=0))
    sign_sim = sign_match_count / 4.0
    enhanced_main_sim = 0.8 * main_sim + 0.2 * sign_sim

    lam_h_hist = row['sb_λ_主队终']
    lam_a_hist = row['sb_λ_客队终']
    lam_h_curr = current_vals['sb_h_live']
    lam_a_curr = current_vals['sb_a_live']
    if lam_h_hist > 0 and lam_a_hist > 0 and lam_h_curr > 0 and lam_a_curr > 0:
        vec_hist = poisson_prob_vector(lam_h_hist, lam_a_hist)
        vec_curr = poisson_prob_vector(lam_h_curr, lam_a_curr)
        norm_hist = np.linalg.norm(vec_hist)
        norm_curr = np.linalg.norm(vec_curr)
        poisson_cos = np.dot(vec_hist, vec_curr) / (norm_hist * norm_curr) if norm_hist>0 and norm_curr>0 else 0.0
    else:
        poisson_cos = 0.0
    first_sim = 1.0 if row['先进球方'] == current_vals['first_scorer'] else 0.5
    extra_sim = 0.9 * poisson_cos + 0.1 * first_sim
    final_sim = 0.9 * enhanced_main_sim + 0.1 * extra_sim
    return final_sim

def compute_similarity_for_tab2(row, current_vals):
    # 用于 TAB2 相似度计算（与 TAB1 相同，但独立命名以区分）
    return compute_similarity_for_tab1(row, current_vals)

def find_similar_matches(current_vals, top_n=10):
    df = load_results()
    if df.empty:
        return pd.DataFrame()
    # 严格匹配参考盘口和实际盘口
    ref_match = df['参考盘口'].apply(lambda x: abs(x - current_vals['ref_handicap']) < 0.001)
    actual_match = df['实际盘口'].apply(lambda x: abs(x - current_vals['actual_handicap']) < 0.001)
    df_filtered = df[ref_match & actual_match].copy()
    if df_filtered.empty:
        return pd.DataFrame()
    similarities = []
    for _, row in df_filtered.iterrows():
        sim = compute_similarity_for_tab2(row, current_vals)
        similarities.append(sim)
    df_filtered['similarity'] = similarities
    return df_filtered.sort_values('similarity', ascending=False).head(top_n)

# ========== 初始化 session_state ==========
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "📈 赔率分析"
if 'tab2_home_team' not in st.session_state:
    st.session_state.tab2_home_team = ""
if 'tab2_away_team' not in st.session_state:
    st.session_state.tab2_away_team = ""
if 'tab2_actual_handicap' not in st.session_state:
    st.session_state.tab2_actual_handicap = 0.0
if 'tab2_reference_handicap' not in st.session_state:
    st.session_state.tab2_reference_handicap = 0.0
if 'tab2_predict_result' not in st.session_state:
    st.session_state.tab2_predict_result = ""

st.set_page_config(layout="wide")
st.title("⚽ 足球预测（SB/小利 双公司λ记录）")

tab_labels = ["📈 赔率分析", "📝 录入比赛", "📊 历史记录", "🆕 自动获取比赛"]
cols = st.columns(len(tab_labels))
for i, label in enumerate(tab_labels):
    if cols[i].button(label, key=f"tab_btn_{i}", use_container_width=True):
        st.session_state.current_tab = label
        st.rerun()
st.markdown("---")

# ========== TAB1 赔率分析 ==========
if st.session_state.current_tab == "📈 赔率分析":
    with st.container():
        st.markdown("### 波胆赔率分析")
        st.info("粘贴波胆赔率数据（制表符或空格分隔），至少包含公司、分类（初/即）以及各波胆比分列。系统将只解析 SB 和小利公司。")
        auto_parse = st.checkbox("自动解析（粘贴后自动执行）", value=True)
        pasted = st.text_area("粘贴数据", height=200, key="pasted_area")
        if auto_parse and pasted and 'last_pasted' not in st.session_state:
            st.session_state.last_pasted = pasted
            df_source, err = parse_pasted_text(pasted)
            if err:
                st.error(err)
            else:
                st.session_state.df_source = df_source
                st.success("自动解析成功！")
                st.rerun()
        if st.button("🔍 手动解析"):
            if pasted:
                df_source, err = parse_pasted_text(pasted)
                if err:
                    st.error(err)
                else:
                    st.session_state.df_source = df_source
                    st.success("解析成功！")
                    st.rerun()
            else:
                st.warning("请粘贴数据")
        if 'df_source' in st.session_state:
            df_source = st.session_state.df_source
            (sb_init_h, sb_init_a, sb_live_h, sb_live_a,
             xl_init_h, xl_init_a, xl_live_h, xl_live_a,
             fingerprint, err) = extract_company_lambdas(df_source)
            if err:
                st.error(err)
            else:
                st.session_state['current_fingerprint'] = fingerprint
                df_history = load_results()
                if not df_history.empty and 'odds_fingerprint' in df_history.columns:
                    matched = df_history[df_history['odds_fingerprint'] == fingerprint]
                    if not matched.empty:
                        st.warning(f"⚠️ 该赔率组合已出现过 {len(matched)} 次，历史比赛结果如下：")
                        for _, row in matched.iterrows():
                            st.write(f"📅 {row['日期']} : {row['主队']} {row['比分']} {row['客队']}  (先进球: {row['先进球方']})")
                        st.info("您可以继续使用当前数据，录入比赛时会自动关联此指纹。")
                    else:
                        st.info("✅ 该赔率组合未出现在历史记录中。")
                st.session_state['sb'] = {
                    'init_h': sb_init_h or 0.0, 'init_a': sb_init_a or 0.0,
                    'live_h': sb_live_h or 0.0, 'live_a': sb_live_a or 0.0
                }
                st.session_state['xl'] = {
                    'init_h': xl_init_h or 0.0, 'init_a': xl_init_a or 0.0,
                    'live_h': xl_live_h or 0.0, 'live_a': xl_live_a or 0.0
                }

                # 显示SB和小利即盘λ
                if sb_live_h is not None and sb_live_h > 0:
                    st.info(f"📊 SB（沙巴）即盘λ：主队 {sb_live_h:.3f} | 客队 {sb_live_a:.3f}")
                if xl_live_h is not None and xl_live_h > 0:
                    st.info(f"📊 小利即盘λ：主队 {xl_live_h:.3f} | 客队 {xl_live_a:.3f}")
                if (sb_live_h is None or sb_live_h == 0) and (xl_live_h is None or xl_live_h == 0):
                    st.warning("未能提取到有效λ值")
                # 为后续预测保留优先使用的市场数据（SB优先，若无效则用小利）
                if sb_live_h is not None and sb_live_h > 0:
                    market_h, market_a, source = sb_live_h, sb_live_a, "SB"
                elif xl_live_h is not None and xl_live_h > 0:
                    market_h, market_a, source = xl_live_h, xl_live_a, "小利"
                else:
                    market_h, market_a, source = 0.0, 0.0, "无"

                # 基于λ终值的历史相近比赛推荐（独立于盘口）
                st.markdown("---")
                st.markdown("## 🔍 历史 λ 终值相近比赛推荐（独立于盘口）")
                current_lambdas = {
                    'sb_h': sb_live_h if sb_live_h else 0.0,
                    'sb_a': sb_live_a if sb_live_a else 0.0,
                    'xl_h': xl_live_h if xl_live_h else 0.0,
                    'xl_a': xl_live_a if xl_live_a else 0.0
                }
                valid_sb = current_lambdas['sb_h'] > 0 and current_lambdas['sb_a'] > 0
                valid_xl = current_lambdas['xl_h'] > 0 and current_lambdas['xl_a'] > 0
                if valid_sb or valid_xl:
                    similar_df = find_similar_by_lambdas(current_lambdas, top_n=4)
                    if not similar_df.empty:
                        st.success(f"找到 {len(similar_df)} 场 λ 终值最接近的历史比赛（基于SB+小利即盘λ）")
                        for _, row in similar_df.iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"**{row['日期']}**")
                                    st.markdown(f"{row['主队']} {row['比分']} {row['客队']}")
                                    st.markdown(f"先进球: **{row['先进球方']}**")
                                with col2:
                                    st.markdown(f"SB λ: 主 {row['sb_λ_主队终']:.3f} / 客 {row['sb_λ_客队终']:.3f}")
                                    st.markdown(f"小利 λ: 主 {row['xl_λ_主队终']:.3f} / 客 {row['xl_λ_客队终']:.3f}")
                                    st.caption(f"欧氏距离: {row['distance']:.4f}")
                                st.markdown("---")
                    else:
                        st.info("历史数据库中暂无 λ 终值有效且相近的比赛。")
                else:
                    st.warning("当前解析出的 λ 终值数据不足（至少需要一家公司的完整主客 λ），无法进行相近推荐。")

                st.markdown("### 🏟️ 输入球队名称进行预测")
                col_team1, col_team2 = st.columns(2)
                with col_team1:
                    team_h = st.text_input("主队名称", key="strength_h", value=st.session_state.tab2_home_team)
                    st.session_state.tab2_home_team = team_h
                with col_team2:
                    team_a = st.text_input("客队名称", key="strength_a", value=st.session_state.tab2_away_team)
                    st.session_state.tab2_away_team = team_a
                ignore_home_away = st.checkbox("不分主客场（取最近5场所有比赛）", value=False)
                actual_handicap = st.selectbox("实际盘口（用于参考）", handicap_options,
                                               index=handicap_options.index(st.session_state.tab2_actual_handicap), key="actual_hc_tab1")
                st.session_state.tab2_actual_handicap = actual_handicap
                st.caption("此盘口将同步到「录入比赛」Tab 的实际盘口字段")
                if st.button("🔮 生成预测", key="predict_btn"):
                    if not team_h or not team_a:
                        st.warning("请填写主队和客队名称")
                    else:
                        sb_home_lam = get_team_recent_lam(team_h, True, 'SB', ignore_home_away)
                        sb_away_lam = get_team_recent_lam(team_a, False, 'SB', ignore_home_away)
                        xl_home_lam = get_team_recent_lam(team_h, True, '小利', ignore_home_away)
                        xl_away_lam = get_team_recent_lam(team_a, False, '小利', ignore_home_away)
                        theoretical_hc = handicap_from_diff(market_h - market_a) if market_h>0 else 0.0
                        st.session_state['tab2_reference_handicap'] = theoretical_hc

                        # 查找相似比赛（盘口完全匹配）
                        df_hist = load_results()
                        if not df_hist.empty:
                            ref_match = df_hist['参考盘口'].apply(lambda x: abs(x - theoretical_hc) < 0.001)
                            actual_match = df_hist['实际盘口'].apply(lambda x: abs(x - actual_handicap) < 0.001)
                            df_same_hc = df_hist[ref_match & actual_match].copy()
                            if not df_same_hc.empty:
                                current_vals = {
                                    'sb_h_init': sb_init_h or 0.0, 'sb_h_live': sb_live_h or 0.0,
                                    'sb_a_init': sb_init_a or 0.0, 'sb_a_live': sb_live_a or 0.0,
                                    'xl_h_init': xl_init_h or 0.0, 'xl_h_live': xl_live_h or 0.0,
                                    'xl_a_init': xl_init_a or 0.0, 'xl_a_live': xl_live_a or 0.0,
                                    'first_scorer': "主队"
                                }
                                similarities = []
                                for _, row in df_same_hc.iterrows():
                                    sim = compute_similarity_for_tab1(row, current_vals)
                                    similarities.append(sim)
                                df_same_hc['similarity'] = similarities
                                # 修改阈值：从 0.9 改为 0.85
                                high_sim = df_same_hc[df_same_hc['similarity'] > 0.85].sort_values('similarity', ascending=False)
                                if not high_sim.empty:
                                    st.markdown("---")
                                    st.markdown("## 🔍 发现高相似历史比赛（盘口完全相同，λ 相似度 > 0.85）")
                                    for _, rec in high_sim.head(5).iterrows():
                                        st.markdown(f"**📅 {rec['日期']} : {rec['主队']} {rec['比分']} {rec['客队']}**")
                                        st.write(f"参考盘口: {rec['参考盘口']:.2f} | 实际盘口: {rec['实际盘口']:.2f} | 先进球方: {rec['先进球方']}")
                                        st.write(f"SB λ: 主初 {rec['sb_λ_主队初']:.3f} 主终 {rec['sb_λ_主队终']:.3f} | 客初 {rec['sb_λ_客队初']:.3f} 客终 {rec['sb_λ_客队终']:.3f}")
                                        st.write(f"小利 λ: 主初 {rec['xl_λ_主队初']:.3f} 主终 {rec['xl_λ_主队终']:.3f} | 客初 {rec['xl_λ_客队初']:.3f} 客终 {rec['xl_λ_客队终']:.3f}")
                                        st.write(f"相似度: {rec['similarity']:.4f}")
                                        st.markdown("---")
                                else:
                                    st.info("盘口完全匹配的历史比赛中，未找到 λ 相似度 > 0.85 的比赛。")

                        st.markdown("---")
                        st.markdown("## 📊 历史实力分析（最近5场）")
                        col_sb, col_xl = st.columns(2)
                        with col_sb:
                            st.markdown("**SB（沙巴）公司**")
                            if ignore_home_away:
                                st.write(f"主队λ终平均值: {sb_home_lam:.3f}" if sb_home_lam else "主队λ: 数据不足")
                                st.write(f"客队λ终平均值: {sb_away_lam:.3f}" if sb_away_lam else "客队λ: 数据不足")
                            else:
                                st.write(f"主队主场λ终平均值: {sb_home_lam:.3f}" if sb_home_lam else "主队主场λ: 数据不足")
                                st.write(f"客队客场λ终平均值: {sb_away_lam:.3f}" if sb_away_lam else "客队客场λ: 数据不足")
                        with col_xl:
                            st.markdown("**小利公司**")
                            if ignore_home_away:
                                st.write(f"主队λ终平均值: {xl_home_lam:.3f}" if xl_home_lam else "主队λ: 数据不足")
                                st.write(f"客队λ终平均值: {xl_away_lam:.3f}" if xl_away_lam else "客队λ: 数据不足")
                            else:
                                st.write(f"主队主场λ终平均值: {xl_home_lam:.3f}" if xl_home_lam else "主队主场λ: 数据不足")
                                st.write(f"客队客场λ终平均值: {xl_away_lam:.3f}" if xl_away_lam else "客队客场λ: 数据不足")
                        st.markdown("## 🎯 预测波胆 vs 市场波胆对比")
                        if sb_home_lam and sb_away_lam:
                            sb_pred = get_most_likely_score(sb_home_lam, sb_away_lam)
                            st.markdown(f"**SB历史模型**：最可能比分 **{sb_pred}**")
                        if xl_home_lam and xl_away_lam:
                            xl_pred = get_most_likely_score(xl_home_lam, xl_away_lam)
                            st.markdown(f"**小利历史模型**：最可能比分 **{xl_pred}**")
                        if market_h > 0:
                            market_pred = get_most_likely_score(market_h, market_a)
                            st.markdown(f"**市场波胆**（{source}即盘）：最可能比分 **{market_pred}**")
                        if sb_home_lam and sb_away_lam:
                            default_pred = get_most_likely_score(sb_home_lam, sb_away_lam)
                        elif xl_home_lam and xl_away_lam:
                            default_pred = get_most_likely_score(xl_home_lam, xl_away_lam)
                        elif market_h > 0:
                            default_pred = market_pred
                        else:
                            default_pred = ""
                        st.session_state['tab2_predict_result'] = default_pred
                        st.markdown(f"**理论盘口（基于市场λ差值）**：{theoretical_hc:+.2f}")
                        st.markdown(f"**您选择的实际盘口**：{actual_handicap:+.2f}")
                        st.markdown("## 💡 推荐建议")
                        def get_probs(lam_h, lam_a):
                            if lam_h is None or lam_a is None or lam_h<=0 or lam_a<=0:
                                return None
                            p_h = p_d = p_a = 0.0
                            for x in range(5):
                                for y in range(5):
                                    prob = poisson.pmf(x, lam_h) * poisson.pmf(y, lam_a)
                                    if x>y: p_h+=prob
                                    elif x==y: p_d+=prob
                                    else: p_a+=prob
                            total = p_h+p_d+p_a
                            return (p_h/total, p_d/total, p_a/total) if total>0 else None
                        if sb_home_lam and sb_away_lam:
                            probs = get_probs(sb_home_lam, sb_away_lam)
                            if probs:
                                st.write(f"**SB历史模型**：主胜 {probs[0]:.1%} | 平局 {probs[1]:.1%} | 客胜 {probs[2]:.1%}")
                        else:
                            st.write("**SB历史模型**：数据不足")
                        if xl_home_lam and xl_away_lam:
                            probs = get_probs(xl_home_lam, xl_away_lam)
                            if probs:
                                st.write(f"**小利历史模型**：主胜 {probs[0]:.1%} | 平局 {probs[1]:.1%} | 客胜 {probs[2]:.1%}")
                        else:
                            st.write("**小利历史模型**：数据不足")
                        if sb_live_h and sb_live_a and sb_live_h>0:
                            probs = get_probs(sb_live_h, sb_live_a)
                            if probs:
                                st.write(f"**SB市场（即盘）**：主胜 {probs[0]:.1%} | 平局 {probs[1]:.1%} | 客胜 {probs[2]:.1%}")
                        else:
                            st.write("**SB市场（即盘）**：数据不足")
                        if xl_live_h and xl_live_a and xl_live_h>0:
                            probs = get_probs(xl_live_h, xl_live_a)
                            if probs:
                                st.write(f"**小利市场（即盘）**：主胜 {probs[0]:.1%} | 平局 {probs[1]:.1%} | 客胜 {probs[2]:.1%}")
                        else:
                            st.write("**小利市场（即盘）**：数据不足")
                        if sb_home_lam and sb_away_lam:
                            hist_h, hist_a, hist_source = sb_home_lam, sb_away_lam, "SB历史"
                        elif xl_home_lam and xl_away_lam:
                            hist_h, hist_a, hist_source = xl_home_lam, xl_away_lam, "小利历史"
                        else:
                            hist_h, hist_a, hist_source = None, None, None
                        if hist_h and market_h>0:
                            probs_hist = get_probs(hist_h, hist_a)
                            probs_market = get_probs(market_h, market_a)
                            if probs_hist and probs_market:
                                diff_h, diff_d, diff_a = probs_hist[0]-probs_market[0], probs_hist[1]-probs_market[1], probs_hist[2]-probs_market[2]
                                st.markdown(f"**{hist_source} vs {source}市场**：")
                                if diff_h > 0.05:
                                    st.success(f"✅ 历史模型相比市场更看好主队（+{diff_h:.1%}）")
                                elif diff_a > 0.05:
                                    st.success(f"✅ 历史模型相比市场更看好客队（+{diff_a:.1%}）")
                                elif diff_d > 0.05:
                                    st.info(f"ℹ️ 历史模型相比市场更看好平局（+{diff_d:.1%}）")
                                else:
                                    st.info("历史模型与市场预期基本一致")
                        elif hist_h:
                            st.info("市场数据不足，无法对比")
                        else:
                            st.warning("历史数据不足，无法给出对比建议")
                        if market_h > 0:
                            st.caption(f"理论盘口（基于市场λ差值）：{theoretical_hc:+.2f} | 实际盘口：{actual_handicap:+.2f}")
                        st.caption("注：本预测仅基于球队近5场历史λ终值平均值，仅供参考。")
                with st.expander("📜 查看从赔率中提取的SB/小利λ值"):
                    st.write("**SB公司**")
                    st.write(f"初盘 λ: 主 {sb_init_h:.3f} | 客 {sb_init_a:.3f}" if sb_init_h else "未找到")
                    st.write(f"即盘 λ: 主 {sb_live_h:.3f} | 客 {sb_live_a:.3f}" if sb_live_h else "未找到")
                    st.write("**小利公司**")
                    st.write(f"初盘 λ: 主 {xl_init_h:.3f} | 客 {xl_init_a:.3f}" if xl_init_h else "未找到")
                    st.write(f"即盘 λ: 主 {xl_live_h:.3f} | 客 {xl_live_a:.3f}" if xl_live_h else "未找到")

# ========== TAB2 录入比赛 ==========
elif st.session_state.current_tab == "📝 录入比赛":
    with st.container():
        st.markdown("### 录入比赛结果（需记录SB/小利λ及盘口）")

        def check_exact_duplicate(current_vals):
            df = load_results()
            if df.empty:
                return None
            mask = (df['参考盘口'] == current_vals['ref_handicap']) & \
                   (df['实际盘口'] == current_vals['actual_handicap']) & \
                   (df['先进球方'] == current_vals['first_scorer'])
            for key in ['sb_h_init', 'sb_h_live', 'sb_a_init', 'sb_a_live',
                        'xl_h_init', 'xl_h_live', 'xl_a_init', 'xl_a_live']:
                col_map = {
                    'sb_h_init': 'sb_λ_主队初', 'sb_h_live': 'sb_λ_主队终',
                    'sb_a_init': 'sb_λ_客队初', 'sb_a_live': 'sb_λ_客队终',
                    'xl_h_init': 'xl_λ_主队初', 'xl_h_live': 'xl_λ_主队终',
                    'xl_a_init': 'xl_λ_客队初', 'xl_a_live': 'xl_λ_客队终'
                }
                hist_col = col_map[key]
                mask = mask & (df[hist_col].apply(lambda x: abs(x - current_vals[key]) < 0.001))
            matched = df[mask]
            if not matched.empty:
                return matched.iloc[0]
            return None

        sb = st.session_state.get('sb', {})
        xl = st.session_state.get('xl', {})
        sb_live_h = sb.get('live_h', 0.0); sb_live_a = sb.get('live_a', 0.0)
        xl_live_h = xl.get('live_h', 0.0); xl_live_a = xl.get('live_a', 0.0)
        sb_init_h = sb.get('init_h', 0.0); sb_init_a = sb.get('init_a', 0.0)
        xl_init_h = xl.get('init_h', 0.0); xl_init_a = xl.get('init_a', 0.0)

        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("日期", datetime.now())
            home_team = st.text_input("主队", value=st.session_state.tab2_home_team, key="tab2_home_input")
            home_team_clean = clean_team_name(home_team)
            st.session_state.tab2_home_team = home_team_clean
            home_score = st.number_input("主队进球", min_value=0, step=1, value=0)
            default_ref = st.session_state.get('tab2_reference_handicap', 0.0)
            if default_ref not in handicap_options:
                default_ref = 0.0
            ref_index = handicap_options.index(default_ref) if default_ref in handicap_options else handicap_options.index(0)
            ref_handicap = st.selectbox("参考盘口", handicap_options, index=ref_index)
            default_actual = st.session_state.tab2_actual_handicap
            if default_actual not in handicap_options:
                default_actual = 0.0
            actual_index = handicap_options.index(default_actual) if default_actual in handicap_options else handicap_options.index(0)
            actual_handicap = st.selectbox("实际盘口", handicap_options, index=actual_index, key="actual_hc_tab2")
            st.session_state.tab2_actual_handicap = actual_handicap
            first_scorer = st.selectbox("先进球方", ["", "主队", "客队", "无进球"])
            predict_result = st.text_input("预测结果（可选）", value=st.session_state.get('tab2_predict_result', ''))

            if st.button("🔍 检查当前数据是否已存在", key="check_duplicate_btn"):
                current_vals_check = {
                    'sb_h_init': sb_init_h, 'sb_h_live': sb_live_h,
                    'sb_a_init': sb_init_a, 'sb_a_live': sb_live_a,
                    'xl_h_init': xl_init_h, 'xl_h_live': xl_live_h,
                    'xl_a_init': xl_init_a, 'xl_a_live': xl_live_a,
                    'ref_handicap': ref_handicap, 'actual_handicap': actual_handicap,
                    'first_scorer': first_scorer
                }
                duplicate = check_exact_duplicate(current_vals_check)
                if duplicate is not None:
                    st.warning(f"⚠️ 当前数据与历史比赛完全重复：{duplicate['日期']} {duplicate['主队']} {duplicate['比分']} {duplicate['客队']}")
                else:
                    st.success("✅ 当前数据未在历史记录中完全匹配，可以保存。")

            if st.button("📚 从历史比赛推荐", key="recommend_btn"):
                current_lams = [sb_init_h, sb_live_h, sb_init_a, sb_live_a, xl_init_h, xl_live_h, xl_init_a, xl_live_a]
                if all(abs(v)<1e-6 for v in current_lams):
                    st.warning("请先填写 SB/小利 的 λ 值（可以从 TAB1 解析后自动填充），当前全部为 0，无法推荐。")
                else:
                    current_vals = {
                        'sb_h_init': sb_init_h, 'sb_h_live': sb_live_h, 'sb_a_init': sb_init_a, 'sb_a_live': sb_live_a,
                        'xl_h_init': xl_init_h, 'xl_h_live': xl_live_h, 'xl_a_init': xl_init_a, 'xl_a_live': xl_live_a,
                        'ref_handicap': ref_handicap, 'actual_handicap': actual_handicap, 'first_scorer': first_scorer
                    }
                    similar_df = find_similar_matches(current_vals, top_n=10)
                    if not similar_df.empty:
                        top_sim = similar_df.iloc[0]['similarity']
                        if top_sim > 0.99:
                            st.success(f"🔔 发现极相似历史比赛（相似度 {top_sim:.4f}），建议参考其赛果！")
                        elif top_sim > 0.8:
                            st.success(f"🔔 发现高相似历史比赛（相似度 {top_sim:.4f}），建议参考其赛果！")
                        elif top_sim > 0.6:
                            st.info(f"中等相似度 {top_sim:.4f}，可参考。")
                        else:
                            st.info(f"相似度较低（{top_sim:.4f}），仅供参考。")
                        st.session_state['similar_matches'] = similar_df
                        st.session_state['similar_index'] = 0
                        with st.expander("🔧 调试信息（相似度详情）"):
                            st.write(f"**当前参考盘口**: {ref_handicap:.2f} | **实际盘口**: {actual_handicap:.2f}")
                            st.write(f"**匹配的历史比赛数量**: {len(similar_df)}")
                            st.write("**当前表单 λ 值**")
                            st.json({k: current_vals.get(k, 0) for k in ['sb_h_live', 'sb_a_live', 'xl_h_live', 'xl_a_live']})
                            st.write("**历史比赛相似度列表（前5条）**")
                            debug_df = similar_df[['日期', '主队', '客队', '比分', 'similarity']].head()
                            st.dataframe(debug_df)
                        st.rerun()

            if 'similar_matches' in st.session_state and not st.session_state.similar_matches.empty:
                similar_df = st.session_state.similar_matches
                idx = st.session_state.get('similar_index', 0)
                if idx < len(similar_df):
                    rec = similar_df.iloc[idx]
                    st.markdown("---")
                    st.markdown(f"**相似比赛 {idx+1}/{len(similar_df)}** (相似度: {rec['similarity']:.4f})")
                    col_hist, col_curr = st.columns(2)
                    with col_hist:
                        st.markdown("**📜 历史比赛数据**")
                        st.write(f"📅 {rec['日期']} : {rec['主队']} {rec['比分']} {rec['客队']}")
                        st.write(f"参考盘口: {rec['参考盘口']:.2f} | 实际盘口: {rec['实际盘口']:.2f} | 先进球方: {rec['先进球方']}")
                        st.write(f"SB λ: 主初 {rec['sb_λ_主队初']:.3f} 主终 {rec['sb_λ_主队终']:.3f} | 客初 {rec['sb_λ_客队初']:.3f} 客终 {rec['sb_λ_客队终']:.3f}")
                        st.write(f"小利 λ: 主初 {rec['xl_λ_主队初']:.3f} 主终 {rec['xl_λ_主队终']:.3f} | 客初 {rec['xl_λ_客队初']:.3f} 客终 {rec['xl_λ_客队终']:.3f}")
                    with col_curr:
                        st.markdown("**📝 当前比赛数据**")
                        curr_home = st.session_state.tab2_home_team
                        curr_away = st.session_state.tab2_away_team
                        curr_ref = st.session_state.get('tab2_reference_handicap', ref_handicap)
                        curr_actual = st.session_state.tab2_actual_handicap
                        curr_first = first_scorer
                        st.write(f"{curr_home} vs {curr_away}")
                        st.write(f"参考盘口: {curr_ref:.2f} | 实际盘口: {curr_actual:.2f} | 先进球方: {curr_first}")
                        st.write(f"SB λ: 主初 {sb_init_h:.3f} 主终 {sb_live_h:.3f} | 客初 {sb_init_a:.3f} 客终 {sb_live_a:.3f}")
                        st.write(f"小利 λ: 主初 {xl_init_h:.3f} 主终 {xl_live_h:.3f} | 客初 {xl_init_a:.3f} 客终 {xl_live_a:.3f}")
                    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
                    with col_btn1:
                        if idx > 0:
                            if st.button("◀ 上一个"):
                                st.session_state.similar_index = idx - 1
                                st.rerun()
                    with col_btn2:
                        if idx + 1 < len(similar_df):
                            if st.button("下一个 ▶"):
                                st.session_state.similar_index = idx + 1
                                st.rerun()
                    with col_btn3:
                        if st.button("📥 使用此推荐填充表单"):
                            st.session_state['sb'] = {
                                'init_h': rec['sb_λ_主队初'], 'init_a': rec['sb_λ_客队初'],
                                'live_h': rec['sb_λ_主队终'], 'live_a': rec['sb_λ_客队终']
                            }
                            st.session_state['xl'] = {
                                'init_h': rec['xl_λ_主队初'], 'init_a': rec['xl_λ_客队初'],
                                'live_h': rec['xl_λ_主队终'], 'live_a': rec['xl_λ_客队终']
                            }
                            st.session_state.tab2_reference_handicap = rec['参考盘口']
                            st.session_state.tab2_actual_handicap = rec['实际盘口']
                            st.session_state.tab2_predict_result = rec.get('预测结果', '')
                            st.session_state.tab2_home_team = rec['主队']
                            st.session_state.tab2_away_team = rec['客队']
                            st.success("已填充表单，请检查并修改后保存")
                            st.rerun()
        with col2:
            away_team = st.text_input("客队", value=st.session_state.tab2_away_team, key="tab2_away_input")
            away_team_clean = clean_team_name(away_team)
            st.session_state.tab2_away_team = away_team_clean
            away_score = st.number_input("客队进球", min_value=0, step=1, value=0)
            st.markdown("**SB公司λ**")
            sb_h_init = st.number_input("SB λ主队初", value=float(sb_init_h), step=0.001, format="%.3f")
            sb_h_live = st.number_input("SB λ主队终", value=float(sb_live_h), step=0.001, format="%.3f")
            sb_a_init = st.number_input("SB λ客队初", value=float(sb_init_a), step=0.001, format="%.3f")
            sb_a_live = st.number_input("SB λ客队终", value=float(sb_live_a), step=0.001, format="%.3f")
            st.markdown("**小利公司λ**")
            xl_h_init = st.number_input("小利 λ主队初", value=float(xl_init_h), step=0.001, format="%.3f")
            xl_h_live = st.number_input("小利 λ主队终", value=float(xl_live_h), step=0.001, format="%.3f")
            xl_a_init = st.number_input("小利 λ客队初", value=float(xl_init_a), step=0.001, format="%.3f")
            xl_a_live = st.number_input("小利 λ客队终", value=float(xl_live_a), step=0.001, format="%.3f")

        if st.button("💾 保存比赛", key="save_btn"):
            if not home_team_clean or not away_team_clean:
                st.error("请填写主队和客队名称")
            else:
                current_vals_check = {
                    'sb_h_init': sb_h_init, 'sb_h_live': sb_h_live,
                    'sb_a_init': sb_a_init, 'sb_a_live': sb_a_live,
                    'xl_h_init': xl_h_init, 'xl_h_live': xl_h_live,
                    'xl_a_init': xl_a_init, 'xl_a_live': xl_a_live,
                    'ref_handicap': ref_handicap, 'actual_handicap': actual_handicap,
                    'first_scorer': first_scorer
                }
                duplicate = check_exact_duplicate(current_vals_check)
                if duplicate is not None:
                    st.error(f"❌ 数据与历史比赛完全重复（{duplicate['日期']} {duplicate['主队']} {duplicate['比分']} {duplicate['客队']}），禁止保存。")
                else:
                    fingerprint = st.session_state.get('current_fingerprint', '')
                    result = save_result(
                        date.strftime("%Y-%m-%d"), home_team_clean, away_team_clean, home_score, away_score,
                        ref_handicap, actual_handicap, predict_result, first_scorer,
                        sb_h_init, sb_h_live, sb_a_init, sb_a_live,
                        xl_h_init, xl_h_live, xl_a_init, xl_a_live,
                        fingerprint
                    )
                    if result is not None:
                        st.success("✅ 比赛结果保存成功！")
                        st.session_state.current_tab = "📈 赔率分析"
                        time.sleep(0.5)
                        st.rerun()

# ========== TAB3 历史记录 ==========
elif st.session_state.current_tab == "📊 历史记录":
    with st.container():
        st.markdown("### 历史记录")
        st.subheader("💾 数据库备份与恢复")
        col_backup1, col_backup2 = st.columns(2)
        with col_backup1:
            if st.button("📀 备份当前数据库"):
                success, msg = backup_database()
                if success:
                    st.success(f"备份成功！文件名：{msg}")
                else:
                    st.error(f"备份失败：{msg}")
        with col_backup2:
            backup_files = get_backup_files()
            if backup_files:
                selected_backup = st.selectbox("选择备份文件", backup_files)
                if st.button("🔄 恢复所选备份"):
                    success, msg = restore_database(selected_backup)
                    if success:
                        st.success(msg)
                        st.info("请点击页面右上角的 'Rerun' 或重新运行应用以刷新数据。")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"恢复失败：{msg}")
            else:
                st.info("暂无备份文件。")
        st.markdown("---")
        dfh = load_results()
        if dfh.empty:
            st.info("暂无记录")
        else:
            display_cols = ['日期', '主队', '客队', '比分', '主队进球', '客队进球', '参考盘口', '实际盘口', '先进球方']
            st.dataframe(dfh[display_cols], use_container_width=True)
            csv = dfh.to_csv(index=False).encode('utf-8-sig')
            st.download_button("导出当前数据为 CSV", csv, "match_results.csv", "text/csv")

# ========== TAB4 自动获取比赛 ==========
elif st.session_state.current_tab == "🆕 自动获取比赛":
    with st.container():
        st.markdown("### 从 Football-Data.org 自动获取比赛")
        st.info("免费版仅支持2021、2022、2023赛季。")
        comp_dict = {"英超": "PL", "西甲": "PD", "德甲": "BL1", "意甲": "SA", "法甲": "FL1"}
        comp_name = st.selectbox("联赛", list(comp_dict.keys()))
        competition_code = comp_dict[comp_name]
        season = st.text_input("赛季", value="2023")
        matchday = st.number_input("轮次", min_value=1, step=1, value=1)
        if st.button("📥 获取比赛并预测"):
            with st.spinner("请求中..."):
                data = fetch_matches_with_retry(competition_code, season, matchday)
            if data is None:
                st.error("获取失败")
            else:
                matches = data.get('matches', [])
                if not matches:
                    st.warning("该轮次无比赛")
                else:
                    matches_list = []
                    for m in matches:
                        home = m['homeTeam']['name']
                        away = m['awayTeam']['name']
                        status = m['status']
                        home_score = m['score']['fullTime']['home']
                        away_score = m['score']['fullTime']['away']
                        result = f"{home_score}:{away_score}" if status == 'FINISHED' else "未开始"
                        matches_list.append({'日期': m['utcDate'], '主队': home, '客队': away, '状态': status, '比分': result})
                    df_matches = pd.DataFrame(matches_list)
                    st.dataframe(df_matches)
                    st.subheader("预测（基于SB历史λ）")
                    ignore_home_away = st.session_state.get('ignore_home_away', False)
                    for _, row in df_matches.iterrows():
                        home, away, status, result = row['主队'], row['客队'], row['状态'], row['比分']
                        if status == 'FINISHED':
                            st.write(f"{home} vs {away}: {result} (已结束)")
                        else:
                            probs, ou, score, _ = simple_predict_by_team_strength(home, away, ignore_home_away)
                            if probs is None:
                                st.write(f"{home} vs {away}: 历史数据不足")
                            else:
                                st.write(f"{home} vs {away}：预测 {score}，{ou}")