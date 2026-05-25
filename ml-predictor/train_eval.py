import os
import re
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier

# List of Seoul Districts (자치구)
SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
]

# Seoul Adjacent Regions (연접지역)
SEOUL_ADJACENT = [
    "의정부시", "남양주시", "구리시", "하남시", "성남시", "과천시", "안양시", "광명시",
    "부천시", "인천광역시", "김포시", "고양시", "양주시"
]

# Gyeonggi Except Adjacent (경기 기타)
GYEONGGI_OTHER = [
    "수원시", "평택시", "동두천시", "안산시", "오산시", "시흥시", "군포시", "의왕시",
    "용인시", "파주시", "이천시", "안성시", "화성시", "광주시", "포천시", "여주시",
    "연천군", "가평군", "양평군"
]

def parse_date(date_str):
    try:
        clean_date = date_str.replace(" ", "").replace("\n", "")
        parts = clean_date.split(".")
        if len(parts) >= 3:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2][:2]))
    except Exception:
        pass
    return None

def eval_tie_breaker(tie_breaker, tb_date, tb_months, cand_date, cand_months):
    if tie_breaker == "추첨" or not tie_breaker:
        return random.choice([0, 1])
    elif tie_breaker == "전원합격" or tie_breaker == "전원":
        return 1
    elif tie_breaker == "전입일자" or tie_breaker == "자치구 전입일":
        if tb_date:
            return 1 if cand_date <= tb_date else 0
        else:
            return random.choice([0, 1])
    elif "거주" in tie_breaker or "기간" in tie_breaker:
        if tb_months:
            return 1 if cand_months >= tb_months else 0
        else:
            return random.choice([0, 1])
    else:
        return random.choice([0, 1])

def generate_applicant_profile(category, target_district):
    # Generates a random applicant with raw attributes
    # residence is where they live (Seoul district, adjacent, Gyeonggi other, or other province)
    residence_pool = SEOUL_DISTRICTS + SEOUL_ADJACENT + GYEONGGI_OTHER + ["부산광역시", "대구광역시", "광주광역시"]
    # Weights: 60% Seoul districts, 20% adjacent, 10% Gyeonggi other, 10% outside
    weights = [0.6 / len(SEOUL_DISTRICTS)] * len(SEOUL_DISTRICTS) + \
              [0.2 / len(SEOUL_ADJACENT)] * len(SEOUL_ADJACENT) + \
              [0.1 / len(GYEONGGI_OTHER)] * len(GYEONGGI_OTHER) + \
              [0.1 / 3] * 3
              
    residence = random.choices(residence_pool, weights=weights)[0]
    
    is_home_owner = random.choices([False, True], weights=[0.95, 0.05])[0] # Mostly homeless
    has_past_home_ownership = random.choice([False, True])
    past_contract_history = random.choices(["없음", "1회", "2회 이상"], weights=[0.9, 0.08, 0.02])[0]
    
    subscription_count = random.randint(0, 48)
    
    # Category specific attributes
    if category == "students":
        age = random.randint(18, 28)
        marriage_duration_years = 0.0
        income_percent = random.randint(40, 140)
        total_asset = random.randint(5_000_000, 150_000_000)
        car_value = 0 if random.random() < 0.95 else random.randint(5_000_000, 25_000_000)
        special_qualifications = []
        residence_duration_months = random.randint(0, 120)
        parents_outside_seoul = random.choice([True, False])
        
    elif category == "youths":
        age = random.randint(19, 39)
        marriage_duration_years = 0.0
        income_percent = random.randint(50, 150)
        total_asset = random.randint(10_000_000, 350_000_000)
        car_value = 0 if random.random() < 0.4 else random.randint(5_000_000, 45_000_000)
        special_qualifications = []
        residence_duration_months = random.randint(0, 240)
        parents_outside_seoul = False
        
    elif category == "newlyweds":
        age = random.randint(24, 45)
        marriage_duration_years = round(random.uniform(0.0, 9.0), 1)
        income_percent = random.randint(60, 170)
        total_asset = random.randint(30_000_000, 450_000_000)
        car_value = 0 if random.random() < 0.2 else random.randint(5_000_000, 50_000_000)
        special_qualifications = []
        residence_duration_months = random.randint(0, 240)
        parents_outside_seoul = False
        
    elif category == "elderly":
        age = random.randint(65, 88)
        marriage_duration_years = 40.0
        income_percent = random.randint(30, 130)
        total_asset = random.randint(20_000_000, 450_000_000)
        car_value = 0 if random.random() < 0.6 else random.randint(5_000_000, 45_000_000)
        special_qualifications = random.choices([[], ["장애인"], ["국가유공자"]], weights=[0.85, 0.1, 0.05])[0]
        residence_duration_months = random.randint(0, 480)
        parents_outside_seoul = False
        
    else: # welfare (주거급여수급자)
        age = random.randint(19, 80)
        marriage_duration_years = random.choice([0.0, 15.0])
        income_percent = random.randint(10, 50) # Mostly very low income
        total_asset = random.randint(2_000_000, 150_000_000)
        car_value = 0 if random.random() < 0.8 else random.randint(2_000_000, 25_000_000)
        special_qualifications = random.choices([[], ["장애인"], ["한부모가족"]], weights=[0.7, 0.2, 0.1])[0]
        residence_duration_months = random.randint(0, 360)
        parents_outside_seoul = False
        
    return {
        "is_home_owner": is_home_owner,
        "has_past_home_ownership": has_past_home_ownership,
        "age": age,
        "residence": residence,
        "marriage_duration_years": marriage_duration_years,
        "income_percent": income_percent,
        "total_asset": total_asset,
        "car_value": car_value,
        "subscription_count": subscription_count,
        "special_qualifications": special_qualifications,
        "past_contract_history": past_contract_history,
        "residence_duration_months": residence_duration_months,
        "parents_outside_seoul": parents_outside_seoul
    }

def evaluate_eligibility_and_score(profile, category, supply_type, target_district):
    # 1. Eligibility Check (무주택 및 소득/자산 기준 검증)
    if profile["is_home_owner"]:
        return False, 3, 0 # Disqualified: not homeless
        
    # Check income / asset / car limits depending on category
    income = profile["income_percent"]
    asset = profile["total_asset"]
    car = profile["car_value"]
    
    if category == "students":
        if income > 100 or asset > 104_000_000 or car > 0:
            return False, 3, 0
    elif category == "youths":
        if income > 100 or asset > 254_000_000 or car > 38_030_000:
            return False, 3, 0
    elif category == "newlyweds":
        if income > 100 or asset > 337_000_000 or car > 38_030_000:
            return False, 3, 0
    elif category == "elderly":
        if income > 100 or asset > 337_000_000 or car > 38_030_000:
            return False, 3, 0
            
    # 2. Rank Calculation (순위 결정)
    res = profile["residence"]
    
    if supply_type == "우선":
        # 우선공급 순위: 해당 자치구 거주지 = 1순위, 타 서울 자치구 거주지 = 2순위
        if res == target_district:
            rank = 1
        elif res in SEOUL_DISTRICTS:
            rank = 2
        else:
            rank = 3 # Outside Seoul fails priority rank
    else: # 일반공급
        # 일반공급 순위: 서울시 및 연접지역 = 1순위, 경기 기타 = 2순위, 그 외 = 3순위
        if res in SEOUL_DISTRICTS or res in SEOUL_ADJACENT:
            rank = 1
        elif res in GYEONGGI_OTHER:
            rank = 2
        else:
            rank = 3
            
    # 3. Score Calculation (가점 계산)
    score = 0
    sub_count = profile["subscription_count"]
    res_dur = profile["residence_duration_months"]
    
    if category == "students":
        # 부모 서울외 거주 = 3점, 서울내 타자치구 거주 = 1점
        score += 3 if profile["parents_outside_seoul"] else 1
        # 청약통장: 24회 이상 = 3점, 6~23회 = 1점
        if sub_count >= 24:
            score += 3
        elif sub_count >= 6:
            score += 1
            
    elif category in ["youths", "newlyweds"]:
        # 서울시 거주기간: 3년(36개월) 이상 = 3점, 3년 미만 = 1점
        score += 3 if res_dur >= 36 else 1
        # 청약통장: 24회 이상 = 3점, 6~23회 = 1점
        if sub_count >= 24:
            score += 3
        elif sub_count >= 6:
            score += 1
            
    elif category == "elderly":
        # 나이: 만 75세 이상 = 3점, 70~74 = 2점, 65~69 = 1점
        age = profile["age"]
        if age >= 75:
            score += 3
        elif age >= 70:
            score += 2
        else:
            score += 1
            
        # 해당 자치구 거주기간: 5년(60개월) 이상 = 3점, 5년 미만 = 2점, 타 서울자치구 = 1점
        if res == target_district:
            score += 3 if res_dur >= 60 else 2
        elif res in SEOUL_DISTRICTS:
            score += 1
            
        # 취약계층(장애인, 국가유공자 등): 해당 = 3점, 해당없음 = 1점
        score += 3 if len(profile["special_qualifications"]) > 0 else 1
        
    elif category == "welfare":
        # 해당 자치구 거주기간: 5년(60개월) 이상 = 3점, 5년 미만 = 2점, 타 서울자치구 = 1점
        if res == target_district:
            score += 3 if res_dur >= 60 else 2
        elif res in SEOUL_DISTRICTS:
            score += 1
            
        # 취약계층: 해당 = 3점, 해당없음 = 1점
        score += 3 if len(profile["special_qualifications"]) > 0 else 1
        
    return True, rank, score

def generate_dataset_from_cutlines(parsed_cutlines_path, num_samples_per_row=30):
    with open(parsed_cutlines_path, "r", encoding="utf-8") as f:
        cutlines_data = json.load(f)
        
    dataset = []
    base_date = datetime(2000, 1, 1)
    
    for page_name, rows in cutlines_data.items():
        for row in rows:
            complex_name = row.get("complex_name", "")
            district = row.get("district", "")
            housing_type_str = row.get("housing_type", "39")
            try:
                housing_type = float(housing_type_str)
            except ValueError:
                housing_type = 39.0
                
            supply_target = row.get("supply_target", "")
            supply_type = row.get("supply_type", "일반")
            
            category = "youths"
            if "대학생" in supply_target:
                category = "students"
            elif "청년" in supply_target:
                category = "youths"
            elif "신혼부부" in supply_target or "한부모" in supply_target:
                category = "newlyweds"
            elif "고령자" in supply_target:
                category = "elderly"
            elif "주거급여" in supply_target:
                category = "welfare"
                
            cutline_rank = row.get("cutline_rank")
            cutline_score = row.get("cutline_score")
            tie_breaker = row.get("tie_breaker") or "추첨"
            tie_breaker_value = row.get("tie_breaker_value")
            
            c_rank = int(cutline_rank) if cutline_rank is not None else 1
            c_score = int(cutline_score) if cutline_score is not None else 0
            
            tb_date = None
            tb_months = None
            if tie_breaker == "전입일자" or tie_breaker == "자치구 전입일":
                if tie_breaker_value:
                    tb_date = parse_date(tie_breaker_value)
            elif "거주" in tie_breaker or "기간" in tie_breaker:
                if tie_breaker_value:
                    try:
                        tb_months = int(re.search(r'\d+', tie_breaker_value).group())
                    except Exception:
                        tb_months = 36
            
            # Generate multiple raw candidate profiles
            for _ in range(num_samples_per_row):
                profile = generate_applicant_profile(category, district)
                
                # Check eligibility, rank, score based on calculated rules
                eligible, cand_rank, cand_score = evaluate_eligibility_and_score(
                    profile, category, supply_type, district
                )
                
                # Randomly generate candidate's date & months for tie-breakers
                random_days = random.randint(0, 9000)
                cand_date = base_date + timedelta(days=random_days)
                cand_date_days = (cand_date - base_date).days
                cand_months = profile["residence_duration_months"]
                
                passed = 0
                if eligible:
                    # Compare candidate calculated rank/score with the target complex cutline
                    if cand_rank < c_rank:
                        passed = 1
                    elif cand_rank > c_rank:
                        passed = 0
                    else: # ranks match
                        if cutline_score is not None:
                            if cand_score > c_score:
                                passed = 1
                            elif cand_score < c_score:
                                passed = 0
                            else: # scores match, apply tie-breaker
                                passed = eval_tie_breaker(tie_breaker, tb_date, tb_months, cand_date, cand_months)
                        else:
                            passed = eval_tie_breaker(tie_breaker, tb_date, tb_months, cand_date, cand_months)
                else:
                    passed = 0 # Disqualified
                
                # Build row of raw applicant details as features
                row_dict = {
                    "is_home_owner": 1 if profile["is_home_owner"] else 0,
                    "has_past_home_ownership": 1 if profile["has_past_home_ownership"] else 0,
                    "age": profile["age"],
                    "residence": profile["residence"],
                    "marriage_duration_years": profile["marriage_duration_years"],
                    "income_percent": profile["income_percent"],
                    "total_asset": profile["total_asset"],
                    "car_value": profile["car_value"],
                    "subscription_count": profile["subscription_count"],
                    "is_vulnerable": 1 if len(profile["special_qualifications"]) > 0 else 0,
                    "past_contract_history": profile["past_contract_history"],
                    "residence_duration_months": profile["residence_duration_months"],
                    
                    # Target housing details (also input features)
                    "target_complex": complex_name,
                    "target_district": district,
                    "target_housing_type": housing_type,
                    "target_supply_target": category,
                    "target_supply_type": supply_type,
                    
                    # Target label
                    "passed": passed
                }
                dataset.append(row_dict)
                
    return pd.DataFrame(dataset)

def main():
    parsed_path = os.path.join(os.path.dirname(__file__), "parsed_cutlines.json")
    if not os.path.exists(parsed_path):
        print("Error: parsed_cutlines.json not found.")
        return
        
    print("Generating synthetic candidate dataset from RAW applicant features...")
    df = generate_dataset_from_cutlines(parsed_path, num_samples_per_row=40)
    print(f"Generated dataset with {df.shape[0]} samples.")
    print("Columns:", df.columns.tolist())
    print("Label distribution (Passed vs Failed):\n", df['passed'].value_counts())
    
    # Feature engineering: Encode categorical features
    cat_cols = ["residence", "past_contract_history", "target_complex", "target_district", "target_supply_target", "target_supply_type"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    X = df.drop(columns=["passed"])
    y = df["passed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    }
    
    results = []
    
    print("\nTraining and evaluating models on RAW applicant features...")
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc
            })
            print(f"[{name}] Done. Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error evaluating model {name}: {e}")
            
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))
    
    md_output_path = os.path.join(os.path.dirname(__file__), "model_evaluation_results.md")
    
    markdown_content = "# 머신러닝 모델 예측 성능 비교 (RAW 신청자 속성 기반)\n\n"
    markdown_content += "이 평가는 가치자산, 나이, 소득분위, 거주지, 청약저축 횟수 등 **신청자 고유의 실제 속성(Raw Attributes)**을 입력값(Features)으로 하고, "
    markdown_content += "역사적 커트라인 데이터에 의해 계산된 **당첨 여부(Passed)**를 예측 대상으로 삼아 훈련된 결과입니다. (총 {}개 샘플)\n\n".format(df.shape[0])
    markdown_content += "| 모델 (Model) | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | F1-Score | ROC-AUC |\n"
    markdown_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    for r in results:
        markdown_content += "| **{}** | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            r["Model"], r["Accuracy"], r["Precision"], r["Recall"], r["F1-Score"], r["ROC-AUC"]
        )
        
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"\nResults saved to {md_output_path}")

if __name__ == "__main__":
    main()
