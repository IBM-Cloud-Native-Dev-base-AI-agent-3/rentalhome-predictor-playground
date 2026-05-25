import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
api_key = os.getenv("GEMINI_API_KEY")

class RankRule(BaseModel):
    rank_1: str = Field(description="1순위 조건")
    rank_2: str = Field(description="2순위 조건")
    rank_3: str = Field(description="3순위 조건")

class PriorityRankRule(BaseModel):
    rank_1: str = Field(description="우선공급 1순위 조건")
    rank_2: str = Field(description="우선공급 2순위 조건")

class ScoreCriteria(BaseModel):
    points_3: str = Field(description="3점 기준")
    points_2: Optional[str] = Field(None, description="2점 기준 (해당 사항 없을 경우 null)")
    points_1: str = Field(description="1점 기준")

class TargetRules(BaseModel):
    income_limit: str = Field(description="소득 제한 요건")
    asset_limit: str = Field(description="자산 제한 요건")
    car_limit: str = Field(description="자동차 가액 제한 요건")
    general_rank: RankRule = Field(description="일반공급 순위 기준")
    priority_rank: PriorityRankRule = Field(description="우선공급 순위 기준")
    score_residence_duration: ScoreCriteria = Field(description="거주기간 가점 기준")
    score_subscription_or_other: ScoreCriteria = Field(description="청약통장 납입 횟수 또는 고령자 나이 등 가점 기준")
    score_additional: Optional[ScoreCriteria] = Field(None, description="추가 가점 기준 (예: 고령자 취약계층 등, 없으면 null)")

class HousingRules(BaseModel):
    students: TargetRules = Field(description="대학생 계층 규칙")
    youths: TargetRules = Field(description="청년 계층 규칙")
    newlyweds: TargetRules = Field(description="신혼부부/한부모가족 계층 규칙")
    elderly: TargetRules = Field(description="고령자 계층 규칙")
    welfare: TargetRules = Field(description="주거급여수급자 계층 규칙")

def main():
    pdf_path = "/Users/jc.kim/Desktop/rentalhome-predictor-playground/rentalhome_rule/rule.pdf"
    output_path = os.path.join(os.path.dirname(__file__), "rules_schema.json")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return
        
    client = genai.Client(api_key=api_key)
    
    print("Uploading PDF to Gemini...")
    pdf_file = client.files.upload(file=pdf_path)
    print(f"Uploaded file name: {pdf_file.name}")
    
    # Wait for processing
    while pdf_file.state.name == "PROCESSING":
        print("Processing PDF...")
        time.sleep(2)
        pdf_file = client.files.get(name=pdf_file.name)
        
    if pdf_file.state.name == "FAILED":
        print("PDF upload and processing failed.")
        return
        
    print("PDF ready. Extracting rules using structured output...")
    
    prompt = """
    이 PDF는 SH 서울주택도시공사의 행복주택 입주자 모집 공고문입니다.
    공고문 전체를 꼼꼼히 분석하여 다음 5가지 계층별 청약 규칙(소득/자산 기준, 일반/우선 순위 기준, 우선공급 가점 배점)을 JSON 스키마 형식으로 추출해주세요:
    
    1. 대학생 계층 (students)
    2. 청년 계층 (youths)
    3. 신혼부부·한부모가족 계층 (newlyweds)
    4. 고령자 계층 (elderly)
    5. 주거급여수급자 계층 (welfare)
    
    각 계층별 가점 기준(score_residence_duration, score_subscription_or_other, score_additional)의 배점을 정확히 매칭해주세요.
    예를 들어 대학생은 서울시 거주기간(최대 3점)과 청약통장 납입 횟수(최대 3점)가 배점입니다.
    고령자는 나이(최대 3점), 거주기간(최대 3점), 취약계층(최대 3점)이 배점입니다.
    배점 기준 텍스트를 정확하게 작성해 주세요.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pdf_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HousingRules,
                temperature=0.0
            )
        )
        
        # Save output
        data = json.loads(response.text)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print("Successfully extracted rules to rules_schema.json!")
        
    except Exception as e:
        print("Error during rules extraction:", e)
    finally:
        print("Cleaning up file...")
        client.files.delete(name=pdf_file.name)
        print("Done.")

if __name__ == "__main__":
    main()
