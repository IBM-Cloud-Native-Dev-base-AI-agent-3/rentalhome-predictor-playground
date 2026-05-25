import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
api_key = os.getenv("GEMINI_API_KEY")

class CutlineRow(BaseModel):
    complex_name: str = Field(description="단지명 (Name of the apartment/complex)")
    district: str = Field(description="단지가 속한 서울시 자치구 (예: 강남구, 강서구, 마포구, 송파구 등. 단지명을 보고 유추하거나 원래 자치구명을 기입하세요.)")
    housing_type: str = Field(description="주택형/공급유형 (Housing area in m²)")
    supply_target: str = Field(description="공급대상/신청자격 (e.g. 대학생, 청년, 신혼부부, 고령자, 주거급여수급자 등)")
    supply_type: str = Field(description="구분 ('우선' 또는 '일반')")
    cutline_rank: Optional[int] = Field(None, description="최저 커트라인 순위 (1, 2, 3). 순위 기준이 없거나 전원 합격/추첨 시 null")
    cutline_score: Optional[int] = Field(None, description="최저 커트라인 가점. 가점 기준이 없거나 일반공급 등 가점이 해당 안 되면 null")
    tie_breaker: Optional[str] = Field(None, description="동점자 처리기준 또는 커트라인 상세 조건 (예: '추첨', '전입일자', '거주기간', '자녀수', '전원합격' 등)")
    tie_breaker_value: Optional[str] = Field(None, description="동점자 처리기준 날짜나 기간 (예: '2013.12.31', '36개월' 등 날짜/기간값만 추출). 없으면 null")
    raw_text: str = Field(description="커트라인 원본 텍스트 전체")

class CutlinePage(BaseModel):
    rows: List[CutlineRow]

def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.png', filename)
    return int(match.group(1)) if match else 9999

def main():
    cutline_dir = "/Users/jc.kim/Desktop/rentalhome-predictor-playground/rentalhome_cutline"
    output_path = os.path.join(os.path.dirname(__file__), "parsed_cutlines.json")
    
    # Load existing progress if available
    progress_data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                print(f"Loaded existing progress. {len(progress_data)} pages already parsed.")
        except Exception as e:
            print("Error loading existing output, starting fresh:", e)
            
    # List and sort all PNG files in directory
    files = [f for f in os.listdir(cutline_dir) if f.endswith(".png")]
    files.sort(key=extract_page_number)
    
    print(f"Found {len(files)} image files to process.")
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    prompt = """
    이 이미지는 SH 행복주택 서류심사대상자 커트라인 표입니다.
    표에 있는 모든 행(Row)을 빠짐없이 분석해서 지정된 JSON 스키마 형식으로 변환해주세요.
    
    [규칙]
    1. 단지명, 주택형, 공급대상, 구분은 표의 해당 칸 값을 그대로 넣으세요.
    2. 단지가 속한 서울시 자치구(district)를 유추하거나 조사하여 채워주세요 (예: 가양동 -> 강서구, 디에이치포레센트/르엘대치 -> 강남구, 보라매자이 -> 동작구, 송파시그니처롯데캐슬 -> 송파구).
    3. 커트라인 컬럼이 분리되어 있거나 합쳐져 있는 경우:
       - 순위(1, 2, 3), 가점(점수 숫자), 동점자 기준(추첨/전입일자/거주기간 등)을 분리하여 각 필드에 맞게 채워주세요.
       - 예: '1순위 6점(전입일자 : 2013.12.31)' -> cutline_rank=1, cutline_score=6, tie_breaker='전입일자', tie_breaker_value='2013.12.31'
       - 예: '1순위 중 추첨' -> cutline_rank=1, cutline_score=None, tie_breaker='추첨', tie_breaker_value=None
       - 예: '추첨' -> cutline_rank=None, cutline_score=None, tie_breaker='추첨', tie_breaker_value=None
       - 예: '전원' -> cutline_rank=None, cutline_score=None, tie_breaker='전원합격', tie_breaker_value=None
    """
    
    newly_parsed = 0
    
    for idx, filename in enumerate(files):
        # Skip if already parsed
        if filename in progress_data:
            continue
            
        filepath = os.path.join(cutline_dir, filename)
        print(f"[{idx+1}/{len(files)}] Processing {filename}...")
        
        retries = 3
        success = False
        
        while retries > 0 and not success:
            try:
                img = Image.open(filepath)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CutlinePage,
                        temperature=0.0
                    )
                )
                
                # Parse JSON string from response
                page_data = json.loads(response.text)
                progress_data[filename] = page_data["rows"]
                
                # Save progress immediately
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
                
                success = True
                newly_parsed += 1
                print(f"Successfully parsed {filename}. Extracted {len(page_data['rows'])} rows.")
                
            except Exception as e:
                retries -= 1
                print(f"Error parsing {filename} (Retries left: {retries}): {e}")
                if retries > 0:
                    time.sleep(5)
                else:
                    print(f"Skipping {filename} due to repeated errors.")
        
        # Add a sleep to be friendly to the API rate limit
        time.sleep(1.5)
        
    print(f"Batch processing completed. Total parsed pages: {len(progress_data)}. Newly parsed: {newly_parsed}")

if __name__ == "__main__":
    main()
