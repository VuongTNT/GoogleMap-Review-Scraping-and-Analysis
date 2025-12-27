import os
import subprocess
import random
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List
from collections import Counter
import math
import re
import requests
import json
import ipynbname

class Scraper:
    def __init__(self, data_dir: str = "data/demo"):
        self.DATA_DIR = data_dir
        self.LOC_DIRS = os.path.join(self.DATA_DIR, "locations")
        self.RESPONSES_DIR = os.path.join(self.DATA_DIR, "responses")
        
        if not os.path.exists(self.RESPONSES_DIR):
            os.makedirs(self.RESPONSES_DIR)
        if not os.path.exists(self.LOC_DIRS):
            os.makedirs(self.LOC_DIRS)

    def scrape_loc_id(self, 
        loc_scraper_dir: str,
        loops: int = 20,
        geo_bounds: Dict[str, Dict[str, tuple]] = None,
        categories: List[str] = None,
        search_radius: int = 500000,
        filter_min_reviews: int = 10    #reject locs with few reviews
        ):
        
        #set up categories
        if not categories:
            categories = [
                "restaurant",
                "bar",
                "cafe",
                "fastfood",
                "food",
                "beverage",
                "noodles"
            ]
        loc_ids_scrapper_input = f"{self.LOC_DIRS}/input.txt"
        with open(loc_ids_scrapper_input, "w", encoding="utf-8") as f:
            for category in categories:
                f.write(f"{category}\n")
        
        #use scraper to get location ids
        for _ in range(loops):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            long, lat, country_name = self.get_geo(geo_bounds)
            loc_ids_scrapper_output = f"{self.LOC_DIRS}/locs-{timestamp}-{country_name}.csv"
            geo = f"{str(long)},{str(lat)}"
            
            cmd = [
                loc_scraper_dir,
                "-input", loc_ids_scrapper_input,
                "-results", loc_ids_scrapper_output,
                "-exit-on-inactivity", "1m",
                "-fast-mode",
                "-geo", geo,
                "-radius", str(search_radius),
                "-depth", "20"
            ]
            
            print(f"🚀 Running scraper for coordinates: {geo} ({country_name})")
            subprocess.run(cmd, capture_output=True, text=True)
            print(f"✅ Finished for coordinates {geo}")
            print(f"📄 Output: {loc_ids_scrapper_output}\n")
        
        #combine location id csvs and remove duplicates
        data_ids_output_file_name = os.path.join(self.DATA_DIR, "data_ids.csv")
        id_col = "data_id"
        review_col = "review_count"
        locs_files = [f for f in os.listdir(self.LOC_DIRS) if f.endswith(".csv") and f.startswith("locs")]
        unique_rows = dict()  # key = data_id, value = review_count
        for csv_file in locs_files:
            file_path = os.path.join(self.LOC_DIRS, csv_file)
            try:
                df = pd.read_csv(file_path)
                if id_col in df.columns and review_col in df.columns:
                    for _, row in df.iterrows():
                        data_id = str(row[id_col])
                        review_count = row[review_col]
                        if data_id not in unique_rows:
                            unique_rows[data_id] = review_count
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
        output_df = pd.DataFrame(list(unique_rows.items()), columns=[id_col, review_col])
        output_df.to_csv(data_ids_output_file_name, index=False)
        print(f"🎉 Saved {len(unique_rows)} unique rows to {data_ids_output_file_name}")
        df_filtered = output_df[output_df["review_count"] >= filter_min_reviews]    #Omit samples with less than # review count
        df_filtered.to_csv(f"{self.LOC_DIRS}/data_ids_filtered.csv", index=False)
        print("✅ Filtered CSV saved as 'data_ids_filtered.csv'")
        
    def get_geo(self, geo_bounds: Dict[str, Dict[str, tuple]] = None) -> tuple:
        if not geo_bounds:
            geo_bounds = {
                "USA": {
                    "latitude": (32, 49),
                    "longitude": (-124, -66)   
                },
                "UK": {
                    "latitude": (50, 58),     
                    "longitude": (-5, 1)       
                },
                "VN_HN": {
                    "latitude": (20, 22),     
                    "longitude": (105, 106)      
                },
                "VN_HCM": {
                    "latitude": (10, 11),     
                    "longitude": (106, 107)      
                },
            }
        country_name = random.choice(list(geo_bounds.keys()))
        lat_min, lat_max = geo_bounds[country_name]["latitude"]
        lon_min, lon_max = geo_bounds[country_name]["longitude"]
        random_lat = random.uniform(lat_min, lat_max)
        random_lon = random.uniform(lon_min, lon_max)
        return (random_lat, random_lon, country_name)
    
    
    def scrape_loc_data(self,
        api_key: str = None,
        sort_reviews_by: str = "Relevant" in ["Relevant", "Highest", "Lowest"],
        max_loc: int = 5
        ):
        
        if not api_key:
            print("No API key provided. Insert API key from link: https://rapidapi.com/alexanderxbx/api/maps-data/playground/apiendpoint_2f487f1c-3516-49b6-87e4-b45a5d05e2e4?env=undefined")
            return        
            
        #prep ids
        df_filtered = pd.read_csv(f"{self.LOC_DIRS}/data_ids_filtered.csv")
        exploited_file_path = f"{self.RESPONSES_DIR}/exploited_id.csv"
        column_name = "data_id"

        if os.path.exists(exploited_file_path):
            existing_df = pd.read_csv(exploited_file_path)
            exploited_id = set(existing_df[column_name].astype(str))
        else:
            exploited_id = set()
            with open(exploited_file_path, "a") as f:
                f.write("data_id")    #Create file for later use
        key_col = "data_id"
        df_new = df_filtered[~df_filtered[key_col].isin(existing_df[key_col])]
        
        print(f"✅ All: {len(df_filtered)} ids")
        print(f"🧹 Existing:  {len(existing_df)} ids")
        print(f"✨ New:      {len(df_new)} ids")
        
        unique_ids = set(df_new["data_id"])
        unique_rows = dict(zip(df_new['data_id'], df_new['review_count']))

        #scrape reviews        
        def get_last_review_cursor(json_data):
            try:
                reviews = json_data['data']['reviews']
                
                if reviews:
                    last_review = reviews[-1]
                    
                    cursor = last_review.get('review_cursor')
                    return cursor
                else:
                    print("The 'reviews' list is empty.")
                    return None
                    
            except KeyError as e:
                print(f"Error: Missing key in the JSON structure: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
    
        url = "https://maps-data.p.rapidapi.com/reviews.php"
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "maps-data.p.rapidapi.com"
        }
        
        EXCEED_QUOTA = False
        existing_df = pd.read_csv(exploited_file_path)
        exploited_id = set(existing_df[column_name].astype(str))
        visited_loc = 0
        for id in unique_ids:
            print("Scraping id: ", id)
            if id in exploited_id:
                print("The id was already used!!!")
                continue
            cursor_count = 0
            cursor = None
            while cursor_count <= unique_rows[id]/40:
                querystring = {"business_id": id,
                            "lang":"en",
                            "limit":"20",
                            "cursor": cursor,
                            "sort":sort_reviews_by}

                response = requests.get(url, headers=headers, params=querystring)
                
                json_data = response.json()

                if isinstance(json_data, dict) and "message" in json_data:
                    EXCEED_QUOTA = True
                    break

                cursor = get_last_review_cursor(json_data)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_id = id.replace(":", "_")
                response_file = os.path.join(self.RESPONSES_DIR, f"response-{timestamp}-{safe_id}.json")

                with open(response_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                print("Saved response to: ", response_file)
                cursor_count += 1

            if EXCEED_QUOTA:
                print("It's time to update your API!!!")
                break
            
            exploited_id.add(id)
            visited_loc += 1
            if visited_loc >= max_loc:
                print(f"Reached the limit of {max_loc} locations. Stopping.")
                break
            
        pd.DataFrame(sorted(exploited_id), columns=[column_name]).to_csv(exploited_file_path, index=False)
        print(f"✅ Saved {len(exploited_id)} exploited IDs to '{exploited_file_path}'")
        # self.extract_reviews()
        
    def extract_reviews(self):
        #extract reviews from responses
        extracted_data = {'review_id': [], 'text': [], 'rating': []}
        filedirs = os.listdir(self.RESPONSES_DIR)
        total_files = len(filedirs)
        processed_files = 0
        num_of_reviews = 0
        for filename in filedirs:
            processed_files += 1
            print(f"Read {processed_files} of {total_files} files, added {num_of_reviews} reviews", end='\r')
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self.RESPONSES_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not read {file_path}: {e}")
                continue
            reviews = data.get("data", {}).get("reviews", [])
            if not reviews:
                continue
            for review in reviews:
                review_id = review['review_id']
                rating = review['review_rate']
                
                translation = review.get("translation", {})
                text = translation.get('en') or review.get("review_text", "")
                if text is None: continue
                text = text.replace("\n", "") or ""      #remove line break so it can be save into a csv"
                if len(text.strip()) < 10:
                    continue                            #reject too short reviews
                extracted_data['review_id'].append(review_id)
                extracted_data['text'].append(text)
                extracted_data['rating'].append(rating)
                num_of_reviews += 1
        print()
        df = pd.DataFrame(extracted_data)
        print("Number of reviews: " + str(df.shape[0]))
        df = df.drop_duplicates()
        print("After dropping duplicates: " + str(df.shape[0]))
        df.to_csv(os.path.join(self.DATA_DIR, "reviews.csv"), index=False)

    