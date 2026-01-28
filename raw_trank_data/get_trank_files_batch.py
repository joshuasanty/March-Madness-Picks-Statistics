#Doesn't Work!
import requests

base_url = "https://barttorvik.com/trank.php"

for year in range(2010, 2024):
    params = {
        "year": year,
        "csv": 1
    }
    filename = f"trank_data_{year}.csv"

    r = requests.get(base_url, params=params)
    r.raise_for_status()  # fail loudly if something breaks

    with open(filename, "wb") as f:
        f.write(r.content)

    print(f"Saved {filename}")
