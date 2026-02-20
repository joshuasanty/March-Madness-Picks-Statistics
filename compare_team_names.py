import pandas as pd
import difflib

#This file just shows which team names don't match between the two datasets
#Note, some teams aren't in recent tournament years so they won't have any matches.
def compare_team_names():
    # Read the MTeams.csv file
    mteams_df = pd.read_csv('kaggle_ncaa_data/MTeams.csv')
    
    # Read the clean_trank_data_2025.csv file
    trank_df = pd.read_csv('clean_trank_data/clean_trank_data_2025.csv')
    
    # Get team names from both datasets
    mteams_names = mteams_df['TeamName'].tolist()
    trank_names = trank_df['Team'].tolist()
    
    # Find differences
    differences = []
    
    # Create a set for faster lookup
    trank_names_set = set(trank_names)
    
    for idx, mteam_name in enumerate(mteams_names, 1):  # Start from 1 for row numbers
        # Check if exact match exists
        if mteam_name not in trank_names_set:
            # Check for close matches using difflib
            close_matches = difflib.get_close_matches(mteam_name, trank_names, n=3, cutoff=0.8)
            
            differences.append({
                'row_number': idx + 1,  # +1 because header is row 1
                'mteams_name': mteam_name,
                'close_matches': close_matches if close_matches else ['No close matches found']
            })
    
    # Also check for teams in trank that don't exist in mteams
    mteams_names_set = set(mteams_names)
    trank_only = []
    
    for idx, trank_name in enumerate(trank_names, 1):
        if trank_name not in mteams_names_set:
            close_matches = difflib.get_close_matches(trank_name, mteams_names, n=3, cutoff=0.8)
            trank_only.append({
                'row_number': idx + 1,
                'trank_name': trank_name,
                'close_matches': close_matches if close_matches else ['No close matches found']
            })
    
    # Write results to text file
    with open('team_name_differences.txt', 'w', encoding='utf-8') as f:
        f.write("Team Name Differences Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Teams in MTeams.csv that don't match exactly in clean_trank_data_2025.csv:\n")
        f.write("-" * 70 + "\n")
        
        if differences:
            for diff in differences:
                f.write(f"Row {diff['row_number']}: {diff['mteams_name']}\n")
                f.write(f"  Close matches in trank data: {', '.join(diff['close_matches'])}\n\n")
        else:
            f.write("No differences found - all MTeams names have exact matches in trank data.\n\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("Teams in clean_trank_data_2025.csv that don't match exactly in MTeams.csv:\n")
        f.write("-" * 70 + "\n")
        
        if trank_only:
            for diff in trank_only:
                f.write(f"Row {diff['row_number']}: {diff['trank_name']}\n")
                f.write(f"  Close matches in MTeams: {', '.join(diff['close_matches'])}\n\n")
        else:
            f.write("No differences found - all trank names have exact matches in MTeams data.\n\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Summary:\n")
        f.write(f"- Total teams in MTeams.csv: {len(mteams_names)}\n")
        f.write(f"- Total teams in clean_trank_data_2025.csv: {len(trank_names)}\n")
        f.write(f"- Teams needing attention from MTeams.csv: {len(differences)}\n")
        f.write(f"- Teams needing attention from trank data: {len(trank_only)}\n")
    
    print(f"Analysis complete! Results saved to 'team_name_differences.txt'")
    print(f"Found {len(differences)} teams in MTeams.csv that need attention")
    print(f"Found {len(trank_only)} teams in trank data that need attention")

if __name__ == "__main__":
    compare_team_names()
