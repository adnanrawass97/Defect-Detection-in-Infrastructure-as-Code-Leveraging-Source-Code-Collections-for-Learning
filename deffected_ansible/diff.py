import os
import requests
import re
import csv
import time

# Access GitHub Personal Access Token from environment variable
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

# Path to your CSV file containing repository, commit IDs, and file paths
CSV_FILE = 'ansible.csv'

# Path to the output CSV file for changes
CHANGES_FILE = 'commit_changes.csv'

def fetch_commit_details(owner, repo, ref):
    """Fetch details of a specific commit from GitHub."""
    url = f'https://api.github.com/repos/{owner}/{repo}/commits/{ref}'
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

    # Make a GET request to fetch commit details with headers
    response = requests.get(url, headers=headers)
   
    if response.status_code == 200:
        commit_details = response.json()
        return commit_details
    else:
        print(f"Error: Could not fetch commit details for '{ref}' in repository '{owner}/{repo}'")
        return None

def process_diff(diff_text):
    """
    Processes a git diff patch to extract added and removed lines.
    
    Args:
    - diff_text (str): The git diff patch text.
    
    Returns:
    - Tuple of lists containing added and removed lines.
    """
    added_lines = []
    removed_lines = []
    block_pattern = re.compile(r'^@@ -\d+,\d+ \+\d+,\d+ @@.*$')
    
    for line in diff_text.split('\n'):
        if block_pattern.match(line):
            continue  # Skip the block header lines
        elif line.startswith('-'):
            removed_lines.append(line[1:])  # Capture and remove the '-' prefix
        elif line.startswith('+'):
            added_lines.append(line[1:])  # Capture and remove the '+' prefix
    
    return added_lines, removed_lines

def main():
    # Open the changes CSV file for writing with headers
    with open(CHANGES_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Commit ID", "Repo Name", "File Path", "Deleted Lines", "Added Lines"])

    # Read repository, commit IDs, and file paths from CSV
    with open(CSV_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)  # Use DictReader to access columns by name
        for row in reader:
            repo = row.get('repository')  # Get the repository name from the 'repository' column
            commit_id = row.get('commit')  # Get the commit ID from the 'commit' column
            file_path = row.get('filepath')  # Get the file path from the 'filepath' column

            if '/' not in repo:
                print(f"Error: Invalid repository format '{repo}'. It should be in the format 'owner/repo'.")
                continue
            
            owner, repo_name = repo.split('/')
            
            if owner and repo_name and commit_id and file_path:
                commit_details = fetch_commit_details(owner, repo_name, commit_id)
                if commit_details:
                    files = commit_details.get('files', [])
                    for file_info in files:
                        filename = file_info.get('filename')
                        if filename == file_path:
                            patch = file_info.get('patch')
                            added_lines, removed_lines = process_diff(patch)
                            with open(CHANGES_FILE, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([commit_id, repo, file_path, "\n".join(removed_lines), "\n".join(added_lines)])
                            break  # Once found, no need to continue searching for the file
                    else:
                        print(f"No '{file_path}' file found for commit '{commit_id}' in repository '{repo}'.")
                else:
                    print(f"Failed to fetch commit details for commit '{commit_id}' in repository '{repo}'.")
            else:
                print("Warning: Insufficient data in CSV row. Skipping.")
            
            # Wait for 1000ms (1 second)
            time.sleep(5)

if __name__ == "__main__":
    main()

# import os
# import requests
# import re
# import base64
# import csv
# import time

# # Access GitHub Personal Access Token from environment variable
# ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

# # Path to your CSV file containing repository, commit IDs
# CSV_FILE = 'ansible.csv'

# # Path to the output CSV file for changes
# CHANGES_FILE = 'commit_changes.csv'

# def fetch_commit_details(owner, repo, ref):
#     """Fetch details of a specific commit from GitHub."""
#     url = f'https://api.github.com/repos/{owner}/{repo}/commits/{ref}'
#     print(url)
#     header = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

#     # Make a GET request to fetch commit details with headers
#     response = requests.get(url, headers=header)

#     if response.status_code == 200:
#         commit_details = response.json()
#         return commit_details
#     else:
#         print(f"Error: Could not fetch commit details for '{ref}' in repository '{owner}/{repo}'")
#         return None

# def process_diff(diff_text):
#     """
#     Processes a git diff patch to extract added and removed lines.
    
#     Args:
#     - diff_text (str): The git diff patch text.
    
#     Returns:
#     - Tuple of lists containing added and removed lines.
#     """
#     added_lines = []
#     removed_lines = []
#     context_lines_before = []
#     context_lines_after = []

#     block_pattern = re.compile(r'^@@ -\d+,\d+ \+\d+,\d+ @@.*$')
    
#     for line in diff_text.split('\n'):
#         if block_pattern.match(line):
#             context_lines_before = []
#             context_lines_after = []
#             continue  # Skip the block header lines
#         elif line.startswith('-'):
#             removed_lines.append(line[1:])  # Capture and remove the '-' prefix
#         elif line.startswith('+'):
#             added_lines.append(line[1:])  # Capture and remove the '+' prefix
#         elif line.startswith(' '):
#             context_lines_before.append(line[1:])
#             context_lines_after.append(line[1:])

#         if len(context_lines_before) > 2:
#             context_lines_before.pop(0)
#         if len(context_lines_after) > 2:
#             context_lines_after.pop(0)
    
#     return added_lines, removed_lines, context_lines_before, context_lines_after

# def main():
#     # Open the changes CSV file for writing with headers
#     with open(CHANGES_FILE, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Commit ID", "Repo Name", "File Path", "Deleted Lines", "Added Lines", "Context Lines Before", "Context Lines After"])

#     # Read repository and commit IDs from CSV
#     with open(CSV_FILE, 'r') as csvfile:
#         reader = csv.DictReader(csvfile)  # Use DictReader to access columns by name
#         for row in reader:
#             repo = row.get('repository')  # Get the repository name from the 'repository' column
#             commit_id = row.get('commit')  # Get the commit ID from the 'commit' column
#             file_path = row.get('filepath')  # Get the file path from the 'filepath' column

#             if '/' not in repo:
#                 print(f"Error: Invalid repository format '{repo}'. It should be in the format 'owner/repo'.")
#                 continue
            
#             owner, repo_name = repo.split('/')
            
#             if owner and repo_name and commit_id:
#                 commit_details = fetch_commit_details(owner, repo_name, commit_id)
#                 if commit_details:
#                     files = commit_details.get('files', [])
#                     if files:
#                         for file in files:
#                             if file['filename'] == file_path:
#                                 patch = file['patch']
#                                 added_lines, removed_lines, context_lines_before, context_lines_after = process_diff(patch)
#                                 with open(CHANGES_FILE, 'a', newline='') as csvfile:
#                                     writer = csv.writer(csvfile)
#                                     writer.writerow([commit_id, repo, file_path, "\n".join(removed_lines), "\n".join(added_lines), "\n".join(context_lines_before), "\n".join(context_lines_after)])
#                                 break
#                         else:
#                             print(f"No '{file_path}' file found for commit '{commit_id}' in repository '{repo}'.")
#                     else:
#                         print(f"No files found for commit '{commit_id}' in repository '{repo}'.")
#                 else:
#                     print(f"Failed to fetch commit details for commit '{commit_id}' in repository '{repo}'.")
#             else:
#                 print("Warning: Insufficient data in CSV row. Skipping.")

#             time.sleep(1)  # Wait for 1 second between each fetch

# if __name__ == "__main__":
#     main()
