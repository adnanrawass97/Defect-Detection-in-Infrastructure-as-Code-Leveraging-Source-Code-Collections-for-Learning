import pandas as pd
# Load data from CSV file
data = pd.read_csv('GLITCH-ansible.csv')
# Define the list of categories
category_of_defect = [
    "sec_hard_secr",
    "sec_hard_user",
    "sec_susp_comm",
    "sec_invalid_bind",
    "sec_https",
    "sec_def_admin",
    "sec_hard_pass",
    "sec_empty_pass",
    "sec_no_int_check",
    "sec_weak_crypt",
   
]

result = pd.DataFrame()

# Creating a dictionary 
category_counts = {category: 0 for category in category_of_defect}
total_count = 0

number_to_collecte=40
number_cat=10
# Iterate over each row in the data
for index, row in data.iterrows():
    
    category_defect = row['category_defect']
    
    if(number_to_collecte*number_cat==total_count):
       break   
    # Checking if the category defect matches a specific category

    if category_counts[category_defect] <= number_to_collecte:
            value=category_counts[category_defect]
            category_counts[category_defect]= value+1
            before_column = row['Defection']
            src_column = row['sc']
            line_number_column = row['Line_Number']

            # DataFrame 
            temp_df = pd.DataFrame({
                'Before':  [before_column] ,
                'Line_Number':  [line_number_column],
                'src': [src_column],
                'label_before': [1]  
            })
            total_count = sum(category_counts.values())
 

            # Append temp_df 
            result = pd.concat([result, temp_df], ignore_index=True)

# Saving 
result.to_csv('GLITCH-ansible_test.csv', index=False)

# Print the count of occurrences for each category
for category, count in category_counts.items():
    print(f"Category: {category}, Count: {count}")
