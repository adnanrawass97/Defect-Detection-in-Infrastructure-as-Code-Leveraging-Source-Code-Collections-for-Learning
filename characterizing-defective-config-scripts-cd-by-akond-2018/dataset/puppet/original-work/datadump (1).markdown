---
layout: "post"
title: "DataDump"
date: "2017-12-22 17:27"
---
### Data dump for paper 'Characterizing Defective Configuration Scripts Used For Continuous Deployment'
#### Instructions

- Download SCRIPT.LABELS.DUMP. This is a pickle file, where a dictionary is stored. The keys of the dictionary is the ID of the files, and the value of the key is a tuple where the first entry is the content of the script, the second entry is the defect label for the script, and the third entry is dataset name (0 for Mozilla, 1 for Openstack, and 2 for Wikimedia). 
- import the pickle file using

>  all_script_dict = pickle.load( open('SCRIPT.LABELS.DUMP', 'rb'))

- iterate over the dictionary
