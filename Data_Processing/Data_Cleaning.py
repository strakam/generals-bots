import json
from datetime import datetime

start=datetime.strptime("2022-06-04T00:00:00Z","%Y-%m-%dT%H:%M:%SZ")
Name = ['MeltedToast', 'Mithraaaa']
for name in Name:
    print(name,end=': ')
    input_file = "./AlphaGen/Replays/Total_Data/"+name+".json"
    with open(input_file,"r",encoding="utf-8") as file:
        data=json.load(file)
    print(len(data),end=' ')
    filtered_data=[
        entry for entry in data
        if entry["type"]=="1v1" and
           entry["turns"]>=80 and 
           start<=datetime.fromtimestamp(entry['started']/1000)
    ]
    output_file = "./AlphaGen/Replays/Total_Data/filtered_"+name+".json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    
    print(len(filtered_data))
