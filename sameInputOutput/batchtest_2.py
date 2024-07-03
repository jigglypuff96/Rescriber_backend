import ollama
import pandas as pd
# params explained 
# https://github.com/ggerganov/llama.cpp/tree/master/examples/main#generation-flags 
# Define the model
model = 'llama3'
text1 = "Is San Zhang and Zhang, San same entity? only answer yes or no"
text2 = """
Detect all the personal idetifiable info in the following text. 
Review the following dataset and come up with insightful observations:
Trip ID	Destination	Start date	End date	Duration (days)	Traveler name	Traveler age	Traveler gender	Traveler nationality	Accommodation type	Accommodation cost	Transportation type	Transportation cost
1	London, UK	5/1/2023	5/8/2023	7	John Smith	35	Male	American	Hotel	1200	Flight	600
"""
text3 = """
Detect all the personal idetifiable info in the following text. return json format
Review the following dataset and come up with insightful observations:
Trip ID	Destination	Start date	End date	Duration (days)	Traveler name	Traveler age	Traveler gender	Traveler nationality	Accommodation type	Accommodation cost	Transportation type	Transportation cost
1	London, UK	5/1/2023	5/8/2023	7	John Smith	35	Male	American	Hotel	1200	Flight	600
2	Phuket, Thailand	6/15/2023	6/20/2023	5	Jane Doe	28	Female	Canadian	Resort	800	Flight	500
3	Bali, Indonesia	7/1/2023	7/8/2023	7	David Lee	45	Male	Korean	Villa	1000	Flight	700
4	New York, USA	8/15/2023	8/29/2023	14	Sarah Johnson	29	Female	British	Hotel	2000	Flight	1000
5	Tokyo, Japan	9/10/2023	9/17/2023	7	Kim Nguyen	26	Female	Vietnamese	Airbnb	700	Train	200
6	Paris, France	10/5/2023	10/10/2023	5	Michael Brown	42	Male	American	Hotel	1500	Flight	800
7	Sydney, Australia	11/20/2023	11/30/2023	10	Emily Davis	33	Female	Australian	Hostel	500	Flight	1200
8	Rio de Janeiro, Brazil	1/5/2024	1/12/2024	7	Lucas Santos	25	Male	Brazilian	Airbnb	900	Flight	600
9	Amsterdam, Netherlands	2/14/2024	2/21/2024	7	Laura Janssen	31	Female	Dutch	Hotel	1200	Train	200
10	Dubai, United Arab Emirates	3/10/2024	3/17/2024	7	Mohammed Ali	39	Male	Emirati	Resort	2500	Flight	800
11	Cancun, Mexico	4/1/2024	4/8/2024	7	Ana Hernandez	27	Female	Mexican	Hotel	1000	Flight	500
12	Barcelona, Spain	5/15/2024	5/22/2024	7	Carlos Garcia	36	Male	Spanish	Airbnb	800	Train	100
13	Honolulu, Hawaii	6/10/2024	6/18/2024	8	Lily Wong	29	Female	Chinese	Resort	3000	Flight	1200
14	Berlin, Germany	7/1/2024	7/10/2024	9	Hans Mueller	48	Male	German	Hotel	1400	Flight	700
15	Marrakech, Morocco	8/20/2024	8/27/2024	7	Fatima Khouri	26	Female	Moroccan	Riad	600	Flight	400
16	Edinburgh, Scotland	9/5/2024	9/12/2024	7	James MacKenzie	32	Male	Scottish	Hotel	900	Train	150
17	Paris	9/1/2023	9/10/2023	9	Sarah Johnson	30	Female	American	Hotel	$900 	Plane	$400
18	Bali	8/15/2023	8/25/2023	10	Michael Chang	28	Male	Chinese	Resort	$1,500 	Plane	$700
19	London	7/22/2023	7/28/2023	6	Olivia Rodriguez	35	Female	British	Hotel	$1,200 	Train	$150
20	Tokyo	10/5/2023	10/15/2023	10	Kenji Nakamura	45	Male	Japanese	Hotel	$1,200 	Plane	$800
21	New York	11/20/2023	11/25/2023	5	Emily Lee	27	Female	American	Airbnb	$600 	Bus	$100
22	Sydney	12/5/2023	12/12/2023	7	James Wilson	32	Male	Australian	Hotel	$1,000 	Plane	$600
23	Rome	11/1/2023	11/8/2023	7	Sofia Russo	29	Female	Italian	Airbnb	$700 	Train	$80
24	Bangkok	9/15/2023	9/23/2023	8	Raj Patel	40	Male	Indian	Hostel	$400 	Plane	$500
25	Paris	12/22/2023	12/28/2023	6	Lily Nguyen	24	Female	Vietnamese	Hotel	$1,400 	Train	$100
26	Hawaii	8/1/2023	8/10/2023	9	David Kim	34	Male	Korean	Resort	$2,000 	Plane	$800
27	Barcelona	10/20/2023	10/28/2023	8	Maria Garcia	31	Female	Spanish	Hotel	$1,100 	Train	$150
28	Japan	5/10/2022	5/18/2022	8	Alice Smith	30	Female	American	Hotel	$800 	Plane	$500
29	Thailand	6/15/2022	6/22/2022	7	Bob Johnson	45	Male	Canadian	Hostel	$200 	Train	$150
30	France	7/2/2022	7/11/2022	9	Charlie Lee	25	Male	Korean	Airbnb	$600 	Car rental	$300
31	Australia	8/20/2022	9/2/2022	13	Emma Davis	28	Female	British	Hotel	$1,000 	Car rental	$500
32	Brazil	9/5/2022	9/14/2022	9	Olivia Martin	33	Female	Australian	Hostel	$150 	Bus	$50
33	Greece	10/12/2022	10/20/2022	8	Harry Wilson	20	Male	American	Airbnb	$400 	Plane	$600
34	Egypt	11/8/2022	11/15/2022	7	Sophia Lee	37	Female	Canadian	Hotel	$700 	Train	$100
35	Mexico	1/5/2023	1/15/2023	10	James Brown	42	Male	British	Airbnb	$500 	Plane	$800
36	Italy	2/14/2023	2/20/2023	6	Mia Johnson	31	Female	American	Hostel	$180 	Train	$120
37	Spain	3/23/2023	3/31/2023	8	William Davis	27	Male	Korean	Hotel	$900 	Car rental	$400
38	Canada	4/19/2023	4/26/2023	7	Amelia Brown	38	Female	Australian	Airbnb	$350 	Bus	$75
39	Paris, France	6/12/2022	6/19/2022	7	Mia Johnson	25	Female	American	Hotel	1400	Plane	600
40	Sydney, Australia	1/2/2023	1/9/2023	7	Adam Lee	33	Male	Canadian	Airbnb	800	Train	150"""
# Define the messages
messages = [
    {
        'role': 'user',
        'content': text2
    }
]

# Base options dictionary
base_options = {
    # "format":"json",
    "stream":True,
    "num_keep": 5,
    "seed": 40,
    "num_predict": 100, #100
    "top_k": 20,
    "top_p": 0.9,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": True,
    "stop": ["\n", "user:"],
    "numa": False,
    "num_ctx": 1024,
    "num_batch": 2,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": False,
    "f16_kv": True,
    "vocab_only": False,
    "use_mmap": True,
    "use_mlock": False,
    "num_thread": 8
}

# Function to append data to CSV
def append_to_csv(data, file):
    df = pd.DataFrame(data)
    df.to_csv(file, mode='a', header=not pd.io.common.file_exists(file), index=False)

# Define the CSV file
csv_file = 'input_output_results.csv'

# List to store results
results = []

notes = ""
# Run 4 requests with modifications
for i in range(6):

    options = base_options.copy()
    
    # Modify options based on request number
    if i == 0 or i==1:
        options['num_keep'] = 30
    
    elif i == 2 or i==3:
        options["num_predict"]=100
    else:
        pass
    #     pass
    #     # options['top_k'] = 30  
    # text2 yes or no
    #text1 long text
    # if i == 0:
    #     text_content = text1
    # elif i == 1:
    #     text_content = text2
    # elif i == 2:
    #     text_content = text1
    # elif i == 3:
    #     text_content = text2
    # elif i == 4:
    #     text_content = text1
    # elif i == 5:
    #     text_content = text2
    text_content = text1
    
    messages = [
        {
            'role': 'user',
            'content': text_content
        }
    ]       
    
    # Call the API
    response = ollama.chat(
        model=model,
        messages=messages,
        options=options
    )
    
    content = response['message']['content']
    print(i,content)
    
    # Prompt user for notes
    # notes = input(f"Enter notes for response {i + 1}: ")
    
    # Append to results list
    results.append({
        'request_number': i + 1,
        'modified_option': options,
        'content': content,
        'notes': notes
    })

# Append all results to CSV
append_to_csv(results, csv_file)

print("Results appended to", csv_file)
