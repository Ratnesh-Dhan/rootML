import os, re
import pandas as pd 

base_path = "./resultsHimanshu"
# base_path = "/mnt/d/RESULTS/november20_coal_classification/november18"
# model_names = os.listdir(base_path)
# print(model_names)
# model_names = ['CNNmodelJUNE24', 'Nadam_earlystopped_best_epoch30']
model_names = ['Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Nadam', 'RMSprop']
for model_name in model_names:
    file = os.path.join(base_path, model_name,"final_output_new_full.txt")

    count = 0
    count2 = 3
    count3 = 0
    count4 = False
    row = []

    with open(file) as f:
        row_data = {}
        for lines in f:
            count = count + 1
            count2 = count2 + 1

            if count % 4 == 0:
                continue
            if count2 % 4 == 0:
                count2 = 0
                continue
            else:
                count3 = count3 + 1
                x = lines.split(':')
                count4 = not count4
                if count4:
                    row_data['Sample'] = x[0].strip().split(' ')[0].strip() 
                y = x[0].strip().split(' ')
                row_data[f'{y[1].strip()} {y[2].strip()}'] = round(float(x[1].strip()), 4)
            
            if count3%2 == 0:
                row.append(row_data)
                row_data = {}

    for i in row:
        print(i) 
    # #✅ Sort by Sample number (numerical order)
    # row.sort(key=lambda r: int(r['Sample'][1:])) # FROM C
    # # row.sort(key=lambda r: int(r['Sample'][3:])) # FOR DBM

    
    # This is 2nd last attempt .
    # row.sort(key=lambda r: int(re.findall(r'\d+', r['Sample'])[0])) 


    def sample_sort(r):
        sample = r['Sample']

        # Extract number
        num = int(re.findall(r'\d+', sample)[0])

        # Put C before DBM
        if sample.startswith('C'):
            prefix = 0
        elif sample.startswith('DBM'):
            prefix = 1
        else:
            prefix = 2

        return (prefix, num)

    row.sort(key=sample_sort)





    #     # Sort properly
    # row.sort(key=lambda r: r['Sample'])

    # Build new dataframe with only required columns
    df = pd.DataFrame(row)[['Sample', 'Average ash']]
    # df = pd.DataFrame(row)
    # output_file = "../results/Excel_Files/Sample_DBM"
    # output_file = "/mnt/d/RESULTS/november20_coal_classification/november18/Excel_Files/C"
    output_file = base_path
    # os.makedirs(output_file, exist_ok=True)
    df.to_excel(os.path.join(output_file, f'{model_name}.xlsx'), index=False)
    