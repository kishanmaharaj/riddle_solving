from datasets import load_dataset
import evaluate
import logging
import jsonlines
import torch
from tqdm import tqdm


# Logging
logging.basicConfig(filename='std_script.log', filemode='w', format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')
logger=logging.getLogger() 
logger.setLevel(logging.INFO)



device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Current Device: " + device)




# GreaseLM
from transformers import GreaseLMProcessor
from transformers import GreaseLMForMultipleChoice
proc = GreaseLMProcessor.from_pretrained("Xikun/greaselm-csqa")
model = GreaseLMForMultipleChoice.from_pretrained("Xikun/greaselm-csqa")


_ = model.to(device)
logger.info("GreaseLM Loaded")



# Riddle Sense Data
options = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
riddle_sense = []
with jsonlines.open('riddle_sense/rs_train.jsonl') as f:
    for line in f:
        riddle_sense.append(line)
        
logger.info("Dataset Loaded")




acc = evaluate.load("accuracy")

batch_size = 4
prediction_count = 0
current_index = batch_size

for i in tqdm(range(0, len(riddle_sense), batch_size), total=int(len(riddle_sense)/batch_size), desc="Evaluating"):
    batch = riddle_sense[i:i+batch_size]
    batch = proc(batch)
    output = model(**batch.to(device))
    predictions = output.logits.argmax(1)
    acc.add_batch(references=batch["labels"], predictions=predictions) 
    
    predictions = predictions.cpu().numpy()
    cnt = len(predictions)
    
    for i in range(len(predictions)):
        logger.info("Current Prediction: " + str(prediction_count) + ":  " + options[predictions[i]])
        prediction_count += 1 



    for index in range (i,current_index):
        if batch_size-cnt < len(predictions):
            riddle_sense[index]['prediction'] = options[predictions[batch_size-cnt]]
            cnt = cnt-1


    current_index = i + batch_size
    
logger.info("Prediction complete")
    
current_accuracy = acc.compute()    
print(current_accuracy)

logger.info("Current Accuracy: " + str(current_accuracy['accuracy']))


# Saving the Inference File
import json
with open('inference.json', 'w') as fout:
    json.dump(riddle_sense, fout)

logger.info("Inference file saved")