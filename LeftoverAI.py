import requests 
import json
import jetson.inference
import jetson.utils
import cv2
import numpy as np

def jprint(obj):
    # create a formatted string of the Python JSON object, this is only needed for error handling
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

# configuring the camera settings
width=1280
height=720
dispW=width
dispH=height
flip=2
# optional: using an IP streaming camera based on a raspberry Pi Zero W
# camSet=' tcpclientsrc host=[insertIP] port=8554 ! gdpdepay ! rtph264depay ! h264parse ! nvv4l2decoder  ! nvvidconv flip-method='+str(flip)+' ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+',format=BGR ! appsink  drop=true sync=false '
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)

# using the model I trained via transfer learning
net=jetson.inference.imageNet('alexnet',['--model=/home/nils/Downloads/jetson-inference/python/training/classification/Leftovers/resnet18.onnx','--input_blob=input_0','--output_blob=output_0','--labels=/home/nils/Downloads/jetson-inference/Foods/labels.txt'])

# configuring input and output
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,frame=cam.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    classID, confidence=net.Classify(img, width, height)
    # using my model to identify the item
    item=''
    item=net.GetClassDesc(classID)
    # displaying item type
    cv2.putText(frame,item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('Cam',frame)
    cv2.moveWindow('Cam',0,0)
    if cv2.waitKey(1)==ord('a'):
        item1=''
        item1=net.GetClassDesc(classID)
    if cv2.waitKey(1)==ord('b'):
        item2=''
        item2=net.GetClassDesc(classID)
    if cv2.waitKey(1)==ord('c'):
        item3=''
        item3=net.GetClassDesc(classID)
    # press p to get 5 recipes
    if cv2.waitKey(1)==ord('p'):
        if 'item1' in locals() and 'item2' in locals() and 'item3' in locals():
            r = requests.get(f"https://api.spoonacular.com/recipes/findByIngredients?apiKey=e626af65626e433da41d815db5eb4f17&ingredients={item1,item2,item3}&instructionsRequired=true&instructions&number=5")
            # api call searching for 5 recipes with instructions which include the three items the model has recognized

            recipes = r.json()
            for i in range(0, 5): # looping through the recipes and handling the json data
                f = open("recipe.txt", "a")
                f.write(f"Recipe {i+1}:\n")
                f.close()
                try:
                    recipe = recipes[i]
                    name = recipe['title'] # getting the recipe name
                    f = open("recipe.txt", "a")
                    f.write(f"{name}\n")
                    f.close()
                    missedIngredients = recipe['missedIngredients'] # getting additional ingredients
                    ingredientNR = len(missedIngredients)
                    f = open("recipe.txt", "a")
                    f.write("Ingredients:\n")
                    f.close()
                    f = open("recipe.txt", "a")
                    f.write(f"{item1}\n{item2}\n{item3}\n")
                    f.close()
                    for i in range(0, ingredientNR): # getting all additional ingridients in another for loop
                        ingredientn = missedIngredients[i]
                        f = open("recipe.txt", "a")
                        f.write(f"{ingredientn['name']}\n")
                        f.close()
                except IndexError:
                    f = open("recipe.txt", "a")
                    f.write("Sorry, this ingredient doesn't have that many recipes yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n")
                f.close()

                id = recipe['id']
                r1 = requests.get(f"https://api.spoonacular.com/recipes/{id}/analyzedInstructions?apiKey=e626af65626e433da41d815db5eb4f17")
                # making another API request to get the steps of the recipe
                allInfo = r1.json()
                # jprint(allInfo) #uncomment this step for error handling
                try:
                    Steps = allInfo[0]
                    justSteps = Steps['steps']
                    stepNR = (len(justSteps))
                    f = open("recipe.txt", "a")
                    f.write("Steps:\n")
                    f.close()
                    for i in range(0, stepNR): #looping through the steps
                        stepn = justSteps[i]
                        f = open("recipe.txt", "a")
                        f.write(f"Step {i+1}: {stepn['step']}\n")
                        f.close()
                except IndexError:
                    # weirdly, some recipes don't contain steps, even though the first API call explicitly asks for recipes with instructions
                    f = open("recipe.txt", "a")
                    f.write("Sadly, this recipe does not have any steps yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n---\n \n")
                f.close()
            


        elif 'item1' in locals() and 'item2' in locals(): 
            r = requests.get(f"https://api.spoonacular.com/recipes/findByIngredients?apiKey=e626af65626e433da41d815db5eb4f17&ingredients={item1,item2}&instructionsRequired=true&instructions&number=5")
            # api call searching for 5 recipes with instructions which include the 2 items the model has recognized

            recipes = r.json()
            for i in range(0, 5): # looping through the recipes and handling the json data
                f = open("recipe.txt", "a")
                f.write(f"Recipe {i+1}:\n")
                f.close()
                try:
                    recipe = recipes[i]
                    name = recipe['title'] # getting the recipe name
                    f = open("recipe.txt", "a")
                    f.write(f"{name}\n")
                    f.close()
                    missedIngredients = recipe['missedIngredients'] # getting additional ingredients
                    ingredientNR = len(missedIngredients)
                    f = open("recipe.txt", "a")
                    f.write("Ingredients:\n")
                    f.close()
                    f = open("recipe.txt", "a")
                    f.write(f"{item1}\n{item2}\n")
                    f.close()
                    for i in range(0, ingredientNR): # getting all additional ingridients in another for loop
                        ingredientn = missedIngredients[i]
                        f = open("recipe.txt", "a")
                        f.write(f"{ingredientn['name']}\n")
                        f.close()
                except IndexError:
                    f = open("recipe.txt", "a")
                    f.write("Sorry, this ingredient doesn't have that many recipes yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n")
                f.close()

                id = recipe['id']
                r1 = requests.get(f"https://api.spoonacular.com/recipes/{id}/analyzedInstructions?apiKey=e626af65626e433da41d815db5eb4f17")
                # making another API request to get the steps of the recipe
                allInfo = r1.json()
                jprint(allInfo) #uncomment this step for error handling
                try:
                    Steps = allInfo[0]
                    justSteps = Steps['steps']
                    stepNR = (len(justSteps))
                    f = open("recipe.txt", "a")
                    f.write("Steps:\n")
                    f.close()
                    for i in range(0, stepNR): #looping through the steps
                        stepn = justSteps[i]
                        f = open("recipe.txt", "a")
                        f.write(f"Step {i+1}: {stepn['step']}\n")
                        f.close()
                except IndexError:
                    # weirdly, some recipes don't contain steps, even though the first API call explicitly asks for recipes with instructions
                    f = open("recipe.txt", "a")
                    f.write("Sadly, this recipe does not have any steps yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n---\n \n")
                f.close()

        elif 'item1' in locals():
            r = requests.get(f"https://api.spoonacular.com/recipes/findByIngredients?apiKey=e626af65626e433da41d815db5eb4f17&ingredients={item1}&instructionsRequired=true&instructions&number=5")
            # api call searching for 5 recipes with instructions which include the item the model has recognized

            recipes = r.json()
            for i in range(0, 5): # looping through the recipes and handling the json data
                f = open("recipe.txt", "a")
                f.write(f"Recipe {i+1}:\n")
                f.close()
                try:
                    recipe = recipes[i]
                    name = recipe['title'] # getting the recipe name
                    f = open("recipe.txt", "a")
                    f.write(f"{name}\n")
                    f.close()
                    missedIngredients = recipe['missedIngredients'] # getting additional ingredients
                    ingredientNR = len(missedIngredients)
                    f = open("recipe.txt", "a")
                    f.write("Ingredients:\n")
                    f.close()
                    f = open("recipe.txt", "a")
                    f.write(f"{item1}\n")
                    f.close()
                    for i in range(0, ingredientNR): # getting all additional ingridients in another for loop
                        ingredientn = missedIngredients[i]
                        f = open("recipe.txt", "a")
                        f.write(f"{ingredientn['name']}\n")
                        f.close()
                except IndexError:
                    f = open("recipe.txt", "a")
                    f.write("Sorry, this ingredient doesn't have that many recipes yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n")
                f.close()

                id = recipe['id']
                r1 = requests.get(f"https://api.spoonacular.com/recipes/{id}/analyzedInstructions?apiKey=e626af65626e433da41d815db5eb4f17")
                # making another API request to get the steps of the recipe
                allInfo = r1.json()
                # jprint(allInfo) #uncomment this step for error handling
                try:
                    Steps = allInfo[0]
                    justSteps = Steps['steps']
                    stepNR = (len(justSteps))
                    f = open("recipe.txt", "a")
                    f.write("Steps:\n")
                    f.close()
                    for i in range(0, stepNR): #looping through the steps
                        stepn = justSteps[i]
                        f = open("recipe.txt", "a")
                        f.write(f"Step {i+1}: {stepn['step']}\n")
                        f.close()
                except IndexError:
                    # weirdly, some recipes don't contain steps, even though the first API call explicitly asks for recipes with instructions
                    f = open("recipe.txt", "a")
                    f.write("Sadly, this recipe does not have any steps yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n---\n \n")
                f.close()

        else:
            r = requests.get(f"https://api.spoonacular.com/recipes/findByIngredients?apiKey=e626af65626e433da41d815db5eb4f17&ingredients={item}&instructionsRequired=true&instructions&number=5")
            # api call searching for 5 recipes with instructions which include the item the model has recognized

            recipes = r.json()
            for i in range(0, 5): # looping through the recipes and handling the json data
                f = open("recipe.txt", "a")
                f.write(f"Recipe {i+1}:\n")
                f.close()
                try:
                    recipe = recipes[i]
                    name = recipe['title'] # getting the recipe name
                    f = open("recipe.txt", "a")
                    f.write(f"{name}\n")
                    f.close()
                    missedIngredients = recipe['missedIngredients'] # getting additional ingredients
                    ingredientNR = len(missedIngredients)
                    f = open("recipe.txt", "a")
                    f.write("Ingredients:\n")
                    f.close()
                    f = open("recipe.txt", "a")
                    f.write(f"{item}\n")
                    f.close()
                    for i in range(0, ingredientNR): # getting all additional ingridients in another for loop
                        ingredientn = missedIngredients[i]
                        f = open("recipe.txt", "a")
                        f.write(f"{ingredientn['name']}\n")
                        f.close()
                except IndexError:
                    f = open("recipe.txt", "a")
                    f.write("Sorry, this ingredient doesn't have that many recipes yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n")
                f.close()

                id = recipe['id']
                r1 = requests.get(f"https://api.spoonacular.com/recipes/{id}/analyzedInstructions?apiKey=e626af65626e433da41d815db5eb4f17")
                # making another API request to get the steps of the recipe
                allInfo = r1.json()
                # jprint(allInfo) #uncomment this step for error handling
                try:
                    Steps = allInfo[0]
                    justSteps = Steps['steps']
                    stepNR = (len(justSteps))
                    f = open("recipe.txt", "a")
                    f.write("Steps:\n")
                    f.close()
                    for i in range(0, stepNR): #looping through the steps
                        stepn = justSteps[i]
                        f = open("recipe.txt", "a")
                        f.write(f"Step {i+1}: {stepn['step']}\n")
                        f.close()
                except IndexError:
                    # weirdly, some recipes don't contain steps, even though the first API call explicitly asks for recipes with instructions
                    f = open("recipe.txt", "a")
                    f.write("Sadly, this recipe does not have any steps yet.\n")
                    f.close()
                f = open("recipe.txt", "a")
                f.write(" \n---\n \n")
                f.close()

    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()