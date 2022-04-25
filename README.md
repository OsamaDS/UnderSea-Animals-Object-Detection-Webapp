# UnderSea-Animals-Object-Detection-Webapp

## Dataset

I choose dataset from RoboFlow. This is Aquarium Dataset. The link of dataset is 
https://public.roboflow.com/object-detection/aquarium

## Model

I built complete end to end object detection Web App. The App is built on Flask framework. When you will run the server then
you only need to upload the image of animals e.g fishes and our model will predict its location and classify it. There is 
Sample_images folder in the repo, In this repo there are testing images on which you can test this model.

## How To Run?

You need to clone the Repo. After Cloning the repo you need to make your project environment. After that you need to write
following command in the terminal to install all packages which are needed for this project.

        pip install -r requirements.txt

When all packages will be installed, then just write follwing command in the terminal to run the local server
    
        python clientApp.py
