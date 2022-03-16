# LeftoverAI

Leftover AI on the NVIDIA Jetson Nano is a simple project to get rid of leftover food from your fridge or shelves. Using computer vision, the program recognizes the ingredients. You can put up to three ingredients in the ingredient list by simple pressing a, b or c on your keaboard. When everything is ready, simply press p to get a txt-file with five recipes containing your ingredients. A fun and delicious way to use up your leftovers.

The model is pre-trained and adjusted to my use case through transfer learning. 200 - 400 pictures of an ingredient were sufficient to get reliable results. Because of the differences in vegetables and fruits it is helpful to train on more than one vegetable/fruit to get the desired reliability.

Hardware for this project:

NVIDIA Jetson Nano Dev Kit
+ charging cable, Wi-Fi antennas, acrylic case, SD Card and cooling fan
RPi Cam V2
+ case and long cable
Keayboard and mouse

Special thanks to Paul McWhorter for his very useful and educational videos.

Link to a short video showcasing the project: https://vimeo.com/687612483
