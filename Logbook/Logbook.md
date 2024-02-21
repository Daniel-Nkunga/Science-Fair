# Daniel Nkunga's Scientific Logbook
 
## ISEF Science Fair 2023 - Building AI Lip Reading Models

### Description
This is a Science Fair project aiming to build AIs that are able to **read lips**. This will be done by multiple methods which will be compared against each other for accuracy, speed, and readability. AI Models will be trained by by these models:
    
    Brute Force: the AI will be given hours of videos to train on and come up with its own conclusions on how to read lips. Its goal is both accuracy and speed, but it will probably get more caught up on a person's stutters/bad speech habits.
    
    Human Based Model: the AI will be trained to read lips more like a human, focusing on getting the theme of the message over the exact words a person is saying. The AI will be given a word of context and told to put more emphasis on eyebrows, eyes, nose, etc. to get more of a gist of what a person is saying. Ideally, generative AI would be used to create these phrases and build text from the context given.4

### Please Note
From the start of this project to September 28th, this logbook was not kept. A prelimenary attempt was made to keep a logbook at the start of this project that is under **Nkunga - Scientific Notebook - OUT OF DATE.pdf** in the folder **Logbook**. This was not kept up to date when progress was made more on the laptop as oposed to tangible rsearch reports. 

## Early Research

### Initial Project Research
#### August 22nd - August 24th 2023
The first two days of the project were spent researching background infomation on the Lip Reading AI that has existed in the past. An article from [Engadget](https://www.engadget.com/ai-is-already-better-at-lip-reading-that-we-are-183016968.html?guccounter=1) summarized an wide overview of the industry's advancements and methods in using AI to read lips. It ultimately led to the decision to make this project more comparative where different training methods would be used to compare accuracy with AIs with the two gained from this article being a brute force method and a more human based method.  
The brute force method would be just giving an AI hours of content with captions and telling it to generate captions of other videos. There would be minimum input or aid on the part of the researcher. The human based method would be based on the fact that when humans read lips, they focus more on getting the main concept of what's being said as opposed to verbatim translations of lip movements to words. The goal is to have a generative AI caption a video while focusing on a given prompt, eyebrow and nose movements, as well as lip movements.   
The article also referrenced that past researched was trained on hours of footage from the BBC that can be used for educational and research purposes. 

#### August 28th - *September 15th 2023
[These dates are not exact] 
The next phase of the process was used to set up a virtual machine to get [Dlib](http://dlib.net/face_landmark_detection.py.html), the program used to place facial recognition dots, up and running. Though not as complicated in hindsight, it took a long time to get it running initially and to place dots on faces.  
Afterwards, dots were isolated to only be over the nose and moth so training could be easier. A project could be made here so long as subjects were only allowed to face directly towards the camera but it was decided to project the dots onto a 3D shape to better track the dots on an object moving in 3D. 

![Dlib facial landmarks on Daniel's Face - September 28, 2023](/Logbook/Images/Facial_Landmarks_Initial.jpg)

    This is an image of Dlib's facial landmarks used to track across Daniel Nkunga's face. This can also be used to track video if video is fed instead of the webcam feed.  
![Dlib facial landmarks isolated across only mouth and nose - September 28, 2023](/Logbook/Images/Facial_Landmarks_Isolated.jpg)

    This is an image of Dlib's facial landmarks used to track across Daniel Nkunga's face. In this case, only dots between 28 and 35 or 49 and 68 are shown to isolate the mouth and nose.

#### September 20th - September 21st 2023
Corresponded with Rob Cooper from the BBC to aquire footage for later training AI. This footage has been used before to train an AI model to read lips for Lipnet's lip reading model. Aquiring and using require a form and is valid for a year.  
Because using the footage requires extra citations to use in research environments, most photos and videos in this research will be done by Daniel or other volunteers as opposed to showing the BBC footage. The footage will still be used to train the AI however. The footage can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). 

#### September 25th - September 28th 2023
Working on updating environment to run Ostadabbas's code of projecting the Dlib facial landmarks onto a 3D model. Despite earlier notes stating that it was understood how to make environments, it is still not a smooth process to get them running. 

#### September 28th 2023
Created this logbook folder. All capstone project reports were uploaded, images were made, and the previous attempt at the Scientific Notebook was uploaded. Daniel is tired. Images are captioned using indents because the markdown that's meant to caption doesn't atually result in anything meaning it probably is for those with visual impairments. More professional logbook layouts will be looked into at a later date.

#### October 3rd 2023
Progress Reprot. Created new environment on Makerspace Desktop to (eventually) run training data on called NkuSciFair. Data does not come in a .mp4, .mov, or any of the other common video formats so accessesing it needs to be explored more. 


#### October 5th 2023
Cleaned on local Science Fair Python Environment of extra downloads. Created a new environment explicitly for testing Ostadabbas's code called 3DLMTesting [standing for 3D landmak testing]. Submitted logbook report 2. 

#### October 12th 2023
Set up conda environments on local device and makerspace desktop. eos-py was installed at the most recent version on both (1.4.0) which was not the recommended 1.0.1. Hopefully not an issue later. All programs except landmarks_detection_video.py ran without errors. Not all of them had an output.

#### October 16th 2023
Working on creating own program to use the 3D landmarks from Ostadabbas to project on a live camera feed. File is under Initial Solo Testing Called 3DLMTesting.py


#### October 24th 2023
Editing the Ostadabbas Code to not use Euler andgles. Creating new folder titles 3DLMTesting in Initial Solo Testing where test can be contained without changing original code. Issue comes from the Euclidian angles which are not working the way they are meant to. This can be solved two ways: 1) fix the version of Dlib and eos to match the recomended version in the README or 2. figure out how euler coordinates work and figure out a way to output them. Problem with one is that it would force me to manually make the wheels for eos which would require me to download and us Visual Studio (which is now in process) and work in C (I do not know how to do that). The problem with two is that it would require me to look into how to give the code a Euler coordinate. I don't know how to do that. Thouch I am looking into it. Might need to ask Monson a brief rundown of how Euler coordinates work. 
![Console error message due to invalid Euler numbers - October 24, 2023](/Logbook/Images/ConsoleError.jpg)


    Just an image of the error message I'm getting when tryin to run the program "landmark_detection_video.py"

#### Editor's Note
I started this project as a GitHub novice and initially grew to dislike it over the course of this project in particular. This was especially true becuase I felt like even though I was trying to make progress, I never seem to get any results. This led to me not committing for weeks at a time. On new years day, I lost my previous laptop meaning that the weeks I spent saving locally were gone. The next entries will no longer follow the day by day results and instead will summarize days/weeks works of progress at once.

#### Late October through Early November 2023 (25th - 15th)
Still trying to get Ostadabbas Facial Landmarks to work but cannot get the environment set up properly. I looks like an error with how the conda environment was set up. Though there were instructions on how to set up eos on the GitHub page, I don't think I followed them properly. 

#### Early to Mid November 2023 (15th - 27th)
For a mental switch, I started recording coodinates for training a future model beucase that would just require dlib. The chosen focus points were just the lips, the lips and nose, and all coordinates. A total of 150 vidoes were recorded. 50 yes videos, 50 no videos, 25 nice videos, and 25 yell videos. During this time, the first Openness.py program was created. Openness.py would be used to trim videos by detecting when the mouth was first open. Though it worked in theory, this is where the limits of Dlib were first brought into focus. Many videos didn't have valid starting/ending frames becuase dlib isn't meant to accuratly capture movements of the lips and is just meant to capture the face. This meant it would often get get locked off the face and never a return an open or closed position. 

#### Late November to Early December (27th - 15th)
The end of the month was spent getting forms prepared. As implied earlier, VideoTrimmer.py was created and run at this time to trim training videos to start when the mouth first open and end when the program was last open. At the end of this period, the first coordinates were also returned for the training videos as .csv files. 

#### Late November to Early January (15th - 15th)
During this time was winter break. I didn't work much over this time but as stated earlier, my laptop battery died during this time. 

#### Late January (15th- 30th)
After getting a new laptop and realizing how much progress was lost, I started creating a dlib environment again. Because it was a lot less of a struggle than I thought it was gonna be and I had gotten a new laptop with a lot of processing power, I also decided to attempt to set up a Media Pipe environment. Media Pipe was my inital plan for this project by my instructor and I were afraid of the computational cost. The environemnt was also relativly simple to set up after all the struggle early last semester and after testing, I realized I had to pivot to using it. Media Pipe not only had more landmarks, an greater example of the use of my research, but landmarks that were able to properly follow lip movements. After brining it back up with my instructor again, we decided it would be a viable switch.

![Dlib facial landmarks on Daniel's Face - January 14th, 2024](/Logbook/Images/MediaPipeTesting.jpg)

Training videos were quickly rerecorded with the same amount of videos for each word. During this time, Openness.py and VideoTrimmer.py were recreated to work for MediaPipe this time and resulted in much better resutls. 

New focus points were chosen of the lips alone, lips and eyebrows, and all landmarks. This is because I couldn't find documentation of other prominent landmarks and the ones chosen should still be relevant for lip reading. 

At this point, the project was also going to flatten the image to get a cannonical face image but ultimately I couldn't figure it out and had to abandon that part of the project. Future research should look into flattening imagess to this format instead of/in conjuction with  refocusing the image.

Coordinates were also extracted from these files with issues only arising with models needing all datapoints. Each file would have about 35,000 data points and no matter what I did, my machine would only do a random spread of fifty without me manually telling it to get data from certain files. Though this was less of an issue for "All" data points, "All Shifted" proved to be even more of a struggle as there were more computations going on in the background that the laptop still refused to do even when explictly told to.

Though there was an attempt to esablish a local Tensor Flow environment on my laptop, I quickly realized I didn't want to (and didn't have the time to) struggle to create one. Google Colab was used to create TensorFlow environments and after a few nights of stuggle the first models were trained. 

More issues arrived once again when it came to the "All Models." Runtime errors kept on occurring with Google Colab not wanting to process millions of lines of data points and I couldn't easily build their models in the same way I built the other models. Ultimatly, I had to brute force their creation using Keras but the process furhter proved the point of this research of needing to limit the information being processed to only relevant information.

![QR Code to project GitHub](/Logbook/Images/Github.jpg)