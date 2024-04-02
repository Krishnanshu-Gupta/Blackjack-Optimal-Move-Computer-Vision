## Using Computer Vision for Blackjack Optimal Move Recommendation

The project will use Computer Vision to classify and detect Playing Cards in a window as player or dealer cards with the face value, and then use Basic Strategy to recommend the optimal move for the player in every situation. The methodology involves using K-Means for image segmentation, card reprojection and feature extraction, training of the KNN classifier using a labeled dataset, and integration of the detection system into a Blackjack Basic Strategy recommendation algorithm. We also aimed to observe the effectiveness of this approach in detecting various card designs under different lighting conditions and occlusions.

### Access Research Paper
The entire research paper for this project, "Optimal Blackjack Strategy Recommender: A Comprehensive Study on Computer Vision Integration for Enhanced Gameplay", is available on Arxix, https://arxiv.org/abs/2404.00191. 

The paper lists all the work conducted: our assumptions, attempted strategies, results, conclusions, etc.

### Test Code
To test this code, simply execute recognition.py and provide the image path as an argument.
```
python3 recognition.py {image_path}
```

Sample Output for 6 different test images. The recommended move is listed in the bottom left of the images.

![](/output1.png)
![](/output2.png)
