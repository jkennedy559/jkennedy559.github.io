---
layout: post
title:  "European Scrapin'"
date:   2020-05-17 21:03:36 +0530
---


BJJ is my drug of choice and while I can't train in lockdown I'm watching alot of old match videos. I found myself looking for results data from the most recent European Championships in the brown belt division to no avail. The IBJJF, the federation that run the major tournaments, release details of who medaled by belt & devision but provide no match level results. I decided to have a go at trying to programmatically piece results together.

Firstly I did a recon on what was available. Flograppling have a subscription service that has video recordings of all matches from major competitions and I can subset matches by belt. I scraped these records and extracted competitor's names, match duration and a reference to the video's url. So I know all the brown matches that took place.

Flograppling don't provide a breakdown by weight division but I did find registrations [here][1] from an IBJJF website and by joining on competitor name & belt I can map weight division to competitor. I can now have good go at classifying match winners and losers given I know the eventual division winners and can classify brackets with mapped competitor weights. 

[1]: https://www.ibjjfdb.com/ChampionshipResults/1415/PublicRegistrations?lang=en-US

That's a start but it would be great to get more granular information.

A match victor is decided by either points or submission. At brown belt matches last for a maximum of 8 minutes or until one competitor submits the other. From the video duration I already extracted I can tell if there has been a submission and I've already classified the winner & loser so I know who scored it.

The match video has a scoreboard with the points tally for each competitor displayed so I signed up for a trial account at Flograppling and built a scraper [here][3] that logs in, brings up the match and takes a screenshot at the end.

<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
</style>

<img src="/images/match_screen.png" alt="Drawing" style="width: 400px;" class="center"/>

I can crop the screenshot to extract the score board.

<img src="/images/scoreboard.png" alt="Drawing" style="width: 400px;"/>

I didn't fancy watching all 500 matches to manually label the score so I needed a method to programmatically extract the points scored from the screenshot. My first attempt involved Pytesseract but it didn't perform well. My next thought was to use a digit classification model.

I found a convolutional neural network implemented with Tensorflow & Keras with > 99% test set accuracy trained on the MNIST dataset. The MNIST dataset is preloaded in Tensorflow and a model training pipeline is provided in one of the example tutorials in the package documentation.

If I process the cropped points images to be similar to the MNIST training examples the model should perform with similar accuracy.

<img src="/images/MNIST_example.png" alt="Drawing" style="width: 400px;"/> 
<img src="/images/cropped_points.png" alt="Drawing" style="width: 400px;"/>  



I took a sample of 90 examples and manually labeled them on a csv to test the model. Not all the scraped screenshots captured the scoreboard adequately for classification and on one of 90 samples there was a double digit score so I excluded these which left me with 87 test examples.

The model needs grayscale images with 28x28 resolution, digits must be roughly centered in white against a black background - I used OpenCV for this task. 

My first attempt at modeling used a binary threshold & image erosion to make the digits appear thiner against the black background.

<img src="/images/eroded_points.png" alt="Drawing" style="width: 400px;"/> 

The model classified 85% correctly but was systemically misclassifying 6s as 5s.

<img src="/images/errors_cnn.png" alt="Drawing" style="width: 400px;"/>

In the end the best results I got involved just thresholding the cropped points image. The model misclassified one example and given three examples were excluded that's 86/90 correct. The notebook is available [here][2].

<img src="/images/final_classification.png" alt="Drawing" style="width: 400px;"/>

The model can't classify double digit scores, I would need an object detector to split each digit for classification separately. I amended the prediction threshold so the model only classifies when it's very certain, the remaining examples I resigned to label manually.

Overall this is a decent attempt to classify the points tally, there will be some errors but it should get the vast majority correct.

[2]: https://github.com/jkennedy559/bjj_data/blob/master/notebook.ipynb
[3]: https://github.com/jkennedy559/bjj_data/blob/master/scrape.py


