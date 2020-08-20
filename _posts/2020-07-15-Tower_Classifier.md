---
layout: post
title:  "Tower Classifier'"
date:   2020-07-15 21:03:36 +0530
---

# Tower Classifier

Like everyone else I have had six months of Covid imposed solitude but unlike others I've been extremely unemployed.

I've been through multiple fads - sour dough, circuits, The Sopranos, etc. Over the last few weeks I've occupied myself with producing a simple web application that classifies pictures of either the [CN Tower](https://en.wikipedia.org/wiki/CN_Tower), [Skytree](https://en.wikipedia.org/wiki/Tokyo_Skytree) or [The Space Needle](https://en.wikipedia.org/wiki/Space_Needle).

[Click here to use web app](https://lucid-sonar-280416.uc.r.appspot.com/)

The app is built with Django and deployed on Google Cloud App Engine. The classifier is a CNN deployed through Google Cloud Functions. The notebook [lesson1.ipynb](https://github.com/jkennedy559/course-v3/blob/master/assignments%20/lesson1.ipynb) shows the model training, using [fastai](https://www.fast.ai/) and the model can be found [here](https://github.com/jkennedy559/course-v3/blob/master/assignments%20/data/towers/export.pkl).