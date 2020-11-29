---
layout: post
title:  "Covid Capacity Calculator"
date:   2020-09-09 21:03:36 +0530
---

Dad works in the hospital and approached me with a headache concerning how to calculate the capacity of a room given its dimensions and a requirement of 2 meter socially distancing. 

![title](/images/circles_in_container.png)

A person in the centre of the room needs a 2 meter protective radius from anyone else. If the area of this circle = π * radius * 2, you would think you could divide total meters squared in the room by the area but that ignores that the full buffer wouldn't be required if the person had their back to the wall.

If there is a formula that solves this problem it's beyond me to come up with it so instead I codified the constraints and iteratively deduced the capacity through optimisation.

I consider the room as a set of x & y coordinates on a grid. A person can occupy a set of coordinates if and only if their 2 meter protective bubble doesn't contain another person. By adding more people to the room iteratively until no more can be added safely I can calculate the capacity.

Code can be seen [here](https://github.com/jkennedy559/Room-Capacity-Calculator) and once I got it working I built a app with [Dash](https://dash.plotly.com/) that visualises the room and deployed it on Google Cloud Platform.

[Click here to use web app](https://room-capacity-599.ew.r.appspot.com/)

