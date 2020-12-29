# Learn Tensor Flow

## Overview

The goal of the project is to pick up some basic machine learning skills while also writing a simple AI chess bot. The idea originates from this summer when I attempted to learn the game of chess via an app. While playing in the app was both enjoyable and educational, and I learned a little about the basics of chess, I found playing the computer AI to be frustrating. It would often, even at the 'intermediate' level, make several ridiculously terrible moves in a row and then follow up with a several piece combo. This is a pretty uninspiring opponent, since I can match neither it's ineptitude nor cleverness. 

### The Idea

The idea then is to train a neural net on a set of games where the players in each games are within fixed ELO ranges. The idea being that one should be able to come up with models which accurately predict moves that a player of that 'skill-level' would make. Ideally, this would create a compelling and fun AI for players of any skill to compete with. That's the idea anyways. 

### The Implementation

I've decided to build the project in Node.js. Ideally the end result would be a node back end with precomputed models and a simple chess React front end. I'll be using Tensorflow.js, since this seems to be the modern way of doing these things. 

## The Beginning

I don't know any ML. I started by following a (very basic) guide I found on Tensorflow's website which computes a linear model that predicts MPG from horsepower based on about 200 datapoints provided in a static resource. This uses Tensorflow in the browser to compute these values.

### Extension 1

I was able to extend this example by also considering the number of cylinders, as this data was provided in the original data set and also seemed like it might be an indicator of the MPG of a vehicle. I was also able to get a model which was nonlinear, a massive improvement.

### Extension 2 (Nodification)

I'd like to convert this to a Node.js project from a simple browser script. We'll see how it goes. The idea here being that I could expose the following endpoints:
  * Compute models. Saves new models to existing folder. Overwrites existing model.
  * Get MPG estimate. For a HP, Cylinder pair, return the predicted MPG.
