# jutsu.ai

Clojure wrapper for deeplearning4j with some added syntactic sugar.

What if I told you that you could do machine learning on the JVM without wanting to cry or
set your hair on fire? The goal of this library is to be the most syntactically elegant machine learning library for Clojure/Java ecosystem. jutsu.ai uses Clojure specific code idioms, like Data as Code, to create an aesthetic and declarative api.

jutsu.ai makes use of my laziest but cleverest achievement which is that the network configuration translates clojure keywords into java method calls. The implication of this is that the majority of the config options provided by deeplearning4j are accessible with this library.

The best of all, no more snakes!

![alt text](http://wp.patheos.com.s3.amazonaws.com/blogs/wildhunt/files/2011/03/saintpatrick.jpg)

Currently this project is a work in progress. However, no drastic API changes are expected. Configurations have been tested on basic feedforward nets, RNN's, and CNN's. If you can't seem to find a configuration feature you are looking for submit an issue and I'll do my best to help.

If you have any suggestions or fixes please feel free to submit a PR as well!

Currently need help documenting the different configuration options. However, deeplearning4j has a very [good resource](https://deeplearning4j.org/glossary) that I recommend checking out.

# Usage

Install jutsu.ai by including this in your project dependencies:

```clojure
[hswick/jutsu.ai "0.1.4"]
```

You will also need to include a ND4J backend:

For CPU:

```clojure
[org.nd4j/nd4j-native-platform "1.0.0-alpha"]
```

For GPU:

```clojure
[org.nd4j/nd4j-cuda-8.0-platform "1.0.0-alpha"]
```

Here is a toy example demonstrating classifying the well known iris dataset.

```clojure
(ns skynet.core
  (:require [jutsu.ai.core :as ai]))

;;Example configuration of a simple multilayer feedforward neural network architecture
;;classifying the iris dataset.
(def iris-net-config
	[:layers 
		[[:dense [:n-in 4 :n-out 10 :activation :relu]]
	 	 [:dense [:n-in 10 :n-out 10 :activation :relu]]
	 	 [:output :negative-log-likelihood [:n-in 10 :n-out 3 :activation :softmax]]]])

(def iris-net
	(-> iris-net-config
		ai/network
		ai/initialize-net))

(defn iris-train []
  (let [iris-iterator (ai/classification-csv-iterator "iris.csv" 150 4 3)]
    (-> iris-net
	(ai/train-net! 200 iris-iterator)
	(ai/save-model "iris-model"))
  (.reset iris-iterator)
  (println (ai/evaluate iris-net iris-iterator))))

(iris-train)
```

# Documentation

Make sure to take a look at the [wiki](https://github.com/hswick/jutsu.ai/wiki) for more documentation.

Here is a [video tutorial](https://www.youtube.com/watch?v=yR_uy8fS52s) I made demonstrating the use of jutsu.ai.

## Additional Resources

Here is a list of libraries that I have created to be used with jutsu.ai:

* [jutsu.matrix](https://github.com/hswick/jutsu.matrix) Clojure wrapper around ND4J to make it easier to work with the underlying matrices in DL4J.
* [jutsu.nlp](https://github.com/hswick/jutsu.nlp) Clojure wrapper around DL4J's NLP features. Provides useful abstractions for Word Vectors.
* [jutsu.ai.ui](https://github.com/hswick/jutsu.ai.ui) GUI for visualizing DL4J training
* [jutsu](https://github.com/hswick/jutsu) Simple web based graphing tool for Clojure

Here is a list of libraries that I've found useful when working with jutsu.ai:

* [claypoole](https://github.com/TheClimateCorporation/claypoole) Juiced up clojure functions like pmap. Very useful when working with big data sets.
* [tablesaw](https://github.com/jtablesaw/tablesaw) Table slicing tool for Java. Nice API and extremely performant.


# Dev

Run `boot night` to startup nightlight and begin editing the project in a browser.

# Contributing

When changing project dependencies or project version number, make sure to update this in the build.boot file.
The Travis build uses boot, and will not pass the checks unless you update it.

The project.clj file is simple there to make it easier for other developers to contribute.

## License

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
