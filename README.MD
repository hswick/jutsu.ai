# jutsu.ai

Clojure wrapper for deeplearning4j with some added syntactic sugar.

What if I told you that you could do machine learning on the JVM without wanting to cry or 
set your hair on fire? The goal of this library is to be the most syntactically elegant machine learning library for Clojure/Java ecosystem. jutsu.ai uses Clojure specific code idioms, like Data as Code, to create an aesthetic and declarative api.

The best of all, no more snakes!


![alt text](http://wp.patheos.com.s3.amazonaws.com/blogs/wildhunt/files/2011/03/saintpatrick.jpg)

If you have any suggestions or fixes please feel free to submit a PR or post an issue!

# Usage

Install jutsu.ai by including this in your project dependencies:

```clojure
[hswick/jutsu.ai "0.0.11"]
```

Here is a toy example demonstrating classifying the well known iris dataset.

```clojure
;;Example configuration of a simple multilayer feedforward neural network architecture
;;classifying the iris dataset.
(def iris-net-config [:layers  [[:dense [:n-in 4 :n-out 10 :activation :relu]]
																[:dense [:n-in 10 :n-out 10 :activation :relu]]
																[:output :negative-log-likelihood [:n-in 10 :n-out 3
					  							   				   														 :activation :softmax]]])

(def iris-net (-> iris-net-config
									ai/network-config
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
# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

## License

Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
