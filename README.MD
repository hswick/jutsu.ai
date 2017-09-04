# jutsu.ai

Clojure wrapper for deeplearning4j with some added syntactic sugar.

What if I told you that you could do machine learning on the JVM without wanting to cry or 
set your hair on fire? The goal of this library is to be the most syntactically elegant machine learning library for Clojure/Java ecosystem. jutsu.ai uses Clojure specific code idioms, like Data as Code, to create an aesthetic and declarative api.

The best of all, no more snakes!


![alt text](http://wp.patheos.com.s3.amazonaws.com/blogs/wildhunt/files/2011/03/saintpatrick.jpg)

If you have any suggestions or fixes please feel free to submit a PR or post an issue!

# Usage

```clojure
;;Example configuration of a simple multilayer feedforward neural network architecture
;;classifying the iris dataset.
(def iris-net-config [:layers  [[:dense [:n-in 4 :n-out 10 :activation :relu]]
				[:dense [:n-in 10 :n-out 10 :activation :relu]]
				[:output [:n-in 10 :n-out 3
					  :activation :softmax
					  :loss :negative-log-likelihood}]]])

(def iris-net (-> iris-net-config
		network-config
		initialize-net))

(defn iris-train []
  (let [iris-iterator (ai/classification-csv-iterator "iris.csv" 150 4 3)]
    (-> iris-net
	(ai/train-net! 200 iris-iterator)
	(ai/save-model "iris-model"))
  (.reset iris-iterator)
  (println (ai/evaluate network iris-iterator))))
  
(iris-train)
    
(->> (m/matrix [7.1 3.0 5.9 2.1])
	(ai/output iris-net)
	(m/max-index)
	(println))
```
# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

## License

Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
