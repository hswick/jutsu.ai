# jutsu.ai

Clojure wrapper around deeplearning4j with some added syntactic sugar.

What if I told you that you could do machine learning on the JVM without wanting to cry or 
set your hair on fire?

Meant to be the most syntactically simple machine learning library for Clojure/Java ecosystem.

No more Snakes!
http://wp.patheos.com.s3.amazonaws.com/blogs/wildhunt/files/2011/03/saintpatrick.jpg

jutsu.ai uses Clojure specific code idioms, like Data as Code, to create an
aesthetic and declarative api.

# Solid Foundations

DeepLearning4j 

It seems a wrapper for deeplearning4j evaded the clojure ecosystem like a white whale. But Alas! Thy beast has been conquered.

# Usage

```clojure
(def topology1
  [{:in 1 :out 50 :activation :relu}
   {:in 50 :out 50 :activation :relu}
   {:in 50 :out 1 :activation :softmax}])

;;In this case order matters
;;:in first
;;:out second
;;:activation third
(def topology2
  [[1 50 :relu]
   [50 50 :relu]
   [50 1 :softmax]])
   
(= topology1 (parse-shorthand topology2))
;;=>true
```
# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

## License

Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
