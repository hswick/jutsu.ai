# jutsu.ai

Clojure wrapper around deeplearning4j with some added syntactic sugar

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
