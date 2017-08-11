(ns jutsu.ai.test
  (:require [clojure.test :refer :all]
            [jutsu.ai.core :as ai]))

(defn train-regression-net []
  (let [dataset-iterator (ai/regression-csv-iterator "rando.csv" 150 1)]
    (-> (ai/regression-net 1 1)
        (ai/initialize-net!)
        (ai/train-net! 100 dataset-iterator)
        (ai/save-model "bignet.zip"))))

(def topology1
  [{:in 1 :out 50 :activation :relu}
   {:in 50 :out 50 :activation :relu}
   {:in 50 :out 1 :activation :softmax}])

(def topology2
  [[1 50 :relu]
   [50 50 :relu]
   [50 1 :softmax]])

(deftest shorthand
  (is (= topology1 (ai/parse-shorthand topology2))))

;;throw error if now activation key
(defn interpret-activation [layer])

;;pass in shorthand with vectors
(defn parse-topology [topology]
  (let [topo (if (= clojure.lang.PersistentVector (class (first topology)))
               (parse-shorthand topology)
               topology)
        proper-topo (map interpret-activation topo)]))
