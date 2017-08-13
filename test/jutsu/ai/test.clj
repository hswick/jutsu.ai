(ns jutsu.ai.test
  (:require [clojure.test :refer :all]
            [jutsu.ai.core :as ai]))

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

(defn train-regression-net []
  (let [dataset-iterator (ai/regression-csv-iterator "data.csv" 150 1)]
    (-> (ai/regression-net 1 1)
        (ai/initialize-net!)
        (ai/train-net! 100 dataset-iterator)
        (ai/save-model "regnet.zip"))))

(train-regression-net)

(defn train-classification-net []
  (let [dataset-iterator (ai/classification-csv-iterator "classification_data.csv" 4 4 10)]
    (println dataset-iterator)
    (-> (ai/classification-net 4 1)
        (ai/initialize-net!)
        (ai/train-net! 10 dataset-iterator)
        (ai/save-model "classnet.zip"))))

(train-classification-net)

(def test-net (ai/network [{:in 4 :out 4 :activation :relu}
                           {:in 4 :out 4 :activation :relu}
                           {:in 4 :out 4 :activation :identity}]
                          (ai/default-classification-options)))

(defn new-train-classification-net []
  (let [dataset-iterator (ai/classification-csv-iterator "classification_data.csv" 4 4 10)]
    (-> test-net
        (ai/initialize-net!)
        (ai/train-net! 10 dataset-iterator)
        (ai/save-model "classnet.zip"))))
        
