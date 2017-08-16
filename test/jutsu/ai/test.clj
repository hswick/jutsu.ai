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

(def layer-config-test [{:in 4 :out 4 :activation :relu}
                        {:in 4 :out 4 :activation :relu}
                        {:in 4 :out 10 :activation :softmax}])

(def test-net (ai/network layer-config-test
               (ai/default-classification-options)))

(deftest init-classification-net
  (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork (class test-net))))

(defn train-classification-net []
  (let [dataset-iterator (ai/classification-csv-iterator "classification_data.csv" 4 4 10)]
    (-> test-net
        (ai/initialize-net!)
        (ai/train-net! 10 dataset-iterator)
        (ai/save-model "classnet.zip"))))

(train-classification-net)
