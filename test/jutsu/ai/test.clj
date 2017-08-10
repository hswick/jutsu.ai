(ns jutsu.ai.test
  (:require [clojure.test :refer :all]
            [jutsu.ai.core :as ai]))

(defn train-regression-net []
  (let [dataset-iterator (ai/regression-csv-iterator "rando.csv" 150 1)]
    (-> (ai/regression-net 1 1)
        (ai/initialize-net!)
        (ai/train-net! 100 dataset-iterator)
        (ai/save-model "bignet.zip"))))
